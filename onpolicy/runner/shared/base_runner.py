import os
from typing import Dict

import numpy as np
import torch
import wandb
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter  # tensorboardX to work with macos

from onpolicy.utils.graph_buffer import GraphReplayBuffer
from onpolicy.utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
	"""Convert torch tensor to a numpy array."""
	return x.detach().cpu().numpy()

class Runner(object):
	"""
	Base class for training recurrent policies.
	:param config: (dict) Config dictionary containing parameters for training.
	"""
	def __init__(self, config:Dict):

		self.all_args = config['all_args']
		self.envs = config['envs']
		self.eval_envs = config['eval_envs']
		self.device = config['device']
		self.num_agents = config['num_agents']
		# total entites is agents + goals + obstacles
		self.num_entities = (self.num_agents + self.num_agents + 
							self.all_args.num_obstacles)
		self.num_obstacles = self.all_args.num_obstacles
		if config.__contains__("render_envs"):
			self.render_envs = config['render_envs']       

		# parameters
		self.env_name = self.all_args.env_name
		self.algorithm_name = self.all_args.algorithm_name
		self.experiment_name = self.all_args.experiment_name
		self.use_centralized_V = self.all_args.use_centralized_V
		self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
		self.num_env_steps = self.all_args.num_env_steps
		self.episode_length = self.all_args.episode_length
		self.n_rollout_threads = self.all_args.n_rollout_threads
		self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
		self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
		self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
		self.hidden_size = self.all_args.hidden_size
		self.use_wandb = self.all_args.use_wandb
		self.use_render = self.all_args.use_render
		self.recurrent_N = self.all_args.recurrent_N

		# interval
		self.save_interval = self.all_args.save_interval
		self.use_eval = self.all_args.use_eval
		self.eval_interval = self.all_args.eval_interval
		self.log_interval = self.all_args.log_interval

		# dir
		self.model_dir = self.all_args.model_dir

		# if not testing model
		if not self.use_render:
			if self.use_wandb:
				self.save_dir = str(wandb.run.dir)
				self.run_dir = str(wandb.run.dir)
			else:
				self.run_dir = config["run_dir"]
				self.log_dir = str(self.run_dir / 'logs')
				if not os.path.exists(self.log_dir):
					os.makedirs(self.log_dir)
				self.writter = SummaryWriter(self.log_dir)
				self.save_dir = str(self.run_dir / 'models')
				if not os.path.exists(self.save_dir):
					os.makedirs(self.save_dir)

		if self.all_args.env_name == "GraphMPE":
			from onpolicy.algorithms.graph_mappo import GR_MAPPO as TrainAlgo
			from onpolicy.algorithms.graph_MAPPOPolicy import \
				GR_MAPPOPolicy as Policy
		else:
			from onpolicy.algorithms.mappo import R_MAPPO as TrainAlgo
			from onpolicy.algorithms.MAPPOPolicy import R_MAPPOPolicy as Policy

		# NOTE change variable input here
		if self.use_centralized_V:
			share_observation_space = self.envs.share_observation_space[0]
		else:
			share_observation_space = self.envs.observation_space[0]

		# policy network
		if self.all_args.env_name == 'GraphMPE':
			self.policy = Policy(self.all_args,
								self.envs.observation_space[0],
								share_observation_space,
								self.envs.node_observation_space[0],
								self.envs.edge_observation_space[0],
								self.envs.action_space[0],
								device=self.device)
		else:
			self.policy = Policy(self.all_args,
								self.envs.observation_space[0],
								share_observation_space,
								self.envs.action_space[0],
								device = self.device)

		if self.model_dir is not None:
			print(f'Restoring from checkpoint stored in {self.model_dir}')
			self.restore()
			self.gif_dir = self.model_dir

		# algorithm
		self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
		
		# buffer
		if self.all_args.env_name == "GraphMPE":
			self.buffer = GraphReplayBuffer(self.all_args, 
											self.num_agents,
											self.envs.observation_space[0],
											share_observation_space,
											self.envs.node_observation_space[0],
											self.envs.agent_id_observation_space[0],
											self.envs.share_agent_id_observation_space[0],
											self.envs.adj_observation_space[0],
											self.envs.action_space[0])
		else:
			self.buffer = SharedReplayBuffer(self.all_args,
											self.num_agents,
											self.envs.observation_space[0],
											share_observation_space,
											self.envs.action_space[0])

	def run(self):
		"""Collect training data, perform training updates, and evaluate policy."""
		raise NotImplementedError

	def warmup(self):
		"""Collect warmup pre-training data."""
		raise NotImplementedError

	def collect(self, step):
		"""Collect rollouts for training."""
		raise NotImplementedError

	def insert(self, data):
		"""
		Insert data into buffer.
		:param data: (Tuple) data to insert into training buffer.
		"""
		raise NotImplementedError
	
	@torch.no_grad()
	def compute(self):
		"""Calculate returns for the collected data."""
		raise NotImplementedError
	
	# def train(self):
	# 	"""Train policies with data in buffer. """
	# 	self.trainer.prep_training()
	# 	train_infos = self.trainer.train(self.buffer)      
	# 	self.buffer.after_update()
	# 	return train_infos
	def train(self,active_agents=None):
		"""Train policies with data in buffer. """
	 
		self.trainer.prep_training()
		if active_agents!=None:
			train_infos = self.trainer.train(self.buffer,active_agents=active_agents, accumulation_steps=1)      
		else:
			train_infos = self.trainer.train(self.buffer, accumulation_steps=1)      
			  
		return train_infos

	def save(self):
		"""Save policy's actor and critic networks."""
		policy_actor = self.trainer.policy.actor
		torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
		policy_critic = self.trainer.policy.critic
		torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

	def restore(self):
		"""Restore policy's networks from a saved model."""
		# for name, param in self.policy.actor.named_parameters():
		# 	print(f"{name}: {param.shape}")
		# print("thanks")
		policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt', 
									map_location=torch.device('cpu'))
		self.policy.actor.load_state_dict(policy_actor_state_dict)
		# for name, param in self.policy.actor.named_parameters():
		# 	print(f"{name}: {param.shape}")
		if not self.all_args.use_render:
			policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt', 
										map_location=torch.device('cpu'))
			self.policy.critic.load_state_dict(policy_critic_state_dict, strict=False)
	
	def process_infos(self, infos):
		"""Process infos returned by environment."""
		env_infos = {}
		for agent_id in range(self.num_agents):
			idv_rews = []
			dist_goals, time_to_goals, min_times_to_goal,distance_mean, distance_variance, mean_by_cov = [], [], [], [], [], []
			idv_collisions, obst_collisions = [], []
			dists_traveled_list = []
			time_taken_list = []
			formation_distances = []
			time_mean_list, time_variance_list, time_mean_by_stddev_list = [], [], []
			for info in infos:
				if 'individual_reward' in info[agent_id].keys():
					idv_rews.append(info[agent_id]['individual_reward'])
				if 'Dist_to_goal' in info[agent_id].keys():
					dist_goals.append(info[agent_id]['Dist_to_goal'])
				if 'Time_req_to_goal' in info[agent_id].keys():
					times = info[agent_id]['Time_req_to_goal']
					if times == -1:
						times = self.all_args.episode_length * self.dt    # NOTE: Hardcoding `dt`
					time_to_goals.append(times)
				if 'Num_agent_collisions' in info[agent_id].keys():
					idv_collisions.append(info[agent_id]['Num_agent_collisions'])
				if 'Num_obst_collisions' in info[agent_id].keys():
					obst_collisions.append(info[agent_id]['Num_obst_collisions'])
				if 'Min_time_to_goal' in info[agent_id].keys():
					min_times_to_goal.append(info[agent_id]['Min_time_to_goal'])
				if 'Distance_mean' in info[agent_id].keys():
					distance_mean.append(info[agent_id]['Distance_mean'])
				if 'Distance_variance' in info[agent_id].keys():
					distance_variance.append(info[agent_id]['Distance_variance'])
				if 'Mean_by_variance' in info[agent_id].keys():
					# print("mean by variance is", info[agent_id]['Mean_by_variance'])
					mean_by_cov.append(info[agent_id]['Mean_by_variance'])
				if 'Dists_traveled' in info[agent_id].keys():
					dists_traveled_list.append(info[agent_id]['Dists_traveled'])
				if 'Time_taken' in info[agent_id].keys():
					time_taken_list.append(info[agent_id]['Time_taken'])
				if 'Formation_dist' in info[agent_id].keys():
					formation_distances.append(info[agent_id]['Formation_dist'])
				if 'Time_mean' in info[agent_id].keys():
					time_mean_list.append(info[agent_id]['Time_mean'])
				if 'Time_stddev' in info[agent_id].keys():
					time_variance_list.append(info[agent_id]['Time_stddev'])
				if 'Time_mean_by_stddev' in info[agent_id].keys():
					# print("mean by variance is", info[agent_id]['Mean_by_variance'])
					time_mean_by_stddev_list.append(info[agent_id]['Time_mean_by_stddev'])
			
			agent_rew = f'agent{agent_id}/individual_rewards'
			times     = f'agent{agent_id}/time_to_goal'
			dists     = f'agent{agent_id}/dist_to_goal'
			agent_col = f'agent{agent_id}/num_agent_collisions'
			obst_col  = f'agent{agent_id}/num_obstacle_collisions'
			min_times = f'agent{agent_id}/min_time_to_goal'
			dist_mean = f'agent{agent_id}/distance_mean'
			dist_variance  = f'agent{agent_id}/distance_variance'
			mean_variance  = f'agent{agent_id}/mean_variance'
			dists_traveled  = f'agent{agent_id}/dists_traveled'
			time_taken  = f'agent{agent_id}/time_taken'
			formation_dist  = f'agent{agent_id}/formation_dist'
			time_mean  = f'agent{agent_id}/time_mean'
			time_variance  = f'agent{agent_id}/time_variance'
			time_mn_by_stddev  = f'agent{agent_id}/time_mn_by_stddev'

			env_infos[agent_rew] = idv_rews
			env_infos[times]     = time_to_goals
			env_infos[min_times] = min_times_to_goal
			env_infos[dists]     = dist_goals
			env_infos[agent_col] = idv_collisions
			env_infos[obst_col]  = obst_collisions
			env_infos[dist_mean] = distance_mean
			env_infos[dist_variance]  = distance_variance
			env_infos[mean_variance]  = mean_by_cov
			env_infos[dists_traveled] = dists_traveled_list
			env_infos[time_taken] = time_taken_list
			env_infos[formation_dist] = formation_distances
			env_infos[time_mean] = time_mean_list
			env_infos[time_variance]  = time_variance_list
			env_infos[time_mn_by_stddev]  = time_mean_by_stddev_list
		return env_infos

	def log_train(self, train_infos:Dict, total_num_steps:int):
		"""
		Log training info.
		train_infos: (dict) 
			information about training update.
		total_num_steps: (int) 
			total number of training env steps.
		"""
		for k, v in train_infos.items():
			if self.use_wandb:
				wandb.log({k: v}, step=total_num_steps)
			else:
				self.writter.add_scalars(k, {k: v}, total_num_steps)

	def log_env(self, env_infos:Dict, total_num_steps:int):
		"""
		Log env info.
		env_infos: (dict) 
			information about env state.
		total_num_steps: (int) 
			total number of training env steps.
		"""
		for k, v in env_infos.items():
			if len(v)>0:
				if self.use_wandb:
					# print("k", k, "v",v)
					wandb.log({k: np.mean(v)}, step=total_num_steps)
				else:
					self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

	def get_fairness_metric(self, env_infos:Dict):
		"""
			Get the collisions from the env_infos
			Example: {'agent0/individual_rewards': [5], 
					'agent0/time_to_goal': [0.6000000000000001], 
					'agent0/min_time_to_goal': [0.23632679886748278], 
					'agent0/dist_to_goal': [0.03768003822249384], 
					'agent0/num_agent_collisions': [1.0], 
					'agent0/num_obstacle_collisions': [0.0], 
		"""
		mean_variance = []
		for k, v in env_infos.items():
			# print("k is",k, "v is",v)
			if 'mean_variance' in k:
				mean_variance.append(v[0])
		return mean_variance


	def get_dist_mean(self, env_infos:Dict):
		"""
			Get the collisions from the env_infos
			Example: {'agent0/individual_rewards': [5], 
					'agent0/time_to_goal': [0.6000000000000001], 
					'agent0/min_time_to_goal': [0.23632679886748278], 
					'agent0/dist_to_goal': [0.03768003822249384], 
					'agent0/': [1.0], 
					'agent0/num_obstacle_collisions': [0.0], 
		"""
		distance_mean = []
		for k, v in env_infos.items():
			# print("k is",k, "v is",v)
			if 'distance_mean' in k:
				distance_mean.append(v[0])
		return distance_mean

	def get_dist_std(self, env_infos:Dict):
		"""
			Get the collisions from the env_infos
			Example: {'agent0/individual_rewards': [5], 
					'agent0/time_to_goal': [0.6000000000000001], 
					'agent0/min_time_to_goal': [0.23632679886748278], 
					'agent0/dist_to_goal': [0.03768003822249384], 
					'agent0/num_agent_collisions': [1.0], 
					'agent0/num_obstacle_collisions': [0.0], 
		"""
		distance_std = []
		for k, v in env_infos.items():
			# print("k is",k, "v is",v)
			if 'distance_variance' in k:
				distance_std.append(v[0])
		return distance_std
	


	def get_time_fairness(self, env_infos:Dict):
		"""
			Get the time_mean_by_stddev from the env_infos
			Example: {'agent0/time_mean': [5], 
					'agent0/time_variance': [1.0], 
					'agent0/time_mn_by_stddev': [0.0], 
		"""
		time_mean_variance = []
		for k, v in env_infos.items():
			# print("k is",k, "v is",v)
			if 'time_mn_by_stddev' in k:
				time_mean_variance.append(v[0])
		return time_mean_variance


	def get_time_mean(self, env_infos:Dict):
		"""
			Get the time_mean from the env_infos
			Example: {'agent0/time_mean': [5], 
					'agent0/time_variance': [1.0], 
					'agent0/time_mn_by_stddev': [0.0], 
		"""
		time_mean = []
		for k, v in env_infos.items():
			# print("k is",k, "v is",v)
			if 'time_mean' in k:
				time_mean.append(v[0])
		return time_mean

	def get_time_std(self, env_infos:Dict):
		"""
			Get the time_variance from the env_infos
			Example: {'agent0/time_mean': [5], 
					'agent0/time_variance': [1.0], 
					'agent0/time_mn_by_stddev': [0.0], 
		"""
		time_std = []
		for k, v in env_infos.items():
			# print("k is",k, "v is",v)
			if 'time_variance' in k:
				time_std.append(v[0])
		return time_std



	def get_dists_traveled(self, env_infos:Dict):
		"""
			Get the collisions from the env_infos
			Example: {'agent0/individual_rewards': [5], 
					'agent0/time_to_goal': [0.6000000000000001], 
					'agent0/min_time_to_goal': [0.23632679886748278], 
					'agent0/dist_to_goal': [0.03768003822249384], 
					'agent0/num_agent_collisions': [1.0], 
					'agent0/num_obstacle_collisions': [0.0], 
		"""
		dists_traveled = []
		for k, v in env_infos.items():
			# print("k is",k)
			# print("v is",v)
			if 'dists_traveled' in k:
				# print("dists are",v, k)
				dists_traveled.append(v[0])
		return dists_traveled
	
	def get_time_taken(self, env_infos:Dict):
		"""
			Get the collisions from the env_infos
			Example: {'agent0/individual_rewards': [5], 
					'agent0/time_to_goal': [0.6000000000000001], 
					'agent0/min_time_to_goal': [0.23632679886748278], 
					'agent0/dist_to_goal': [0.03768003822249384], 
					'agent0/num_agent_collisions': [1.0], 
					'agent0/num_obstacle_collisions': [0.0], 
		"""
		time_taken = []
		for k, v in env_infos.items():
			# print("k is",k)
			# print("v is",v)
			if 'time_taken' in k:
				# print("dists are",v)
				time_taken.append(v[0])
		return time_taken
	
	def get_collisions(self, env_infos:Dict):
		"""
			Get the collisions from the env_infos
			Example: {'agent0/individual_rewards': [5], 
					'agent0/time_to_goal': [0.6000000000000001], 
					'agent0/min_time_to_goal': [0.23632679886748278], 
					'agent0/dist_to_goal': [0.03768003822249384], 
					'agent0/num_agent_collisions': [1.0], 
					'agent0/num_obstacle_collisions': [0.0], 
					'agent1/individual_rewards': [5], 
					'agent1/time_to_goal': [0.6000000000000001], 
					'agent1/min_time_to_goal': [0.3067362645187025], 
					'agent1/dist_to_goal': [0.0387233764393595], 
					'agent1/num_agent_collisions': [1.0], 
					'agent1/num_obstacle_collisions': [0.0]}

		"""
		collisions = 0
		for k, v in env_infos.items():
			if 'agent_collision' in k:
					collisions += v[0]/2.0
			if 'obstacle_collisions' in k:
				if len(v)>0:
					collisions += v[0]
		return collisions

	def get_fraction_episodes(self, env_infos:Dict):
		"""
			Get the fraction of episode required to get to the goals
			from env_infos
			Example: {'agent0/individual_rewards': [5], 
					'agent0/time_to_goal': [0.6000000000000001], 
					'agent0/min_time_to_goal': [0.23632679886748278], 
					'agent0/dist_to_goal': [0.03768003822249384], 
					'agent0/num_agent_collisions': [1.0], 
					'agent0/num_obstacle_collisions': [0.0], 
					'agent1/individual_rewards': [5], 
					'agent1/time_to_goal': [0.6000000000000001], 
					'agent1/min_time_to_goal': [0.3067362645187025], 
					'agent1/dist_to_goal': [0.0387233764393595], 
					'agent1/num_agent_collisions': [1.0], 
					'agent1/num_obstacle_collisions': [0.0]}
		"""
		fracs = []
		success = []
		time_taken = []
		for k, v in env_infos.items():
			if 'time_to_goal' in k and 'min_time_to_goal' not in k:
				# print("v is",v, self.all_args.episode_length * self.dt,v[0]/(self.all_args.episode_length * self.dt))
				fracs.append(v[0] / (self.all_args.episode_length * self.dt))
				time_taken.append(v[0])
				# if didn't reach goal then time_to_goal >= episode_len * dt
				# if v[0] < self.all_args.episode_length * self.dt:
				# 	success.append(1)
				# else:
				# 	success.append(0)
			## add success based on dist_to_goal
			if 'dist_to_goal' in k:
				if v[0] < self.all_args.min_dist_thresh:
					# print("v is",v, self.all_args.min_dist_thresh)
					success.append(1)
				else:
					success.append(0)
		assert len(success) == self.all_args.num_agents
		# if sum(success) == self.all_args.num_agents:
		# 	success = True
		# else:
		# 	success = False
					
		# print ("success",success)
		return fracs, success, time_taken
		
	# create get_formation_success() function 
	def get_formation_success(self, env_infos:Dict):
		"""
			Get the formation success info from env_infos
			Example: {'agent0/formation_dist': [5], 
					'agent1/formation_dist': [5]}
		"""
		formation_success = []
		for k, v in env_infos.items():
			if 'formation_dist' in k:
				# print("formation dist", v)
				formation_success.append(v[0])

		# print("formation success", formation_success)
		return formation_success

