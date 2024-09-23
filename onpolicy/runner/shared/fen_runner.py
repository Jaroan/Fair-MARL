import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from typing import Tuple

def _t2n(x):
	return x.detach().cpu().numpy()

class FENGraphMAPPORunner(Runner):
	"""
	Runner class to perform training, evaluation and data collection for the FEN Graph MAPPO.
	"""
	def __init__(self, config):
		super(FENGraphMAPPORunner, self).__init__(config)
		self.controller = None  # Initialize controller network
		self.sub_policies = []  # List to store sub-policies
		self.u = np.zeros((self.num_agents,))  # Individual utilities
		self.u_bar = np.zeros((self.num_agents,))  # Average utilities
		self.c = 1.0  # Constant for utility calculation
		self.T = 10  # Update interval

	def run(self):
		self.warmup()

		episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

		for episode in range(episodes):
			# The controller chooses one sub-policy
			z = self.controller.choose_policy()

			for step in range(self.episode_length):
				values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step, z)

				# Env step
				obs, rewards, dones, infos = self.envs.step(actions_env)

				data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

				# Insert data into buffer
				self.insert(data)

				# Update utilities and reselect policy if necessary
				if step % self.T == 0:
					self.update_sub_policy(z)
					self.update_utilities()
					z = self.controller.choose_policy()

			# Compute returns
			self.compute()

			# Update networks
			self.update_controller()
			
			# Log information
			if episode % self.log_interval == 0:
				self.log()

	def collect(self, step: int, z: int) -> Tuple[np.ndarray, ...]:
		self.trainer.prep_rollout()
		
		# Use the chosen sub-policy to get actions
		policy = self.sub_policies[z]
		
		value, action, action_log_prob, rnn_states, rnn_states_critic = policy.get_actions(
			np.concatenate(self.buffer.share_obs[step]),
			np.concatenate(self.buffer.obs[step]),
			np.concatenate(self.buffer.rnn_states[step]),
			np.concatenate(self.buffer.rnn_states_critic[step]),
			np.concatenate(self.buffer.masks[step]))

		# [self.envs, agents, dim]
		values = np.array(np.split(_t2n(value), self.n_rollout_threads))
		actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
		action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
		rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
		rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

		return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions

	def update_sub_policy(self, z: int):
		self.trainer.prep_training()
		self.sub_policies[z].update(self.buffer)

	def update_utilities(self):
		# Update average utilities (simplified gossip algorithm)
		self.u_bar = np.mean(self.u)
		
		# Calculate utility-based rewards
		r_hat = self.u_bar / self.c + np.abs(self.u / self.u_bar - 1)
		
		# Update buffer rewards
		self.buffer.rewards[-1] = r_hat

	def update_controller(self):
		self.trainer.prep_training()
		self.controller.update(self.buffer)

	def log(self):
		# Implement logging logic here
		pass



	@torch.no_grad()
	def render(self, get_metrics:bool=False):
		"""
			Visualize the env.
			get_metrics: bool (default=False)
				if True, just return the metrics of the env and don't render.
		"""
		envs = self.envs
		self.reset_number = 0
		all_frames = []
		total_dists_traveled, total_time_taken = [], []
		rewards_arr, success_rates_arr, num_collisions_arr, frac_episode_arr, fairness_param = [], [], [], [],[]
		dist_mean_arr, time_mean_arr = [],[]
		stddev_param = []
		dists_trav_list = np.zeros((self.num_agents))
		time_taken_list = np.zeros((self.num_agents))
		time_fairness, time_stddev_param, time_mean = [], [], []
		print("num_episodes: ", self.all_args.render_episodes)


		###### store the fairness values in a CSV file ##########
		with open('/Users/jasmine/Jasmine/MIT/MARL/Codes/Team-Fair-MARL/'+str(self.all_args.model_name)+'_firstgoaldone_nogoal_fair_vs_success.csv', 'a', newline="") as f1:
			# create the csv writer
			writer = csv.writer(f1)
		#########################################################

		for episode in range(self.all_args.render_episodes):
			# print("episode", episode)
			obs, agent_id, node_obs, adj = envs.reset()
			if not get_metrics:
				if self.all_args.save_gifs:
					image = envs.render('rgb_array')[0][0]
					all_frames.append(image)
				else:
					envs.render('human')

			rnn_states = np.zeros((self.n_rollout_threads, 
									self.num_agents, 
									self.recurrent_N, 
									self.hidden_size), 
									dtype=np.float32)
			masks = np.ones((self.n_rollout_threads, 
							self.num_agents, 1), 
							dtype=np.float32)
			available_actions = np.ones((self.num_agents, 5), 
										dtype=np.float32)
			episode_rewards = []
			
			for step in range(self.episode_length):

				calc_start = time.time()
				zero_masks = masks[0] == 0

				if 	not zero_masks.all():
					available_actions = np.ones((self.num_agents, 5), 
										dtype=np.float32)
				# Broadcast the boolean mask to match the shape of available_actions
				broadcasted_zero_masks = np.broadcast_to(zero_masks, available_actions.shape)

				available_actions[zero_masks[:,0]] = np.array([1, 0, 0, 0, 0])
				self.trainer.prep_rollout()
				action, rnn_states = self.trainer.policy.act(
													np.concatenate(obs),
													np.concatenate(node_obs),
													np.concatenate(adj),
													np.concatenate(agent_id),
													np.concatenate(rnn_states),
													np.concatenate(masks),
													available_actions = available_actions,
													deterministic=True)
				actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
				rnn_states = np.array(np.split(_t2n(rnn_states), 
									self.n_rollout_threads))

				if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
					for i in range(envs.action_space[0].shape):
						uc_actions_env = np.eye(
								envs.action_space[0].high[i]+1)[actions[:, :, i]]
						if i == 0:
							actions_env = uc_actions_env
						else:
							actions_env = np.concatenate((actions_env, 
														uc_actions_env), 
														axis=2)
				elif envs.action_space[0].__class__.__name__ == 'Discrete':
					actions_env = np.squeeze(np.eye(
											envs.action_space[0].n)[actions], 2)
				else:
					raise NotImplementedError

				# Obser reward and next obs
				obs, agent_id, node_obs, adj, \
					rewards,dones, infos, reset_count = envs.step(actions_env)

				episode_rewards.append(rewards)
				dones_env = np.all(dones)
				rnn_states[dones == True] = np.zeros(((dones == True).sum(), 
													self.recurrent_N, 
													self.hidden_size), 
													dtype=np.float32)
				masks = np.ones((self.n_rollout_threads, 
								self.num_agents, 1), 
								dtype=np.float32)
				masks[dones == True] = np.zeros(((dones == True).sum(), 1), 
												dtype=np.float32)
				dones_env = np.all(dones, axis=1)
				masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
				if not get_metrics:
					if self.all_args.save_gifs:

						image = envs.render('rgb_array')[0][0]
						all_frames.append(image)
						calc_end = time.time()
						elapsed = calc_end - calc_start
						if elapsed < self.all_args.ifi:
							time.sleep(self.all_args.ifi - elapsed)
					else:
						envs.render('human')


				self.reset_number += reset_count
				if reset_count > 0:
					break

				if self.reset_number == self.all_args.render_episodes :
					break

			env_infos = self.process_infos(infos)

			num_collisions = self.get_collisions(env_infos)
			frac, success,time_taken = self.get_fraction_episodes(env_infos)
			if np.any(frac==1):
				frac_max = 1.0
			else:
				frac_max = np.max(frac)
			rewards_arr.append(np.mean(np.sum(np.array(episode_rewards), axis=0)))
			frac_episode_arr.append(frac_max)
			success_rates_arr.append(success)
			num_collisions_arr.append(num_collisions)
			fairness_metric = self.get_fairness_metric(env_infos)
			stddev_metric = self.get_dist_std(env_infos)
			fairness_param.append(fairness_metric[-1])

			stddev_param.append(1.0/(stddev_metric[-1]+0.0001))

			dist_mean = self.get_dist_mean(env_infos)
			dist_mean_arr.append(dist_mean[-1])
			time_mean = self.get_time_mean(env_infos)
			time_mean_arr.append(time_mean[-1])
	
			dists_traveled = self.get_dists_traveled(env_infos)
			dists_trav_list +=dists_traveled
			time_taken_list +=time_taken

			total_dists_traveled.append(np.sum(dists_traveled))
			total_time_taken.append(np.sum(time_taken))
			time_fairness_metric = self.get_time_fairness(env_infos)
			time_stddev_metric = self.get_time_std(env_infos)
			time_fairness.append(time_fairness_metric[-1])
			time_stddev_param.append(1.0/(time_stddev_metric[-1]+0.0001))

			csv_data1 = [self.num_obstacles, 
						self.num_agents,
						self.all_args.world_size,
						self.episode_length,
						self.all_args.render_episodes,
						self.reset_number, 
						step, 
						fairness_metric[-1],
						np.mean(success),
						frac_max,
						total_dists_traveled[-1],
						total_time_taken[-1],
						]
			###### store the fairness values in a CSV file ##########
			with open('/Users/jasmine/Jasmine/MIT/MARL/Codes/Team-Fair-MARL/'+str(self.all_args.model_name)+'_firstgoaldone_nogoal_fair_vs_success_new.csv', 'a', newline="") as f1:
				# create the csv writer
				writer = csv.writer(f1)
				writer.writerow(csv_data1)
			#########################################################

		# Calculate the statistics for the box whisker plot
		fair_minimum = np.min(fairness_param)
		fair_0_1_quantile = np.percentile(fairness_param, 10)
		fair_median = np.median(fairness_param)
		fair_0_9_quantile = np.percentile(fairness_param, 90)
		fair_maximum = np.max(fairness_param)
		fair_mean = np.mean(fairness_param)

		time_fair_minimum = np.min(time_fairness)
		time_fair_0_1_quantile = np.percentile(time_fairness, 10)
		time_fair_median = np.median(time_fairness)
		time_fair_0_9_quantile = np.percentile(time_fairness, 90)
		time_fair_maximum = np.max(time_fairness)
		time_fair_mean = np.mean(time_fairness)

		# calculate statistics for dist_mean_arr
		dist_mean_minimum = np.min(dist_mean_arr)
		dist_mean_0_1_quantile = np.percentile(dist_mean_arr, 10)
		dist_mean_median = np.median(dist_mean_arr)
		dist_mean_0_9_quantile = np.percentile(dist_mean_arr, 90)
		dist_mean_maximum = np.max(dist_mean_arr)
		dist_mean_mean = np.mean(dist_mean_arr)

		# calculate statistics for time_mean_arr
		time_mean_minimum = np.min(time_mean_arr)
		time_mean_0_1_quantile = np.percentile(time_mean_arr, 10)
		time_mean_median = np.median(time_mean_arr)
		time_mean_0_9_quantile = np.percentile(time_mean_arr, 90)
		time_mean_maximum = np.max(time_mean_arr)
		time_mean_mean = np.mean(time_mean_arr)

		# calculate statistics for success rates
		success_rates_minimum = np.min(success_rates_arr)
		success_rates_0_1_quantile = np.percentile(success_rates_arr, 10)
		success_rates_median = np.median(success_rates_arr)
		success_rates_0_9_quantile = np.percentile(success_rates_arr, 90)
		success_rates_maximum = np.max(success_rates_arr)
		success_rates_mean = np.mean(success_rates_arr)

		total_dists_traveled_median = np.median(total_dists_traveled)
		total_dists_traveled_mean = np.mean(total_dists_traveled)
		total_dists_traveled_0_1_quantile = np.percentile(total_dists_traveled, 10)
		total_dists_traveled_0_9_quantile = np.percentile(total_dists_traveled, 90)
		total_dists_traveled_min = np.min(total_dists_traveled)
		total_dists_traveled_max = np.max(total_dists_traveled)

		total_time_taken_median = np.median(total_time_taken)
		total_time_taken_mean = np.mean(total_time_taken)
		total_time_taken_0_1_quantile = np.percentile(total_time_taken, 10)
		total_time_taken_0_9_quantile = np.percentile(total_time_taken, 90)
		total_time_taken_min = np.min(total_time_taken)
		total_time_taken_max = np.max(total_time_taken)		


		np.set_printoptions(linewidth=400)
		print("Rewards", np.mean(rewards_arr))
		print("Frac of episode", np.mean(frac_episode_arr))

		# report the success rates statistics
		print("Success rates mean", success_rates_mean)
		print("Success rates median",success_rates_median)

		print("Num collisions", np.mean(num_collisions_arr))
		print("Fairness Median", fair_median)
		print("Fairness Mean", fair_mean)
		# Print the values
		print("Fair Minimum:", fair_minimum)
		print("Fair 0_1 Quantile:", fair_0_1_quantile)
		print("Fair Median:", fair_median)
		print("Fair 0.9 Quantile:", fair_0_9_quantile)
		print("Fair Maximum:", fair_maximum)

		print("Dists traveled", dists_trav_list)
		print("Time taken", time_taken_list)


		# print time fairness and time stddev
		print("Time Fair Minimum:", time_fair_minimum)
		print("Time Fair 0.1 Quantile:", time_fair_0_1_quantile)
		print("Time Fair Median:", time_fair_median)
		print("Time Fair 0.9 Quantile:", time_fair_0_9_quantile)
		print("Time Fair Maximum:", time_fair_maximum)
		print("Time Fair Mean:", time_fair_mean)

		# print dist mean and time mean
		print("Dist Mean Minimum:", dist_mean_minimum)
		print("Dist Mean 0.1 Quartile:", dist_mean_0_1_quantile)
		print("Dist Mean Median:", dist_mean_median)
		print("Dist Mean 0.9 Quartile:", dist_mean_0_9_quantile)
		print("Dist Mean Maximum:", dist_mean_maximum)
		print("Dist Mean Mean:", dist_mean_mean)

		print("Time Mean Minimum:", time_mean_minimum)
		print("Time Mean 0.1 Quartile:", time_mean_0_1_quantile)
		print("Time Mean Median:", time_mean_median)
		print("Time Mean 0.9 Quartile:", time_mean_0_9_quantile)
		print("Time Mean Maximum:", time_mean_maximum)
		print("Time Mean Mean:", time_mean_mean)

		print("Total Dists Traveled Median:", total_dists_traveled_median)
		print("Total Time Taken Median:", total_time_taken_median)
		rewards_mean = np.mean(rewards_arr)

		csv_data = [
			self.num_obstacles, 
			self.num_agents,
			self.all_args.world_size,
			self.episode_length,
			self.all_args.render_episodes,
			fair_mean,  # Add mean value for fairness_param
			fair_minimum,  # Add minimum value for fairness_param
			fair_0_1_quantile,  # Add first quartile value for fairness_param
			fair_median,  # Add median value for fairness_param
			fair_0_9_quantile,  # Add third quartile value for fairness_param
			fair_maximum,  # Add maximum value for fairness_param

			np.mean(frac_episode_arr),
			success_rates_minimum,
			success_rates_0_1_quantile,
			success_rates_median ,
			success_rates_0_9_quantile ,
			success_rates_maximum,
			success_rates_mean,
			np.mean(num_collisions_arr),
			rewards_mean,
			rewards_mean / self.num_agents,
			rewards_mean / (self.num_agents * self.episode_length),
			dists_trav_list,
			time_taken_list,
			time_fair_mean,  # Add mean value for time_fairness
			time_fair_minimum,  # Add minimum value for time_fairness
			time_fair_0_1_quantile,  # Add first quartile value for time_fairness
			time_fair_median,  # Add median value for time_fairness
			time_fair_0_9_quantile,  # Add third quartile value for time_fairness
			time_fair_maximum,  # Add maximum value for time_fairness

			dist_mean_mean,  # Add mean value for dist_mean_arr
			dist_mean_minimum,  # Add minimum value for dist_mean_arr
			dist_mean_0_1_quantile,  # Add first quartile value for dist_mean_arr
			dist_mean_median,  # Add median value for dist_mean_arr
			dist_mean_0_9_quantile,  # Add third quartile value for dist_mean_arr
			dist_mean_maximum,  # Add maximum value for dist_mean_arr
			time_mean_mean,  # Add mean value for time_mean_arr
			time_mean_minimum,  # Add minimum value for time_mean_arr
			time_mean_0_1_quantile,  # Add first quartile value for time_mean_arr
			time_mean_median,  # Add median value for time_mean_arr
			time_mean_0_9_quantile,  # Add third quartile value for time_mean_arr
			time_mean_maximum,  # Add maximum value for time_mean_arr
			total_dists_traveled_median,  # Add median value for total_dists_traveled
			total_dists_traveled_mean,
			total_dists_traveled_0_1_quantile,
			total_dists_traveled_0_9_quantile,
			total_dists_traveled_min,
			total_dists_traveled_max,
			total_time_taken_median,  # Add median value for total_time_taken
			total_time_taken_mean,
			total_time_taken_0_1_quantile,
			total_time_taken_0_9_quantile,
			total_time_taken_min,
			total_time_taken_max			
		]

		# csv_2 = [fairness_param]
		# open the file in the write mode
		with open('/Users/jasmine/Jasmine/MIT/MARL/Codes/Team-Fair-MARL/'+str(self.all_args.model_name)+'_firstgoaldone_nogoal_results_collect_new.csv', 'a', newline="") as f:
			# create the csv writer
			writer = csv.writer(f)
			# write a row to the csv file
			writer.writerow(csv_data)

		if not get_metrics:
			if self.all_args.save_gifs:
				# print("Hello")
				# print("gif dir", self.gif_dir)
				imageio.mimsave(str(self.gif_dir) + '/'+str(self.all_args.model_name)+'_check'+str(self.all_args.num_agents)+'.gif', 
								all_frames, duration=self.all_args.ifi, loop=0)