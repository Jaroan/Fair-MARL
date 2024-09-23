"""
	Navigation for `n` agents to `n` goals from random initial positions
	With random obstacles added in the environment
	Each agent is destined to get to its own goal unlike
	`simple_spread.py` where any agent can get to any goal (check `reward()`)
"""
from typing import Optional, Tuple, List
import argparse
import numpy as np
from numpy import ndarray as arr
from scipy import sparse
# import pyomo.environ as pe
# import scipy.spatial.distance as dist
import os,sys
sys.path.append(os.path.abspath(os.getcwd()))
from scipy.optimize import linear_sum_assignment

from multiagent.core import World, Agent, Landmark, Entity, Wall
from multiagent.scenario import BaseScenario

# from marl_fair_assign import solve_fair_assignment

entity_mapping = {'agent': 0, 'landmark': 1, 'obstacle':2, 'wall':3}


def get_thetas(poses):
	# compute angle (0,2pi) from horizontal
	thetas = [None] * len(poses)
	for i in range(len(poses)):
		# (y,x)
		thetas[i] = find_angle(poses[i])
	return thetas


def find_angle(pose):
	# compute angle from horizontal
	angle = np.arctan2(pose[1], pose[0])
	if angle < 0:
		angle += 2 * np.pi
	return angle

class Scenario(BaseScenario):
	def make_world(self, args:argparse.Namespace) -> World:
		"""
			Parameters in args
			––––––––––––––––––
			• num_agents: int
				Number of agents in the environment
				NOTE: this is equal to the number of goal positions
			• num_obstacles: int
				Number of num_obstacles obstacles
			• collaborative: bool
				If True then reward for all agents is sum(reward_i)
				If False then reward for each agent is what it gets individually
			• max_speed: Optional[float]
				Maximum speed for agents
				NOTE: Even if this is None, the max speed achieved in discrete 
				action space is 2, so might as well put it as 2 in experiments
				TODO: make list for this and add this in the state
			• collision_rew: float
				The reward to be negated for collisions with other agents and 
				obstacles
			• goal_rew: float
				The reward to be added if agent reaches the goal
			• min_dist_thresh: float
				The minimum distance threshold to classify whether agent has 
				reached the goal or not
			• use_dones: bool
				Whether we want to use the 'done=True' when agent has reached 
				the goal or just return False like the `simple.py` or 
				`simple_spread.py`
			• episode_length: int
				Episode length after which environment is technically reset()
				This determines when `done=True` for done_callback
			• graph_feat_type: str
				The method in which the node/edge features are encoded
				Choices: ['global', 'relative']
					If 'global': 
						• node features are global [pos, vel, goal, entity-type]
						• edge features are relative distances (just magnitude)
						• 
					If 'relative':
						• TODO decide how to encode stuff

			• max_edge_dist: float
				Maximum distance to consider to connect the nodes in the graph
		"""
		# pull params from args
		self.world_size = args.world_size
		self.num_agents = args.num_agents
		self.num_scripted_agents = args.num_scripted_agents
		self.num_obstacles = args.num_obstacles
		self.collaborative = args.collaborative
		self.max_speed = args.max_speed
		self.collision_rew = args.collision_rew
		self.goal_rew = args.goal_rew
		self.min_dist_thresh = args.min_dist_thresh
		self.use_dones = args.use_dones
		self.episode_length = args.episode_length
		self.target_radius = 0.5  # fixing the target radius for now
		self.ideal_theta_separation = (
			2 * np.pi
		) / self.num_agents  # ideal theta difference between two agents
		self.identity_size = 0
		self.goal_match_index = np.arange(self.num_agents)
		# fairness args
		self.fair_wt = args.fair_wt
		self.fair_rew = args.fair_rew

		# create heatmap matrix to determine the goal agent pairs
		self.goal_reached = np.zeros(self.num_agents)
		self.wrong_goal_reached = np.zeros(self.num_agents)
		self.goal_matched = np.zeros(self.num_agents)
		self.agent_dist_traveled = np.zeros(self.num_agents)
		self.agent_time_taken = np.zeros(self.num_agents)

		if not hasattr(args, 'max_edge_dist'):
			self.max_edge_dist = 1
			print('_'*60)
			print(f"Max Edge Distance for graphs not specified. "
					f"Setting it to {self.max_edge_dist}")
			print('_'*60)
		else:
			self.max_edge_dist = args.max_edge_dist
		####################
		world = World()
		# graph related attributes
		world.cache_dists = True # cache distance between all entities
		world.graph_mode = True
		world.graph_feat_type = args.graph_feat_type
		world.world_length = args.episode_length
		# metrics to keep track of
		world.current_time_step = 0
		# to track time required to reach goal
		world.times_required = -1 * np.ones(self.num_agents)
		world.dists_to_goal = -1 * np.ones(self.num_agents)
		# set any world properties
		world.dim_c = 2
		self.num_landmarks = args.num_landmarks # no. of goals equal to no. of agents
		num_scripted_agents_goals = self.num_scripted_agents
		world.collaborative = args.collaborative

		# add agents
		global_id = 0
		world.agents = [Agent() for i in range(self.num_agents)]
		world.scripted_agents = [Agent() for _ in range(self.num_scripted_agents)]
		for i, agent in enumerate(world.agents + world.scripted_agents):
			agent.id = i
			agent.name = f'agent {i}'
			agent.collide = True
			agent.silent = True
			agent.global_id = global_id
			global_id += 1
			# NOTE not changing size of agent because of some edge cases; 
			# TODO have to change this later
			# agent.size = 0.15
			agent.max_speed = self.max_speed
		# add landmarks (goals)
		world.landmarks = [Landmark() for i in range(self.num_landmarks)]
		world.scripted_agents_goals = [Landmark() for i in range(num_scripted_agents_goals)]
		for i, landmark in enumerate(world.landmarks):
			landmark.id = i
			landmark.name = f'landmark {i}'
			landmark.collide = False
			landmark.movable = False
			landmark.global_id = global_id
			global_id += 1
		# add obstacles
		world.obstacles = [Landmark() for i in range(self.num_obstacles)]
		for i, obstacle in enumerate(world.obstacles):
			obstacle.name = f'obstacle {i}'
			obstacle.collide = True
			obstacle.movable = False
			obstacle.global_id = global_id
			global_id += 1
		## add wall
		# num obstacles per wall is twice the length of the wall
		wall_length = np.random.uniform(0.2, 0.8)
		# print("wall_length",wall_length)
		self.wall_length = wall_length * self.world_size/4
		self.num_obstacles_per_wall = np.int32(1*self.wall_length/world.agents[0].size)
		# print("num_obstacles_per_wall",self.num_obstacles_per_wall)
		num_walls =2
		world.walls = [Wall() for i in range(num_walls)]
		for i, wall in enumerate(world.walls):
			wall.name = f'wall {i}'
			wall.collide = True
			wall.movable = False
			wall.global_id = global_id
			global_id += 1
		# num_walls = 1

		# make initial conditions
		self.reset_world(world)
		return world

	def reset_world(self, world:World) -> None:
		# metrics to keep track of
		world.current_time_step = 0
		# to track time required to reach goal
		world.times_required = -1 * np.ones(self.num_agents)
		world.dists_to_goal = -1 * np.ones(self.num_agents)

		# track distance left to the goal
		world.dist_left_to_goal = -1 * np.ones(self.num_agents)
		# number of times agents collide with stuff
		world.num_obstacle_collisions = np.zeros(self.num_agents)
		world.num_goal_collisions = np.zeros(self.num_agents)
		world.num_agent_collisions = np.zeros(self.num_agents)
		world.formation_dist = np.zeros(self.num_agents)
		world.formation_complete = np.zeros(self.num_agents)



		#################### set colours ####################
		# set colours for agents
		for i, agent in enumerate(world.agents):
			agent.color = np.array([0.35, 0.35, 0.85])
			if i%3 == 0:
				agent.color = np.array([0.35, 0.35, 0.85])
			elif i%3 == 1:
				agent.color = np.array([0.85, 0.35, 0.35])
			else:
				agent.color = np.array([0.35, 0.85, 0.35])
			agent.state.p_dist = 0.0
			agent.state.time = 0.0
		# set colours for scripted agents
		for i, agent in enumerate(world.scripted_agents):
			agent.color = np.array([0.15, 0.15, 0.15])
		# set colours for landmarks
		for i, landmark in enumerate(world.landmarks):
			if i%3 == 0:
				landmark.color = np.array([0.35, 0.35, 0.35])
			elif i%3 == 1:
				landmark.color = np.array([0.85, 0.35, 0.35])
			else:
				landmark.color = np.array([0.35, 0.85, 0.35])
		# set colours for scripted agents goals
		for i, landmark in enumerate(world.scripted_agents_goals):
			landmark.color = np.array([0.15, 0.95, 0.15])
		# set colours for obstacles
		for i, obstacle in enumerate(world.obstacles):
			obstacle.color = np.array([0.25, 0.25, 0.25])
		# set colours for wall obstacles
		# for i, wall_obstacle in enumerate(world.wall_obstacles):
		# 	wall_obstacle.color = np.array([0.25, 0.25, 0.25])
		#####################################################
		self.random_scenario(world)

	def random_scenario(self, world):
		"""
			Randomly place agents and landmarks
		"""
		####### set random positions for entities ###########
		# set random static obstacles first
		for obstacle in world.obstacles:
			obstacle.state.p_pos = 0.8 * np.random.uniform(-self.world_size/2, 
															self.world_size/2, 
															world.dim_p)
			obstacle.state.p_vel = np.zeros(world.dim_p)
		#####################################################




		############## create wall positions #################
		wall_position = np.random.uniform(0.2, 0.9)
		wall_axis  = np.array([wall_position * self.world_size/2, -wall_position * self.world_size/2])
								# wall_position * self.world_size/2, -wall_position * self.world_size/2])
		wall_obst_count = 0
		# print("wall_obst_count",wall_obst_count)
		# set wall positions
		for i , wall in enumerate(world.walls):
			# print("wall",i)
			wall.orient='V'
			# if i<2:
			# 	wall.orient='H'

			wall.width = 0.1
			wall.hard = True
			wall.endpoints=np.array([-self.wall_length, self.wall_length])
			wall.axis_pos = wall_axis[i]
			if wall.orient == 'H':
				# Horizontal wall
				x_min, x_max = wall.endpoints
				x = (x_min+ x_max)/2
				y = wall.axis_pos
				wall.state.p_pos = np.array([x, y])  # Set the physical position
				wall.state.p_vel = np.zeros(world.dim_p)  # Set the physical velocity

			elif wall.orient == 'V':
				# Vertical wall
				# print("wall",i)
				x = wall.axis_pos
				y_min, y_max = wall.endpoints
				y = (y_min + y_max) / 2
				wall.state.p_pos = np.array([x, y])  # Set the physical position
				wall.state.p_vel = np.zeros(world.dim_p)  # Set the physical velocity

		#####################################################




		# set agents at random positions not colliding with obstacles
		num_agents_added = 0
		agents_added = []
		boundary_thresh = 0.9
		# uniform_pos = np.linspace(-boundary_thresh*self.world_size/2, boundary_thresh*self.world_size/2, self.num_agents)
		# print("uni",uniform_pos)

		# print("HI")
		# # circle arrangement
		# agent_theta = np.linspace(0, 2*np.pi, self.num_agents, endpoint=False)
		# # print("theta",agent_theta)
		# radius = 0.92 * self.world_size/2

		while True:
			# print(uniform_pos)
			if num_agents_added == self.num_agents:
				break
			# # for random pos
			random_pos = np.random.uniform(-self.world_size/2, 
											self.world_size/2, 
											world.dim_p)
			line_pos = random_pos
			## print("random pos",line_pos)


			agent_size = world.agents[num_agents_added].size
			obs_collision = self.is_obstacle_collision(line_pos, agent_size, world)
			# goal_collision = self.is_goal_collision(uniform_pos, agent_size, world)

			agent_collision = self.check_agent_collision(line_pos, agent_size, agents_added)
			# print("obs_collision",obs_collision,"agent_collision",agent_collision)
			if not obs_collision and not agent_collision:
				world.agents[num_agents_added].state.p_pos = line_pos
				world.agents[num_agents_added].state.p_vel = np.zeros(world.dim_p)
				world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
				agents_added.append(world.agents[num_agents_added])
				num_agents_added += 1
			# print(num_agents_added)
		agent_pos = [agent.state.p_pos for agent in world.agents]


		# set landmarks (goals) at random positions not colliding with obstacles 
		# and also check collisions with already placed goals
		num_goals_added = 0
		# goals_added = []
		# uniform_pos = np.linspace(0, boundary_thresh*self.world_size/2, self.num_agents)
		# angle_shift = np.pi/self.num_agents
		# goal_theta = np.linspace(0+angle_shift, 2*np.pi+angle_shift, self.num_agents, endpoint=False)
		# # print("goal_theta",goal_theta)
		# goal_radius = 0.45 * self.world_size/2
		# print("self.num_landmarks",self.num_landmarks)
		while True:
			if num_goals_added == self.num_landmarks:
				break


			# # for random pos
			random_pos = 0.5 * np.random.uniform(-self.world_size/2, 
												self.world_size/2, 
												world.dim_p)
			# random_pos = np.array([0.0, 0.0])
			line_pos = random_pos


			goal_size = world.landmarks[num_goals_added].size
			obs_collision = self.is_obstacle_collision(line_pos, goal_size, world)
			landmark_collision = self.is_landmark_collision(line_pos, 
												goal_size, 
												world.landmarks[:num_goals_added])
			if not landmark_collision and not obs_collision:
			# if not landmark_collision:
				world.landmarks[num_goals_added].state.p_pos = line_pos
				world.landmarks[num_goals_added].state.p_vel = np.zeros(world.dim_p)
				num_goals_added += 1
		goal_pos = [goal.state.p_pos for goal in world.landmarks]

		# set wall at end of world positions 
		# and also check collisions with already placed goals
		# num_goals_added = 0
		# goals_added = []
		#####################################################


		############ update the cached distances ############
		world.calculate_distances()
		self.update_graph(world)
		####################################################

		#################### set first expected positions ####################
		# set the expected positions for the agents
		self.expected_poses = np.zeros((self.num_agents, world.dim_p))
		landmark_pose = world.landmarks[0].state.p_pos
		relative_poses = [
			agent.state.p_pos - landmark_pose for agent in world.agents
		]
		thetas = get_thetas(relative_poses)
		# anchor at the agent with min theta (closest to the horizontal line)
		theta_min = min(thetas)
		# print("reset theta_min",theta_min,"thetas",thetas)
		self.expected_poses = np.array([
			landmark_pose
			+ self.target_radius
			* np.array(
				[
					np.cos(theta_min + i * self.ideal_theta_separation),
					np.sin(theta_min + i * self.ideal_theta_separation),
				]
			)
			for i in range(self.num_agents)
		])
		### set a flag for expected Postions to indicate if an agent is currently occupying that position
		self.expected_poses_occupied = np.zeros(self.num_agents)
		# print("reset expected_poses",self.expected_poses)
		############ find minimum times to goals ############
		if self.max_speed is not None:
			for agent in world.agents:
				self.min_time(agent, world)
		#####################################################		
		# ########### set fair goals for each agent ###########
		# costs = dist.cdist(agent_pos, goal_pos)
		# x, objs = solve_fair_assignment(costs)
		# # print('x',x, objs)
		# self.goal_match_index = np.where(x==1)[1]
		# if self.goal_match_index.size == 0 or self.goal_match_index.shape != (self.num_agents,):
		# 	self.goal_match_index = np.arange(self.num_agents)
		# # print("goalmatch",self.goal_match_index,agent.id)
		# color_list= [np.array([0.35, 0.35, 0.85]),np.array([0.85, 0.35, 0.35]),np.array([0.35, 0.85, 0.35])]
		# for i, agent in enumerate(world.agents):
		# 	# landmark.color = np.array([0.15, 0.85, 0.15])
		# 	agent.color = color_list[self.goal_match_index[i]%3]
		# 		# landmark.color = np.array([0.15, 0.85, 0.15])

		#####################################################
		# reset  

	def info_callback(self, agent:Agent, world:World) -> Tuple:
		# TODO modify this 
		# rew = 0
		# collisions = 0
		# occupied_landmarks = 0
		# goal = world.get_entity('landmark',self.expected_poses[agent.id])
		world.dists = np.array([np.linalg.norm(agent.state.p_pos - l) for l in self.expected_poses])


	
		## Calculate the distance to the central landmark and check if that falls within the given formation radius
		world.formation_dist[agent.id] = np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)
		if world.formation_dist[agent.id] < 1.05*(self.target_radius) and world.formation_dist[agent.id] > 0.95*(self.target_radius):
			world.formation_complete[agent.id] = 1
					# only update times_required for the first time it reaches the goal
			# print("info callback expected_poses",self.expected_poses)
			if world.times_required[agent.id] == -1:
				# If the minimum distance is already less than self.min_dist_thresh, use the previous goal.
				world.times_required[agent.id] = world.current_time_step * world.dt
				world.dists_to_goal[agent.id] = agent.state.p_dist
		world.dist_left_to_goal[agent.id] = np.min(world.dists)
		# print("world.dist_left_to_goal[agent.id]",world.dist_left_to_goal[agent.id],"world.dists",world.dists)

		if world.times_required[agent.id] == -1:
			world.dists_to_goal[agent.id] = agent.state.p_dist

		if agent.collide:
			if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
				world.num_obstacle_collisions[agent.id] += 1
			for a in world.agents:
				if a is agent:
					continue
				if self.is_collision(agent, a):
					world.num_agent_collisions[agent.id] += 1


		world.dist_traveled_mean = np.mean(world.dists_to_goal)
		world.dist_traveled_stddev = np.std(world.dists_to_goal)
		# print("Hi agent", agent.id, self.agent_dist_traveled[agent.id],world.dists_to_goal[agent.id],"mean",world.dist_traveled_mean,"stddev", world.dist_traveled_stddev)
		# print("fair", world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001))
		agent_info = {
			'Dist_to_goal': world.dist_left_to_goal[agent.id],
			'Time_req_to_goal': world.times_required[agent.id],
			# NOTE: total agent collisions is half since we are double counting. # EDIT corrected this.
			'Num_agent_collisions': world.num_agent_collisions[agent.id], 
			'Num_obst_collisions': world.num_obstacle_collisions[agent.id],
			'Distance_mean': world.dist_traveled_mean, 
			'Distance_variance': world.dist_traveled_stddev,
			'Mean_by_variance': world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001),
			'Dists_traveled': world.dists_to_goal[agent.id],
			'Time_taken': self.agent_time_taken[agent.id],
			'Formation_dist': world.formation_complete[agent.id]

			# add goal reached params
			# 'Goal_reached':
			# 'Goal_matched':

		}
		if self.max_speed is not None:
			agent_info['Min_time_to_goal'] =  agent.goal_min_time ## agent.goal_min_time
		return agent_info

	# check collision of entity with obstacles and walls
	def is_obstacle_collision(self, pos, entity_size:float, world:World) -> bool:
		# pos is entity position "entity.state.p_pos"
		collision = False
		for obstacle in world.obstacles:
			delta_pos = obstacle.state.p_pos - pos
			dist = np.linalg.norm(delta_pos)
			dist_min = 1.05*(obstacle.size + entity_size)
			# print("1Dist", dist,"dist_min",dist_min, collision)

			if dist < dist_min:
				collision = True
				break	
		
		# check collision with walls
		for wall in world.walls:
			if wall.orient == 'H':
				# Horizontal wall, check for collision along the y-axis
				if wall.axis_pos - entity_size / 2 <= pos[1] <= wall.axis_pos + entity_size / 2:
					if wall.endpoints[0] - entity_size / 2 <= pos[0] <= wall.endpoints[1] + entity_size / 2:
						collision = True
						break
			elif wall.orient == 'V':
				# Vertical wall, check for collision along the x-axis
				if wall.axis_pos - entity_size / 2 <= pos[0] <= wall.axis_pos + entity_size / 2:
					if wall.endpoints[0] - entity_size / 2 <= pos[1] <= wall.endpoints[1] + entity_size / 2:
						collision = True
						break
		return collision
	# check collision of entity with obstacles and walls


	# check collision of agent with other agents
	def check_agent_collision(self, pos, agent_size, agent_added) -> bool:
		collision = False
		if len(agent_added):
			for agent in agent_added:
				delta_pos = agent.state.p_pos - pos
				dist = np.linalg.norm(delta_pos)
				if dist < 1.05*(agent.size + agent_size):
					collision = True
					break
		return collision

	# check collision of agent with another agent
	def is_collision(self, agent1:Agent, agent2:Agent) -> bool:
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.linalg.norm(delta_pos)
		dist_min = 1.05*(agent1.size + agent2.size)
		return True if dist < dist_min else False

	def is_landmark_collision(self, pos, size:float, landmark_list:List) -> bool:
		collision = False
		for landmark in landmark_list:
			delta_pos = landmark.state.p_pos - pos
			dist = np.sqrt(np.sum(np.square(delta_pos)))
			dist_min = 1.05*(size + landmark.size)
			if dist < dist_min:
				collision = True
				break
		return collision

	# get min time required to reach to goal without obstacles
	def min_time(self, agent:Agent, world:World) -> float:
		assert agent.max_speed is not None, "Agent needs to have a max_speed"
		# agent_id = agent.id
		# get the goal associated to this agent
		# landmark = world.get_entity(entity_type='landmark', id=self.expected_poses[agent.id])
		dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
										self.expected_poses[agent.id])))
		min_time = dist / agent.max_speed
		agent.goal_min_time = min_time
		return min_time


	def done(self, agent, world):
		if self.use_dones:
			condition1 = world.current_time_step >= world.world_length
			self.is_success = np.all(self.delta_dists < self.min_dist_thresh)
			return condition1 or self.is_success
		else:
			# print("not using done")
			if world.current_time_step >= world.world_length:
				return True
			else:
				return False


	# # done condition for each agent
	# def done(self, agent:Agent, world:World) -> bool:
	# 	# if we are using dones then return appropriate done
	# 	if self.use_dones:
	# 		if world.current_time_step >= world.world_length:
	# 			return True
	# 		else:
	# 			# landmark = world.get_entity('landmark',self.goal_match_index[agent.id])
	# 			dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
	# 											self.expected_poses[agent.id])))
	# 			if dist < self.min_dist_thresh:
	# 				return True
	# 			else:
	# 				return False
	# 	# it not using dones then return done 
	# 	# only when episode_length is reached
	# 	else:
	# 		# print("not using done")
	# 		if world.current_time_step >= world.world_length:
	# 			return True
	# 		else:
	# 			return False


	def _bipartite_min_dists(self, dists):
		ri, ci = linear_sum_assignment(dists)
		min_dists = dists[ri, ci]
		return min_dists



	def reward(self, agent, world):
		rew = 0

		if agent.id == 0:
			landmark_pose = world.landmarks[0].state.p_pos
			relative_poses = [
				agent.state.p_pos - landmark_pose for agent in world.agents
			]
			thetas = get_thetas(relative_poses)
			# anchor at the agent with min theta (closest to the horizontal line)
			theta_min = min(thetas)
			# print("rew theta_min",theta_min,"thetas",thetas)
			self.expected_poses = np.array([
				landmark_pose
				+ self.target_radius
				* np.array(
					[
						np.cos(theta_min + i * self.ideal_theta_separation),
						np.sin(theta_min + i * self.ideal_theta_separation),
					]
				)
				for i in range(self.num_agents)
			])
			# print("reward expected_poses",self.expected_poses)
			self.dists = np.array(
				[
					[np.linalg.norm(a.state.p_pos - pos) for pos in self.expected_poses]
					for a in world.agents
				]
			)
			# print("agent_pos",np.array([a.state.p_pos for a in world.agents]))
			# print("dists",self.dists)

			## for every column in dists if the value is below min_dist_thresh then set self.expected_poses_occupied to 1 for that index
			mask = self.dists < self.min_dist_thresh
			self.expected_poses_occupied = np.any(mask, axis=0).astype(int)
			# print("expected_poses_occupied",self.expected_poses_occupied)

			# optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
			self.delta_dists = self._bipartite_min_dists(self.dists)
			# print("delta_dists",self.delta_dists)
			world.dists = self.delta_dists

		# print("self.goal_rew",self.goal_rew,"self.delta_dists",self.delta_dists[agent.id])
		## non shared rewards
		if self.delta_dists[agent.id] < self.min_dist_thresh:
			# print("goal_rew",self.goal_rew)
			rew += self.goal_rew
		else:
			# print("delta_dists",self.delta_dists[agent.id])
			rew -= self.delta_dists[agent.id]
		if agent.collide:
			for a in world.agents:
				# do not consider collision with itself
				if a.id == agent.id:
					continue
				if self.is_collision(a, agent):
					# print("self.collision_rew",self.collision_rew)
					rew -= self.collision_rew
			if self.is_obstacle_collision(pos=agent.state.p_pos,
										entity_size=agent.size, world=world):
				# print("obstacle collision",self.collision_rew)
				rew -= self.collision_rew
			# total_penalty = np.mean(np.clip(self.delta_dists, 0.0, 2))
			# self.joint_reward = -total_penalty
			# # print("joint_reward",self.joint_reward)
		# print("rew",rew)
		
		return np.clip(rew, -2*self.collision_rew, self.goal_rew)


	def observation(self, agent:Agent, world:World) -> arr:
		"""
			Return:
				[agent_vel, agent_pos, goal_pos, goal_occupied]
		"""
		# print("observation expected_poses",self.expected_poses)
		world.dists = np.array([np.linalg.norm(agent.state.p_pos - l) for l in self.expected_poses])
		if np.min(world.dists) < self.min_dist_thresh:
			# If the minimum distance is already less than self.min_dist_thresh, use the previous goal.
			agents_goal = self.expected_poses[np.argmin(world.dists)]
			self.expected_poses_occupied[np.argmin(world.dists)] = 1
			goal_occupied = np.array([self.expected_poses_occupied[np.argmin(world.dists)]])
		else:
			unoccupied_goals = self.expected_poses[self.expected_poses_occupied==0]
			if len(unoccupied_goals) > 0:
				# if goals are unoccupied, match based on bipartite matching
				self.dists = np.array(
				[
					[np.linalg.norm(a.state.p_pos - pos) for pos in self.expected_poses]
					for a in world.agents
				]
				)
				# self.delta_dists = self._bipartite_min_dists(self.dists)
				# print("delta_dists",self.delta_dists)
				_, goals = linear_sum_assignment(self.dists)
				# print("ri",agents,"ci",goals)
				agents_goal = self.expected_poses[goals[agent.id]]
				## check if the goal is occupied
				goal_occupied = np.array([self.expected_poses_occupied[goals[agent.id]]])
			else:
				# Handle the case when all goals are occupied.
				# You can choose a default action here or raise an exception.
				agents_goal = agent.state.p_pos
				self.expected_poses_occupied = np.zeros(self.num_agents)
				goal_occupied = np.array([self.expected_poses_occupied[agent.id]])
		
		goal_pos = [agents_goal - agent.state.p_pos]
		# print("goal_pos",goal_pos)
		return np.concatenate([agent.state.p_vel, agent.state.p_pos] + goal_pos + goal_occupied)


	def get_id(self, agent:Agent) -> arr:
		return np.array([agent.global_id])

	def collect_rewards(self,world):
		"""
		This function collects the rewards of all agents at once to reduce further computations of the reward
		input: world and agent information
		output: list of agent names and array of corresponding reward values [agent_names, agent_rewards]
		"""
		agent_names = []
		agent_rewards = np.zeros(self.num_agents)
		count = 0
		for agent in world.agents:
			# print(agent.name)
			agent_names.append(agent.name)
			agent_rewards[count] = self.reward(agent, world)
			# print("Rew", agent_rewards[count])
			count +=1

		return agent_names, agent_rewards

	# def modify_reward(self,reward):
	# 	"""
	# 	input: reward of one agent
	# 	output: modified verison of that reward
	# 	"""
	# 	factor = 0.0
	# 	update_reward = reward*factor
	# 	return update_reward
	
	def collect_dist(self, world):
		"""
		This function collects the distances of all agents at once to reduce further computations of the reward
		input: world and agent information
		output: list of agent names and array of corresponding distance values [agent_names, agent_rewards]
		"""
		agent_names = []
		agent_dist = np.zeros(self.num_agents)
		agent_pos =  np.zeros((self.num_agents, 2)) # create a zero vector with the size of the number of agents and positions of dim 2
		count = 0
		for agent in world.agents:
			# print(agent.name)
			agent_names.append(agent.name)
			agent_dist[count] = agent.state.p_dist
			agent_pos[count]= agent.state.p_pos
			# print("Rew", agent_rewards[count])
			count +=1
		# mean_dist = np.mean(agent_dist)
		# std_dev_dist = np.std(agent_dist)
		# print("agent_dist", agent_dist, mean_dist, std_dev_dist)

		# print("mean/std", mean_dist/(std_dev_dist+0.0001))
		return np.mean(agent_dist), np.std(agent_dist), agent_pos
	
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	
	def collect_goal_info(self, world):
		goal_pos =  np.zeros((self.num_agents, 2)) # create a zero vector with the size of the number of goal and positions of dim 2
		count = 0
		for goal in world.landmarks:
			# print("goal" , goal.name, "at", goal.state.p_pos)
			goal_pos[count]= goal.state.p_pos
			count +=1
		return goal_pos

	def graph_observation(self, agent:Agent, world:World) -> Tuple[arr, arr]:
		"""
			FIXME: Take care of the case where edge_list is empty
			Returns: [node features, adjacency matrix]
			• Node features (num_entities, num_node_feats):
				If `global`: 
					• node features are global [pos, vel, goal, entity-type]
					• edge features are relative distances (just magnitude)
					NOTE: for `landmarks` and `obstacles` the `goal` is 
							the same as its position
				If `relative`:
					• node features are relative [pos, vel, goal, entity-type] to ego agents
					• edge features are relative distances (just magnitude)
					NOTE: for `landmarks` and `obstacles` the `goal` is 
							the same as its position
			• Adjacency Matrix (num_entities, num_entities)
				NOTE: using the distance matrix, need to do some post-processing
				If `global`:
					• All close-by entities are connectd together
				If `relative`:
					• Only entities close to the ego-agent are connected
			
		"""
		# num_entities = len(world.entities)
		# node observations
		node_obs = []
		if world.graph_feat_type == 'global':
			for i, entity in enumerate(world.entities):
				node_obs_i = self._get_entity_feat_global(entity, world)
				node_obs.append(node_obs_i)
		elif world.graph_feat_type == 'relative':
			for i, entity in enumerate(world.entities):
				node_obs_i = self._get_entity_feat_relative(agent, entity, world)

				node_obs.append(node_obs_i)

		node_obs = np.array(node_obs)
		adj = world.cached_dist_mag

		return node_obs, adj

	def update_graph(self, world:World):
		"""
			Construct a graph from the cached distances.
			Nodes are entities in the environment
			Edges are constructed by thresholding distances
		"""
		dists = world.cached_dist_mag
		# just connect the ones which are within connection 
		# distance and do not connect to itself
		connect = np.array((dists <= self.max_edge_dist) * \
							(dists > 0)).astype(int)
		sparse_connect = sparse.csr_matrix(connect)
		sparse_connect = sparse_connect.tocoo()
		row, col = sparse_connect.row, sparse_connect.col
		edge_list = np.stack([row, col])
		world.edge_list = edge_list
		if world.graph_feat_type == 'global':
			world.edge_weight = dists[row, col]
		elif world.graph_feat_type == 'relative':
			world.edge_weight = dists[row, col]
	
	def _get_entity_feat_global(self, entity:Entity, world:World) -> arr:
		"""
			Returns: ([velocity, position, goal_pos, entity_type])
			in global coords for the given entity
		"""
		pos = entity.state.p_pos
		vel = entity.state.p_vel
		if 'agent' in entity.name:
			# goal_pos = world.get_entity('landmark', self.goal_match_index[entity.id]).state.p_pos
			goal_pos = self.expected_poses[entity.id]
			# print("goal_pos",goal_pos)
			entity_type = entity_mapping['agent']
		elif 'landmark' in entity.name:
			goal_pos = pos
			entity_type = entity_mapping['landmark']
		elif 'obstacle' in entity.name:
			goal_pos = pos
			entity_type = entity_mapping['obstacle']
		else:
			raise ValueError(f'{entity.name} not supported')

		return np.hstack([vel, pos, goal_pos, entity_type])

	def _get_entity_feat_relative(self, agent:Agent, entity:Entity, world:World) -> arr:
		"""
			Returns: ([velocity, position, goal_pos, entity_type])
			in coords relative to the `agent` for the given entity
		"""
		agent_pos = agent.state.p_pos
		agent_vel = agent.state.p_vel
		entity_pos = entity.state.p_pos
		entity_vel = entity.state.p_vel
		rel_pos = entity_pos - agent_pos
		rel_vel = entity_vel - agent_vel
		# print("entity", entity.name, "rel_vel", rel_vel, entity_vel,agent_vel)
		if 'agent' in entity.name:
			# goal_pos = world.get_entity('landmark',  0).state.p_pos
			# goal_pos = self.expected_poses[entity.id]
			# print("RELgoal_pos",goal_pos)

			### inefficient loops
			# print("graph obs expected_poses",self.expected_poses)
			world.dists = np.array([np.linalg.norm(entity.state.p_pos - l) for l in self.expected_poses])
			if np.min(world.dists) < self.min_dist_thresh:
				# If the minimum distance is already less than self.min_dist_thresh, use the previous goal.
				self.expected_poses_occupied[np.argmin(world.dists)] = 1

				goal_pos = self.expected_poses[np.argmin(world.dists)]
				goal_occupied = np.array([self.expected_poses_occupied[np.argmin(world.dists)]])
			else:
				unoccupied_goals = self.expected_poses[self.expected_poses_occupied==0]
				if len(unoccupied_goals) > 0:
					# if goals are unoccupied, match based on bipartite matching
					self.dists = np.array(
					[
						[np.linalg.norm(a.state.p_pos - pos) for pos in self.expected_poses]
						for a in world.agents
					]
					)
					# self.delta_dists = self._bipartite_min_dists(self.dists)
					# print("delta_dists",self.delta_dists)
					_, goals = linear_sum_assignment(self.dists)
					# print("ri",agents,"ci",goals)
					goal_pos = self.expected_poses[goals[agent.id]]
					goal_occupied = np.array([self.expected_poses_occupied[goals[agent.id]]])
				else:
					# Handle the case when all goals are occupied.
					# You can choose a default action here or raise an exception.
					goal_pos = entity.state.p_pos
					self.expected_poses_occupied = np.zeros(self.num_agents)
					goal_occupied = np.array([self.expected_poses_occupied[agent.id]])
			# print("REL2goal_pos",goal_pos)
			rel_goal_pos = goal_pos - agent_pos
			entity_type = entity_mapping['agent']
		elif 'landmark' in entity.name:
			# print("LANDMARK ####################")
			rel_goal_pos = rel_pos
			goal_occupied = np.array([1])
			entity_type = entity_mapping['landmark']
		elif 'obstacle' in entity.name:
			rel_goal_pos = rel_pos
			goal_occupied = np.array([1])
			entity_type = entity_mapping['obstacle']
		elif 'wall' in entity.name:
			rel_goal_pos = rel_pos
			goal_occupied = np.array([1])
			entity_type = entity_mapping['wall']
			## get wall corner point's relative position
			# print("wall", entity.name, entity.endpoints, entity.orient,entity.width, entity.axis_pos,entity.axis_pos+entity.width/2)
			# print("agent", agent_pos)
			wall_o_corner = np.array([entity.endpoints[0],entity.axis_pos+entity.width/2]) - agent_pos
			wall_d_corner = np.array([entity.endpoints[1],entity.axis_pos-entity.width/2]) - agent_pos
			return np.hstack([rel_vel, rel_pos, rel_goal_pos,goal_occupied,wall_o_corner,wall_d_corner, entity_type])
			# return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type,wall_o_corner])#,wall_d_corner])

		else:
			raise ValueError(f'{entity.name} not supported')
		# print("rel_vel", rel_vel, "rel_pos", rel_pos, "rel_goal_pos", rel_goal_pos)
		return np.hstack([rel_vel, rel_pos, rel_goal_pos,goal_occupied,rel_pos,rel_pos, entity_type])
		# return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type,rel_pos])#,rel_pos])



# actions: [None, ←, →, ↓, ↑, comm1, comm2]
if __name__ == "__main__":

	from multiagent.environment import MultiAgentGraphEnv
	from multiagent.policy import InteractivePolicy

	# makeshift argparser
	class Args:
		def __init__(self):
			self.num_agents:int=3
			self.world_size=2
			self.num_scripted_agents=0
			self.num_obstacles:int=3
			self.collaborative:bool=False 
			self.max_speed:Optional[float]=2
			self.collision_rew:float=5
			self.goal_rew:float=50
			self.min_dist_thresh:float=0.1
			self.use_dones:bool=True
			self.episode_length:int=25
			self.max_edge_dist:float=1
			self.graph_feat_type:str='relative'
			self.fair_wt=1
			self.fair_rew=1
	args = Args()

	scenario = Scenario()
	# create world
	world = scenario.make_world(args)
	# create multiagent environment
	env = MultiAgentGraphEnv(world=world, reset_callback=scenario.reset_world, 
						reward_callback=scenario.reward, 
						observation_callback=scenario.observation, 
						graph_observation_callback=scenario.graph_observation,
						info_callback=scenario.info_callback, 
						done_callback=scenario.done,
						id_callback=scenario.get_id,
						update_graph=scenario.update_graph,
						shared_viewer=False)
	# render call to create viewer window
	env.render()
	# create interactive policies for each agent
	policies = [InteractivePolicy(env,i) for i in range(env.n)]
	# execution loop
	obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
	stp=0

	prev_rewards = []
	while True:
		# query for action from each agent's policy
		act_n = []
		dist_mag = env.world.cached_dist_mag

		for i, policy in enumerate(policies):
			act_n.append(policy.action(obs_n[i]))
		# step environment
		# print(act_n)
		obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)
		prev_rewards= reward_n
		# print("rewards",reward_n)
		# print("node_obs", node_obs_n[0].shape)
		# print(obs_n[0].shape, node_obs_n[0].shape, adj_n[0].shape)

		# print(obs_n[0], node_obs_n[0], adj_n[0])
		# render all agent views
		env.render()
		stp+=1
		# display rewards
