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

import os,sys
sys.path.append(os.path.abspath(os.getcwd()))
from scipy.optimize import linear_sum_assignment

from multiagent.core import World, Agent, Landmark, Entity, Wall
from multiagent.scenario import BaseScenario


entity_mapping = {'agent': 0, 'landmark': 1, 'obstacle':2, 'wall':3}
def leaky_ReLU(x):
  data = np.max(max(0.01*x,x))
  return np.array(data, dtype=float)
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
		self.min_obs_dist = args.min_obs_dist
		self.use_dones = args.use_dones
		# print("use_dones", self.use_dones)
		self.episode_length = args.episode_length

		# fairness args
		self.fair_wt = args.fair_wt
		self.fair_rew = args.fair_rew


		# create heatmap matrix to determine the goal agent pairs
		self.goal_reached = -1*np.ones(self.num_agents)
		self.wrong_goal_reached = np.zeros(self.num_agents)
		self.goal_matched = np.zeros(self.num_agents)


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
		# self.num_obstacles_per_wall = np.int32(1*self.wall_length/world.agents[0].size)
		# self.num_obstacles_per_wall = 2
		# print("num_obstacles_per_wall",self.num_obstacles_per_wall)
		self.num_walls = args.num_walls
		world.walls = [Wall() for i in range(self.num_walls)]
		for i, wall in enumerate(world.walls):
			wall.id = i
			wall.name = f'wall {i}'
			wall.collide = True
			wall.movable = False
			wall.global_id = global_id
			global_id += 1
		# num_walls = 1

		# # add wall obstacles
		# world.wall_obstacles = [Landmark() for i in range(self.num_obstacles_per_wall*num_walls)]
		# for i, wall_obstacles in enumerate(world.wall_obstacles):
		# 	wall_obstacles.name = f'obstacle {i}'
		# 	wall_obstacles.collide = False
		# 	wall_obstacles.movable = False
		# 	wall_obstacles.global_id = global_id
		# 	global_id += 1
		# make initial conditions
			
		self.zeroshift = args.zeroshift
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
		world.agent_dist_traveled = np.zeros(self.num_agents)
		self.agent_dist_traveled = np.zeros(self.num_agents)
		self.agent_time_taken = np.zeros(self.num_agents)
		self.goal_history = -1*np.ones((self.num_agents))
		self.goal_reached = -1*np.ones(self.num_agents)
		self.optimal_match_index = np.arange(self.num_agents)
		wall_length = np.random.uniform(0.2, 0.8)
		# print("wall_length",wall_length)
		self.wall_length = wall_length * self.world_size/4


		#################### set colours ####################
		# set colours for agents
		for i, agent in enumerate(world.agents):
			# agent.color = np.array([0.35, 0.35, 0.85])
			# if i%3 == 0:
			# 	agent.color = np.array([0.35, 0.35, 0.85])
			# elif i%3 == 1:
			# 	agent.color = np.array([0.85, 0.35, 0.35])
			# else:
			# 	agent.color = np.array([0.35, 0.85, 0.35])
			agent.state.p_dist = 0.0
			agent.state.time = 0.0
		# set colours for scripted agents
		for i, agent in enumerate(world.scripted_agents):
			agent.color = np.array([0.15, 0.15, 0.15])
		# set colours for landmarks
		for i, landmark in enumerate(world.landmarks):
			if i%3 == 0:
				landmark.color = np.array([0.35, 0.35, 0.85])
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
		###### set random positions for entities ###########
		# set random static obstacles first
		for obstacle in world.obstacles:
			obstacle.state.p_pos = 0.8 * np.random.uniform(-self.world_size/2, 
															self.world_size/2, 
															world.dim_p)
			obstacle.state.p_vel = np.zeros(world.dim_p)
		#####################################################
		# # set random static wall obstacles first
		# for wall_obstacle in world.wall_obstacles:
		# 	wall_obstacle.state.p_pos = 0.8 * np.random.uniform(-self.world_size/2, 
		# 													self.world_size/2, 
		# 													world.dim_p)
		# 	wall_obstacle.state.p_vel = np.zeros(world.dim_p)



		############## create wall positions #################
		wall_position = np.random.uniform(0.2, 0.9)
		# wall_position = 0.5
		wall_axis  = np.array([wall_position * self.world_size/2, -wall_position * self.world_size/2])
		wall_obst_count = 0
		# print("wall_obst_count",wall_obst_count)
		# set wall positions
		for i , wall in enumerate(world.walls):
			# print("wall",i)
			# wall.orient='V'
			#select wall orientation randomly for every episode
			wall.orient = np.random.choice(['H','V'])



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
				# Calculate the interval between obstacles
				# interval = (x_max - x_min) / (self.num_obstacles_per_wall + 1)
				# # print(1+wall_obst_count, self.num_obstacles_per_wall + 1+wall_obst_count)
				# for j in range(1+wall_obst_count, self.num_obstacles_per_wall + 1+wall_obst_count):
				# 	wall_obstacle = world.wall_obstacles[j-1]
				# 	wall_obstacle.state.p_pos = np.array([x_min + j * interval, y])
				# 	# print("wall_obstacle.state.p_pos",wall_obstacle.state.p_pos)
				# 	wall_obstacle.state.p_vel = np.zeros(world.dim_p)
				# 	wall_obst_count += 1
				# 	# print("wall_obst_count",wall_obst_count)
			elif wall.orient == 'V':
				# Vertical wall
				x = wall.axis_pos
				y_min, y_max = wall.endpoints
				y = (y_min + y_max) / 2
				wall.state.p_pos = np.array([x, y])  # Set the physical position
				wall.state.p_vel = np.zeros(world.dim_p)  # Set the physical velocity
				# Calculate the interval between obstacles
				# print(1+wall_obst_count, self.num_obstacles_per_wall + 1+wall_obst_count)
				# interval = (y_max - y_min) / (self.num_obstacles_per_wall + 1)
				# y_pos_reset = 0
				# for j in range(1+wall_obst_count, self.num_obstacles_per_wall + 1+wall_obst_count):
				# 	y_pos_reset += 1
				# 	wall_obstacle = world.wall_obstacles[j-1]
				# 	wall_obstacle.state.p_pos = np.array([x, y_min + y_pos_reset * interval])
				# 	# print(y_min + y_pos_reset * interval)
				# 	# print("wall_obstacle.state.p_pos",wall_obstacle.state.p_pos)
				# 	wall_obstacle.state.p_vel = np.zeros(world.dim_p)
				# 	wall_obst_count += 1
				# 	# print("wall_obst_count",wall_obst_count)

		# print("wall_obst_count",wall_obst_count)
		# print("len(world.wall_obstacles)",len(world.wall_obstacles))
		#####################################################



		# set agents at random positions not colliding with obstacles
		num_agents_added = 0
		agents_added = []
		boundary_thresh = 0.9
		uniform_pos = np.linspace(-boundary_thresh*self.world_size/2, boundary_thresh*self.world_size/2, self.num_agents)
		# print("uni",uniform_pos)
		while True:
			# print(uniform_pos)
			if num_agents_added == self.num_agents:
				break
			# # for random pos
			random_pos = np.random.uniform(-self.world_size/2, 
											self.world_size/2, 
											world.dim_p)
			line_pos = random_pos

			# # for uniform pos
			# line_pos = [uniform_pos[num_agents_added]]
			# ##uniform_pos = np.linspace(-self.world_size/2, self.world_size/2, self.num_agents)
			# line_pos = np.insert(np.array(line_pos), -1, -boundary_thresh*self.world_size/2)
			# # # print("agent pos",line_pos)
			# # random_pos= line_pos
			agent_size = world.agents[num_agents_added].size
			obs_collision = self.is_obstacle_collision(line_pos, agent_size, world)
			# goal_collision = self.is_goal_collision(uniform_pos, agent_size, world)
			# print("obs_collision",obs_collision)
			# print("num_agents_added",num_agents_added)
			agent_collision = self.check_agent_collision(line_pos, agent_size, agents_added)
			if not obs_collision and not agent_collision:
				world.agents[num_agents_added].state.p_pos = line_pos
				world.agents[num_agents_added].state.p_vel = np.zeros(world.dim_p)
				world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
				world.agents[num_agents_added].status = False
				agents_added.append(world.agents[num_agents_added])
				num_agents_added += 1
			# print(num_agents_added)
		# agent_pos = [agent.state.p_pos for agent in world.agents]
		#####################################################
		


		# set landmarks (goals) at random positions not colliding with obstacles 
		# and also check collisions with already placed goals
		num_goals_added = 0
		# goals_added = []
		uniform_pos = np.linspace(0, boundary_thresh*self.world_size/2, self.num_agents)

		while True:
			if num_goals_added == self.num_agents:
				break
			# # for random pos
			random_pos = 0.8 * np.random.uniform(-self.world_size/2, 
												self.world_size/2, 
												world.dim_p)
			line_pos = random_pos


			# # for uniform pos
			# line_pos = [uniform_pos[num_goals_added]]
			# line_pos = np.insert(line_pos, 1, self.world_size/2)
			## random_pos = line_pos
			# # print("goal Pos",line_pos)
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
				
		self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
		self.landmark_poses_occupied = np.zeros(self.num_agents)

		# goal_pos = [goal.state.p_pos for goal in world.landmarks]
		#####################################################
		# agent_pos = [agent.state.p_pos for agent in world.agents]
		############ find minimum times to goals ############
		if self.max_speed is not None:
			for agent in world.agents:
				self.min_time(agent, world)
		#####################################################
		############ update the cached distances ############
		world.calculate_distances()
		self.update_graph(world)
		####################################################
		########### set nearest goals for each agent #######
		# # bipartite matching
		world.dists = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
						   for a in world.agents])
		## Get the shortest distance landmark for each agent based on world.dists
		# print("world.dists",world.dists)
		# # # # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
		ri, ci = linear_sum_assignment(world.dists)
		# print("ri",ri, "ci",ci)
		self.optimal_match_index = ci

		if self.optimal_match_index.size == 0 or self.optimal_match_index.shape != (self.num_agents,):
			self.optimal_match_index = np.arange(self.num_agents)
		## print(x,optimal_match_index,agent.id, optimal_match_index[0], optimal_match_index[1], optimal_match_index[2])
		color_list= [np.array([0.35, 0.35, 0.85]),np.array([0.85, 0.35, 0.35]),np.array([0.35, 0.85, 0.35])]
		for i, agent in enumerate(world.agents):
			# landmark.color = np.array([0.15, 0.85, 0.15])
			agent.color = color_list[self.optimal_match_index[i]%3]
				# landmark.color = np.array([0.15, 0.85, 0.15])



	def info_callback(self, agent:Agent, world:World) -> Tuple:
		# TODO modify this 
		# rew = 0
		# collisions = 0
		# occupied_landmarks = 0
		world.dists = np.array([np.linalg.norm(agent.state.p_pos - l.state.p_pos) for l in world.landmarks])
		nearest_landmark = np.argmin(world.dists)
		dist_to_goal = world.dists[nearest_landmark]
		if dist_to_goal < self.min_dist_thresh and (nearest_landmark != self.goal_reached[agent.id] and self.goal_reached[agent.id] != -1):
			# print("AGENT", agent.id, "reached NEW goal",world.dist_left_to_goal[agent.id])
			self.goal_reached[agent.id] = nearest_landmark
			world.dist_left_to_goal[agent.id] = dist_to_goal
		
		# only update times_required for the first time it reaches the goal
		if dist_to_goal < self.min_dist_thresh and (world.times_required[agent.id] == -1):
			# print("agent", agent.id, "reached goal",world.dist_left_to_goal[agent.id])
			world.times_required[agent.id] = world.current_time_step * world.dt
			world.dists_to_goal[agent.id] = agent.state.p_dist
			world.dist_left_to_goal[agent.id] = dist_to_goal
			self.goal_reached[agent.id] = nearest_landmark
			# print("dist to goal",world.dists_to_goal[agent.id],world.dists_to_goal)
		if world.times_required[agent.id] == -1:
			# print("agent", agent.id, "not reached goal yet",world.dist_left_to_goal[agent.id])
			world.dists_to_goal[agent.id] = agent.state.p_dist
			world.dist_left_to_goal[agent.id] = dist_to_goal
		if dist_to_goal > self.min_dist_thresh and world.times_required[agent.id] != -1:
			# print("AGENT", agent.id, "left goal",dist_to_goal,world.dist_left_to_goal[agent.id])
			world.dists_to_goal[agent.id] = agent.state.p_dist
			world.times_required[agent.id] = world.current_time_step * world.dt
			world.dist_left_to_goal[agent.id] = dist_to_goal
		if dist_to_goal < self.min_dist_thresh and (nearest_landmark == self.goal_reached[agent.id]):
			# print("AGENT", agent.id, "on SAME goal",world.dist_left_to_goal[agent.id])
			## TODO: How to update time as well?
			world.dist_left_to_goal[agent.id] = dist_to_goal
			self.goal_reached[agent.id] = nearest_landmark
		if agent.collide:
			if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
				world.num_obstacle_collisions[agent.id] += 1
			for a in world.agents:
				# print("agent", a.id, a.state.p_pos)
				if a is agent:
					self.agent_dist_traveled[a.id] = a.state.p_dist
					self.agent_time_taken[a.id] = a.state.time
					# print("agent_dist_traveled",self.agent_dist_traveled)
					continue
				if self.is_collision(agent, a):
					world.num_agent_collisions[agent.id] += 1
			# print("agent_dist_traveled",self.agent_dist_traveled)




		world.dist_traveled_mean = np.mean(world.dists_to_goal)
		world.dist_traveled_stddev = np.std(world.dists_to_goal)
		# create the mean and stddev for time taken
		world.time_taken_mean = np.mean(world.times_required)
		world.time_taken_stddev = np.std(world.times_required)
		# print("agent_dist_traveled",self.agent_dist_traveled)
		# print("mean",world.dist_traveled_mean)
		# print("stddev",world.dist_traveled_stddev)
		# print("Fairness",world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001))
		# print("Hi agent", agent.id, self.agent_dist_traveled[agent.id])
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
			'Time_taken': world.times_required[agent.id],
			# Time mean and stddev
			'Time_mean': world.time_taken_mean,
			'Time_stddev': world.time_taken_stddev,
			'Time_mean_by_stddev': world.time_taken_mean/(world.time_taken_stddev+0.0001),
			# add goal reached params
			# 'Goal_reached':
			# 'Goal_matched':

		}
		if self.max_speed is not None:
			agent_info['Min_time_to_goal'] = agent.goal_min_time
		return agent_info

	# check collision of entity with obstacles
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
		# for wall_obstacle in world.wall_obstacles:
		# 	delta_pos = wall_obstacle.state.p_pos - pos
		# 	dist = np.linalg.norm(delta_pos)
		# 	dist_min = wall_obstacle.size + entity_size
		# 	if dist < dist_min:
		# 		collision = True
		# 		break		
		
		# check collision with walls
		for wall in world.walls:
			if wall.orient == 'H':
				# Horizontal wall, check for collision along the y-axis
				if 1.05*(wall.axis_pos - entity_size / 2) <= pos[1] <= 1.05*(wall.axis_pos + entity_size / 2):
					if 1.05*(wall.endpoints[0] - entity_size / 2) <= pos[0] <= 1.05*(wall.endpoints[1] + entity_size / 2):
						collision = True
						break
			elif wall.orient == 'V':
				# Vertical wall, check for collision along the x-axis
				if 1.05*(wall.axis_pos - entity_size / 2) <= pos[0] <= 1.05*(wall.axis_pos + entity_size / 2):
					if 1.05*(wall.endpoints[0] - entity_size / 2) <= pos[1] <= 1.05*(wall.endpoints[1] + entity_size / 2):
						collision = True
						# print("wall collision")
						break
		return collision


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
		agent_id = agent.id
		# get the goal associated to this agent
		landmark = world.get_entity(entity_type='landmark', id=self.optimal_match_index[agent_id])
		dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
										landmark.state.p_pos)))
		min_time = dist / agent.max_speed
		agent.goal_min_time = min_time
		return min_time

	# done condition for each agent
	def done(self, agent:Agent, world:World) -> bool:
		# if we are using dones then return appropriate done
		# print("dones", self.use_dones)
		if self.use_dones:
			if world.current_time_step >= world.world_length:
				return True
			else:
				landmark = world.get_entity('landmark', self.optimal_match_index[agent.id])
				dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
												landmark.state.p_pos)))
				if dist < self.min_dist_thresh:
					return True
				else:
					return False
		# it not using dones then return done 
		# only when episode_length is reached
		else:
			if world.current_time_step >= world.world_length:
				return True
			else:
				return False


	def _bipartite_min_dists(self, dists):
		ri, ci = linear_sum_assignment(dists)
		min_dists = dists[ri, ci]
		return min_dists

	def reward(self, agent:Agent, world:World) -> float:
	# def fairness_appended_reward(self, agent:Agent, world:World) -> float:

		rew = 0

		if world.dists_to_goal[agent.id] == -1:
			mean_dist,  std_dev_dist, _ = self.collect_dist(world)
			fairness_param = mean_dist/(std_dev_dist+0.0001)
		else:
			fairness_param = world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001)
		if agent.id == 0:

			self.dists = np.array(
				[
					[np.linalg.norm(a.state.p_pos - l) for l in self.landmark_poses]
					for a in world.agents
				]
			)
			# ## for every column in dists if the value is below min_dist_thresh then set self.landmark_poses_occupied to 1 for that index
			# mask = self.dists < self.min_dist_thresh
			# self.landmark_poses_occupied = np.any(mask, axis=0).astype(float)

			# optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
			self.delta_dists = self._bipartite_min_dists(self.dists)

			world.dists = self.delta_dists
		if self.delta_dists[agent.id] < self.min_dist_thresh:
			if agent.status ==False:
				agent.status = True
				agent.state.p_vel[0]=0.0
				agent.state.p_vel[1]=0.0
				# print("AGENT",agent.id,"goal_rew","{:.3f}".format(self.goal_rew))
				rew += self.goal_rew
				# print("agent.state.p_vel",agent.state.p_vel)
		else:
			# # print("delta_dists",self.delta_dists[agent.id])
			# world.dists = np.array([np.linalg.norm(agent.state.p_pos - l) for l in self.landmark_poses])
			# if np.min(world.dists) < self.min_dist_thresh:
			# 	if agent.status ==False:
			# 		agent.status = True
			# 		agent.state.p_vel[0]=0.0
			# 		agent.state.p_vel[1]=0.0
			# 		# print("reached wrong goal")
			# 		rew -= self.delta_dists[agent.id]
			# else:
				rew -= self.delta_dists[agent.id]

		if agent.collide:
			for a in world.agents:
				# do not consider collision with itself
				if a.id == agent.id:
					continue
				if self.is_collision(a, agent):
					rew -= self.collision_rew


			if self.is_obstacle_collision(pos=agent.state.p_pos,
										entity_size=agent.size, world=world):
				rew -= self.collision_rew
		# ## add fairness parameter

		# fairness_param = mean_dist/(std_dev_dist+0.0001)
		# ## print(f"{agent.id}, {mean_dist}, {std_dev_dist} fairness:{fairness_param:0.3f}") # f'{a:.2f}'

		# fair_weight = self.fair_wt/10.0
		# fair_rew = self.fair_rew/10.0

		# if fairness_param > 10.0:
		# 	rew += fair_rew*fair_weight
		# elif fairness_param < 1.0:
		# 	rew -= fair_rew*fair_weight
		# else: rew -= fairness_param*fair_weight
		# print("self.fair_rew", self.fair_rew)
		scaled_input = fairness_param
		tanh_output = np.tanh(scaled_input-self.zeroshift)
		fair_rew = self.fair_rew * tanh_output
		# reduce the negative reward if the fairness is not met
		if fair_rew < -self.fair_rew:
			fair_rew = -self.fair_rew
				
		# # use leaky relu
		# shifted_fairness = fairness_param - self.zeroshift
		# # print("shifted_fairness",shifted_fairness)
		# fair_rew = leaky_ReLU(shifted_fairness)
		# # set the fair_rew to self.fair_rew if it is greater than self.fair_rew in an optimal way
		# if fair_rew > self.fair_rew:
		# 	fair_rew = self.fair_rew
		## print("fair_rew",fair_rew)
			
		
		rew += fair_rew 
		return np.clip(rew, -2*self.collision_rew, self.goal_rew+self.fair_rew)
	# def observation(self, agent:Agent, world:World) -> arr:
	# 	"""
	# 		Return:
	# 			[agent_vel, agent_pos, goal_pos, goal_occupied]
	# 	"""
	# 	world.dists = np.array([np.linalg.norm(agent.state.p_pos - l) for l in self.landmark_poses])
	# 	min_dist = np.min(world.dists)
	# 	if min_dist < self.min_dist_thresh:
	# 		# If the minimum distance is already less than self.min_dist_thresh, use the previous goal.
	# 		chosen_goal = np.argmin(world.dists)
	# 		agents_goal = self.landmark_poses[chosen_goal]
	# 		self.landmark_poses_occupied[chosen_goal] = 1
	# 		goal_occupied = np.array([self.landmark_poses_occupied[chosen_goal]])
	# 	else:
	# 		# create another variable to store which goals are uncoccupied using an index of 0, 1 or 2 based on self.landmark_poses
	# 		unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!=1]
	# 		unoccupied_goals_indices = np.where(self.landmark_poses_occupied != 1)[0]
	# 		if len(unoccupied_goals) > 0:
	# 			## determine which goal from self.landmark_poses is this chosen unocccupied goal
	# 			## use the index of the unoccupied goal to get the goal from self.landmark_poses
	# 			min_dist_goal = np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
	# 			agents_goal = unoccupied_goals[min_dist_goal]
	# 			min_dist = np.min(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
	# 			# print("min_dist",min_dist)
	# 			self.landmark_poses_occupied[unoccupied_goals_indices[min_dist_goal]] = np.maximum(1.0-min_dist,0.0)
	# 			# print("goal_occupied",self.landmark_poses_occupied[unoccupied_goals_indices[min_dist_goal]])
	# 			## check if the goal is occupied
	# 			goal_occupied = np.array(np.maximum(1.0-min_dist,0.0))
	# 		else:
	# 			# Handle the case when all goals are occupied.
	# 			agents_goal = agent.state.p_pos
	# 			self.landmark_poses_occupied = np.zeros(self.num_agents)
	# 			goal_occupied = np.array([self.landmark_poses_occupied[agent.id]])
		
	# 	goal_pos = [agents_goal - agent.state.p_pos]
	# 	return np.concatenate([agent.state.p_vel, agent.state.p_pos] + goal_pos + goal_occupied)
	def observation(self, agent:Agent, world:World) -> arr:
		"""
			Return:
				[agent_vel, agent_pos, goal_pos, goal_occupied]
		"""

		world.dists = np.array([np.linalg.norm(agent.state.p_pos - l) for l in self.landmark_poses])
		min_dist = np.min(world.dists)
		sorted_indices = np.argsort(world.dists)  # Get indices that would sort the distances
		# print("Agent! sorted indices", sorted_indices)
		top_two_indices = sorted_indices[:2]  # Get indices of top two closest distances
		second_closest_goal = self.landmark_poses[top_two_indices[1]]
		# get goal occupied flag for that goal
		second_closest_goal_occupied = np.array([self.landmark_poses_occupied[top_two_indices[1]]])
		if min_dist < self.min_obs_dist:
			# If the minimum distance is already less than self.min_dist_thresh, use the previous goal.
			# If the minimum distance is already less than self.min_obs_dist, use the previous goal.
			chosen_goal = np.argmin(world.dists)
			agents_goal = self.landmark_poses[chosen_goal]
			if min_dist < self.min_dist_thresh:
				self.landmark_poses_occupied[chosen_goal] = 1.0
				self.goal_history[chosen_goal] = agent.id

				# print("obs AT GOAL",agent.id,np.min(world.dists), "goal_occupied",self.landmark_poses_occupied[chosen_goal])

			else:
				# goal_proximity is finding how many agents are nearthi chosen goal
				goal_proximity = np.array([np.linalg.norm(agents_goal - agent.state.p_pos)  for agent in world.agents])
				# print("Agent",agent.id,"chosen_goal", chosen_goal, "goal_proximity",goal_proximity, "flags",self.landmark_poses_occupied, "history",self.goal_history)
				closest_dist_to_goal = np.min(goal_proximity)
				# agent veered off the goal
				if self.landmark_poses_occupied[chosen_goal] == 1.0:

					# if there are no agents on the goal, then the agent can take the goal and change the occupancy value
					if np.any(goal_proximity < self.min_dist_thresh):
						# print("Agent!", "{:.0f}".format(self.goal_history[chosen_goal]), " is already at goal", "{:.0f}".format(chosen_goal), "min_dist", "{:.3f}".format(min_dist), "occupied flags",  self.landmark_poses_occupied, "history", self.goal_history)
						pass
					else:
						# self.landmark_poses_occupied[chosen_goal] = 1.0 - min_dist
						# print("FLag about to update",self.landmark_poses_occupied[chosen_goal])
						self.landmark_poses_occupied[chosen_goal] = 1.0-closest_dist_to_goal
						# print("Flag updated",self.landmark_poses_occupied[chosen_goal])
						# print("Agent!", "{:.0f}".format(self.goal_history[chosen_goal]), " veered off the goal", "{:.0f}".format(chosen_goal), "min_dist", "{:.3f}".format(min_dist), "occupied flags",  self.landmark_poses_occupied, "history", self.goal_history)

				# another agent already at goal, can't overwrite the flag
				elif self.landmark_poses_occupied[chosen_goal] != 1.0:
					# self.landmark_poses_occupied[chosen_goal] = 1.0 - min_dist
					# print("ag! ", agent.id, "NEAR GOAL", chosen_goal, "dist", "{:.3f}".format(min_dist), "goal_occupied_flag", "{:.3f}".format(self.landmark_poses_occupied[chosen_goal]), "occupied flags", self.landmark_poses_occupied, "history", self.goal_history)
					# print("FLag about to update",self.landmark_poses_occupied[chosen_goal])
					self.landmark_poses_occupied[chosen_goal] = 1.0-closest_dist_to_goal
					# print("Flag updated",self.landmark_poses_occupied[chosen_goal])
			# self.landmark_poses_occupied[chosen_goal] = 1
			goal_occupied = np.array([self.landmark_poses_occupied[chosen_goal]])
			goal_history = self.goal_history[chosen_goal]
			# print("ob1 chosen_goal",chosen_goal, "agents_goal",agents_goal, "goal_occupied",goal_occupied)
		else:
			# create another variable to store which goals are uncoccupied using an index of 0, 1 or 2 based on self.landmark_poses

			unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= 1]
			# print("Self.landmark_poses",self.landmark_poses)
			# print("landmark_poses_occupied",self.landmark_poses_occupied)
			# print("unoccupied_goals",unoccupied_goals)
			unoccupied_goals_indices = np.where(self.landmark_poses_occupied != 1)[0]
			# print("unoccupied_goals_indices",unoccupied_goals_indices)
			if len(unoccupied_goals) > 0:
				# # if goals are unoccupied, match based on bipartite matching
				# self.dists = np.array(
				# [
				# 	[np.linalg.norm(a.state.p_pos - pos) for pos in self.landmark_poses]
				# 	for a in world.agents
				# ]
				# )
				# # self.delta_dists = self._bipartite_min_dists(self.dists)
				# # print("delta_dists",self.delta_dists)
				# _, goals = linear_sum_assignment(self.dists)
				# # print("ri",agents,"ci",goals)
				# agents_goal = self.landmark_poses[goals[agent.id]]


				## use closest goal
				# print("unoccupied_goals",unoccupied_goals)
				# print(np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1)), unoccupied_goals[np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))])

				## determine which goal from self.landmark_poses is this chosen unocccupied goal
				## use the index of the unoccupied goal to get the goal from self.landmark_poses
				min_dist_goal = np.argmin(np.linalg.norm(agent.state.p_pos - unoccupied_goals, axis=1))
				# print("min_dist_goal",min_dist_goal,unoccupied_goals_indices[min_dist_goal])
				agents_goal = unoccupied_goals[min_dist_goal]
				## check if the goal is occupied
				goal_occupied = np.array([self.landmark_poses_occupied[unoccupied_goals_indices[min_dist_goal]]])
				goal_history = self.goal_history[unoccupied_goals_indices[min_dist_goal]]
				# print("ob2 goal_occupied",goal_occupied)
				# print("agents_goal",agents_goal)

			else:
				# Handle the case when all goals are occupied.
				agents_goal = agent.state.p_pos
				self.landmark_poses_occupied = np.zeros(self.num_agents)
				goal_history = self.goal_history[agent.id]
				goal_occupied = np.array([self.landmark_poses_occupied[agent.id]])
		
		goal_pos = agents_goal - agent.state.p_pos
		rel_second_closest_goal = second_closest_goal - agent.state.p_pos

		goal_history = np.array([goal_history])
		# print("agent", agent.id,"goal_pos",agents_goal, "goal_occupied",np.round(goal_occupied,4), "min_dist",min_dist)
		return np.concatenate((agent.state.p_vel, agent.state.p_pos, goal_pos,goal_occupied,goal_history, rel_second_closest_goal,second_closest_goal_occupied))
		# if world.dists_to_goal[agent.id] == -1:
		# 	mean_dist,  std_dev_dist, _ = self.collect_dist(world)
		# 	fairness_param = mean_dist/(std_dev_dist+0.0001)
		# 	# print("mean", mean_dist, "std_dev_dist", std_dev_dist, "fairness", fairness_param)
		# else:
		# 	fairness_param = world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001)
		# # print("fairness_param",fairness_param)

		# # ## convert fairness param to a numpy array
		# fairness_param = np.array([fairness_param])
		# # print("fairness_param",np.concatenate((agent.state.p_vel, agent.state.p_pos, goal_pos,goal_occupied,goal_history,fairness_param)))
		# return np.concatenate((agent.state.p_vel, agent.state.p_pos, goal_pos,goal_occupied,goal_history,fairness_param))# + goal_occupied+ goal_history)

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
			# if a is agent: continue
		# print("num_agents",self.num_agents)
		# for i, entity in enumerate(world.entities):
		# 	if agent.name == entity.name:
		# 		print("ag", agent.name)
		# 		agent_names.append(agent.name)
		# 		agent_rewards[count] = self.reward(agent, world)
		# 		print("Rew", agent_rewards[count])
		# 		count +=1

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
		output: mean distance, standard deviation of distance, and positions of agents
		"""
		# agent_names = [agent.name for agent in world.agents]  # Collect agent names
		agent_dist = np.array([agent.state.p_dist for agent in world.agents])  # Collect distances
		agent_pos = np.array([agent.state.p_pos for agent in world.agents])  # Collect positions

		mean_dist = np.mean(agent_dist)
		std_dev_dist = np.std(agent_dist)
		
		return mean_dist, std_dev_dist, agent_pos
	
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
		## agent_names, agent_rewards = self.collect_rewards(world)
		## reward_sum = agent_rewards.max() # np.sum(agent_rewards)
		# if world.dists_to_goal[agent.id] == -1:
		# 	# print("dist_to_goal",world.dists_to_goal[agent.id])
		# 	mean_dist,  std_dev_dist, _ = self.collect_dist(world)
		# 	fairness_param = mean_dist/(std_dev_dist+0.0001)
		# 	# print("mean", mean_dist, "std_dev_dist", std_dev_dist, "fairness", fairness_param)
		# else:
		# 	# print("dist_to_goal",world.dists_to_goal[agent.id])
		# 	fairness_param = world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001)
		# 		## convert fairness param to a numpy array
		# fairness_param = np.array([fairness_param])
		fairness_param = 0.0
		# print("fairness_param",fairness_param)
		# print()
		# goal_pos = self.collect_goal_info(world)
		# print("goal Pos are",goal_pos, goal_pos.flatten())
		if world.graph_feat_type == 'global':
			for i, entity in enumerate(world.entities):
				# print("i", i, entity.name, agent.name, "dist",entity.state.p_dist)
				# if agent.name == entity.name  :
				# 	# print("found an agent", agent.name, agent.state.p_pos) #self.reward(agent, world)
				# 	# message_reward = self.modify_reward(agent_rewards[agent_names.index(agent.name)] )
				# 	message_reward = 0.0 # setting to zero to avoid computations
				# 	fairness_param = 0.0 # mean_dist/(std_dev_dist+0.0001)
				# 	# print("fairness_param",fairness_param)
				# 	# goal_info_param = goal_pos.flatten()

				# else:
				# 	# print("no agent")
				# 	message_reward = 0.0
				# 	fairness_param =0.0
				# 	# goal_info_param = 0.0

				node_obs_i = self._get_entity_feat_global(entity, world)

				# node_obs_i = np.insert(node_obs_i, -1, message_reward)
				# node_obs_i = np.insert(node_obs_i, -1, fairness_param)
				# node_obs_i = np.insert(node_obs_i, -1, goal_info_param)


				# message_reward = np.reshape(message_reward, (1,))
				# print("node",node_obs_i,message_reward)
				# node_obs_i = np.concatenate((node_obs_i, message_reward), axis=0)
				# print("node_obs_i",node_obs_i.shape)
				node_obs.append(node_obs_i)
		elif world.graph_feat_type == 'relative':
			for i, entity in enumerate(world.entities):
				# print("i", i, self.reward(agent, world))
				# if agent.name == entity.name:
				# # 	# print("found an agent")
				# # 	# message_reward = self.modify_reward(agent_rewards[agent_names.index(agent.name)] )
				# # 	message_reward = 0.0 # setting to zero to avoid computations
				# 	# fairness_param = mean_dist/(std_dev_dist+0.0001)
				# # 	# print("fairness_param",fairness_param)		
				# else:
				# # 	# print("no agent")
				# # 	message_reward = 0.0
				# 	fairness_param =0.0
				node_obs_i = self._get_entity_feat_relative(agent, entity, world, fairness_param)
				# # node_obs_i = np.insert(node_obs_i, -1, message_reward)
				# node_obs_i = np.insert(node_obs_i, -1, fairness_param)
				# message_reward = np.reshape(message_reward, (1,))
				# print("node_obs_i",node_obs_i.shape,message_reward)
				# node_obs_i = np.concatenate((node_obs_i, message_reward), axis=0)
				# print("node_obs_i",node_obs_i.shape, node_obs_i)
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
			goal_pos = world.get_entity('landmark', self.optimal_match_index[entity.id]).state.p_pos
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

	def _get_entity_feat_relative(self, agent:Agent, entity:Entity, world:World, fairness_param: np.ndarray) -> arr:
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
		# print("entity", entity.name, "rel_pos", rel_pos, "rel_vel", rel_vel)
		if 'agent' in entity.name:
			world.dists = np.array([np.linalg.norm(entity.state.p_pos - l) for l in self.landmark_poses])
			min_dist = np.min(world.dists)
			if min_dist < self.min_obs_dist:
				# If the minimum distance is already less than self.min_dist_thresh, use the previous goal.
				chosen_goal = np.argmin(world.dists)
				goal_pos = self.landmark_poses[chosen_goal]
				# if min_dist < self.min_dist_thresh:
				# 	self.landmark_poses_occupied[chosen_goal] = 1
				# 	# print("gr AT GOAL",entity.id,np.min(world.dists), "goal_occupied",self.landmark_poses_occupied[chosen_goal])

				# else:

				# 	# another agent already at goal, can't overwrite the flag
				# 	if self.landmark_poses_occupied[chosen_goal] != 1.0:
				# 		self.landmark_poses_occupied[chosen_goal] = 1.0-min_dist
					# print("gr CLOSE",entity.id, np.min(world.dists), "goal_occupied",self.landmark_poses_occupied[chosen_goal])

				# self.landmark_poses_occupied[chosen_goal] = 1
				goal_history = self.goal_history[chosen_goal]

				goal_occupied = np.array([self.landmark_poses_occupied[chosen_goal]])

			else:
				unoccupied_goals = self.landmark_poses[self.landmark_poses_occupied!= 1]
				# print("graph unoccupied_goals",unoccupied_goals)
				unoccupied_goals_indices = np.where(self.landmark_poses_occupied != 1)[0]
				# print("graph unoccupied_goals_indices",unoccupied_goals_indices)
				if len(unoccupied_goals) > 0:
					# # if goals are unoccupied, match based on bipartite matching
					# self.dists = np.array(
					# [
					# 	[np.linalg.norm(a.state.p_pos - pos) for pos in self.landmark_poses]
					# 	for a in world.agents
					# ]
					# )
					# # self.delta_dists = self._bipartite_min_dists(self.dists)
					# # print("delta_dists",self.delta_dists)
					# _, goals = linear_sum_assignment(self.dists)
					# # print("ri",agents,"ci",goals)
					# goal_pos = self.landmark_poses[goals[agent.id]]
					# goal_occupied = np.array([self.landmark_poses_occupied[goals[agent.id]]])

					## use closest goal

					## determine which goal from self.landmark_poses is this chosen unocccupied goal
					## use the index of the unoccupied goal to get the goal from self.landmark_poses
					min_dist_goal = np.argmin(np.linalg.norm(entity.state.p_pos - unoccupied_goals, axis=1))
					goal_pos = unoccupied_goals[min_dist_goal]
					## check if the goal is occupied
					goal_occupied = np.array([self.landmark_poses_occupied[unoccupied_goals_indices[min_dist_goal]]])
					goal_history = self.goal_history[unoccupied_goals_indices[min_dist_goal]]
					# print("graph goal_pos",goal_pos, "goal_occupied",goal_occupied)

				else:
					# Handle the case when all goals are occupied.
					goal_pos = entity.state.p_pos
					self.landmark_poses_occupied = np.zeros(self.num_agents)
					goal_occupied = np.array([self.landmark_poses_occupied[entity.id]])
					goal_history = self.goal_history[agent.id]
					
					# print("else",goal_pos, goal_occupied)
			goal_history = np.array([goal_history])

			rel_goal_pos = goal_pos - agent_pos
			entity_type = entity_mapping['agent']
		elif 'landmark' in entity.name:
			rel_goal_pos = rel_pos
			goal_occupied = np.array([1])
			goal_history = entity.id if entity.id != None else 0

			# rel_goal_pos = np.repeat(rel_pos, self.num_landmarks)
			entity_type = entity_mapping['landmark']
		elif 'obstacle' in entity.name:
			rel_goal_pos = rel_pos
			goal_occupied = np.array([1])
			goal_history = entity.id if entity.id != None else 0


			# rel_goal_pos = np.repeat(rel_pos, self.num_landmarks)

			entity_type = entity_mapping['obstacle']
		elif 'wall' in entity.name:
			rel_goal_pos = rel_pos
			goal_occupied = np.array([1])
			goal_history = entity.id if entity.id != None else 0
			# rel_goal_pos = np.repeat(rel_pos, self.num_landmarks)
			entity_type = entity_mapping['wall']
			## get wall corner point's relative position
			## print("wall", entity.name, entity.endpoints, entity.orient,entity.width, entity.axis_pos,entity.axis_pos+entity.width/2)
			## print("agent", agent_pos)
			wall_o_corner = np.array([entity.endpoints[0],entity.axis_pos+entity.width/2]) - agent_pos
			wall_d_corner = np.array([entity.endpoints[1],entity.axis_pos-entity.width/2]) - agent_pos
			# print(np.array([entity.endpoints[0],entity.axis_pos+entity.width/2]),"wall_o_corner", wall_o_corner,np.array([entity.endpoints[1],entity.axis_pos-entity.width/2]),"wall_d_corner", wall_d_corner)
			return np.hstack([rel_vel, rel_pos, rel_goal_pos,goal_occupied,goal_history,wall_o_corner,wall_d_corner,entity_type])
			# return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type,wall_o_corner])#,wall_d_corner])

		else:
			raise ValueError(f'{entity.name} not supported')
		# print("rel_pos",rel_pos, "rel_vel", rel_vel, "rel_goal_pos", rel_goal_pos, "goal_occupied", goal_occupied)
		return np.hstack([rel_vel, rel_pos, rel_goal_pos,goal_occupied,goal_history,rel_pos,rel_pos,entity_type])
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
			self.num_obstacles:int=0
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
