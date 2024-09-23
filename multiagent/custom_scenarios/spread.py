"""
    N agents, N landmarks. 
    Agents are rewarded based on how far any agent is from each landmark. 
    Agents are penalized if they collide with other agents. 
    So, agents have to learn to cover all the landmarks while 
    avoiding collisions.
"""
from typing import Optional, Tuple, List
import argparse
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args:argparse.Namespace) -> World:
        # pull params from args
        self.num_agents = args.num_agents
        self.num_obstacles = args.num_obstacles
        self.collaborative = args.collaborative
        self.max_speed = args.max_speed
        self.collision_rew = args.collision_rew
        self.goal_rew = args.goal_rew
        self.min_dist_thresh = args.min_dist_thresh
        self.use_dones = args.use_dones
        self.episode_length = args.episode_length
        self.obs_type = args.obs_type
        self.max_edge_dist = args.max_edge_dist
        self.num_nbd_entities = args.num_nbd_entities
        self.use_comm = args.use_comm
        ####################
        world = World()
        world.world_length = args.episode_length
        world.current_time_step = 0
        # set any world properties first
        world.dim_c = 2
        self.num_agents = args.num_agents
        num_landmarks = self.num_agents
        world.collaborative = args.collaborative
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = f'agent {i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.max_speed = self.max_speed
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = f'landmark {i}'
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world:World) -> None:
        # metrics to keep track of
        world.current_time_step = 0
        # to track time required to reach goal
        world.times_required = -1 * np.ones(self.num_agents)
        # track distance left to the goal
        world.dist_left_to_goal = -1 * np.ones(self.num_agents)
        # number of times agents collide with stuff
        world.num_obstacle_collisions = np.zeros(self.num_agents)
        world.num_agent_collisions = np.zeros(self.num_agents)

        #################### set colours ####################
        # set colours for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.85, 0.15])
        #####################################################

        ####### set random positions for entities ###########
        num_agents_added = 0
        while True:
            if num_agents_added == self.num_agents:
                break
            random_pos = np.random.uniform(-1, +1, world.dim_p)
            agent_size = world.agents[num_agents_added].size
            world.agents[num_agents_added].state.p_pos = random_pos
            world.agents[num_agents_added].state.p_vel = np.zeros(world.dim_p)
            world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
            num_agents_added += 1
        # set landmarks at random positions not colliding with obstacles and 
        # also check collisions with placed goals
        num_goals_added = 0
        goals_added = []
        while True:
            if num_goals_added == self.num_agents:
                break
            random_pos = np.random.uniform(-1, +1, world.dim_p)
            goal_size = world.landmarks[num_goals_added].size
            landmark_collision = self.is_landmark_collision(random_pos, 
                                                goal_size, 
                                                world.landmarks[:num_goals_added])
            if not landmark_collision:
                world.landmarks[num_goals_added].state.p_pos = random_pos
                world.landmarks[num_goals_added].state.p_vel = np.zeros(world.dim_p)
                num_goals_added += 1
        #####################################################

        ############ find minimum times to goals ############
        if self.max_speed is not None:
            for agent in world.agents:
                self.min_time(agent, world)
        #####################################################

        # # set random initial states
        # for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)

    def info_callback(self, agent:Agent, world:World) -> Tuple:
        # TODO modify this 
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        goal = world.get_entity('landmark', agent.id)
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
                                        goal.state.p_pos)))
        world.dist_left_to_goal[agent.id] = dist
        # only update times_required for the first time it reaches the goal
        if dist < self.min_dist_thresh and (world.times_required[agent.id] == -1):
            world.times_required[agent.id] = world.current_time_step * world.dt

        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1

        agent_info = {
            'Dist_to_goal': world.dist_left_to_goal[agent.id],
            'Time_req_to_goal': world.times_required[agent.id],
            # NOTE: total agent collisions is half since we are double counting
            'Num_agent_collisions': world.num_agent_collisions[agent.id], 
        }
        if self.max_speed is not None:
            agent_info['Min_time_to_goal'] = agent.goal_min_time
        return agent_info
    
    # check collision of agent with another agent
    def is_collision(self, agent1:Agent, agent2:Agent) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_landmark_collision(self, pos, size:float, landmark_list:List) -> bool:
        collision = False
        for landmark in landmark_list:
            delta_pos = landmark.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = size + landmark.size
            if dist < dist_min:
                collision = True
                break
        return collision

    # get min time required to reach to goal without obstacles
    def min_time(self, agent:Agent, world:World) -> float:
        assert agent.max_speed is not None, "Agent needs to have a max_speed"
        agent_id = agent.id
        # get the goal associated to this agent
        landmark = world.get_entity(entity_type='landmark', id=agent_id)
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
                                        landmark.state.p_pos)))
        min_time = dist / agent.max_speed
        agent.goal_min_time = min_time
        return min_time

    def benchmark_data(self, agent:Agent, world:World) -> Tuple:
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) 
                    for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1:Agent, agent2:Agent) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent:Agent, world:World) -> float:
        # Agents are rewarded based on minimum agent distance to each landmark, 
        # penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) 
                    for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= self.collision_rew
        return rew

    def observation(self, agent:Agent, world:World) -> np.ndarray:
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + 
                            entity_pos + other_pos + comm)

    def done(self, agent:Agent, world:World) -> bool:
        # return done only when episode_length is reached
        if world.current_time_step >= world.world_length:
            return True
        else:
            return False
    # def done(self, agent:Agent, world:World) -> bool:
    #     # done is False if done_callback is not passed to 
    #     # environment.MultiAgentEnv
    #     # This is same as original version
    #     # Check `_get_done()` in environment.MultiAgentEnv
    #     return False

if __name__ == "__main__":

    from multiagent.environment import MultiAgentOrigEnv
    from multiagent.policy import InteractivePolicy

    # makeshift argparser
    class Args:
        def __init__(self):
            self.num_agents:int=3
            self.num_obstacles:int=3
            self.collaborative:bool=False 
            self.max_speed:Optional[float]=2
            self.collision_rew:float=5
            self.goal_rew:float=5
            self.min_dist_thresh:float=0.1
            self.use_dones:bool=False
            self.episode_length:int=25
            self.share_env = False
            self.obs_type = 'local'
            self.use_comm = False
            self.num_nbd_entities = 3
            self.max_edge_dist = 1
    args = Args()

    scenario = Scenario()

    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentOrigEnv(world=world, reset_callback=scenario.reset_world, 
                        reward_callback=scenario.reward, 
                        observation_callback=scenario.observation, 
                        # info_callback=scenario.info_callback, 
                        done_callback= scenario.done,
                        shared_viewer = False)
    # render call to create viewer window
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    stp=0
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, info_n = env.step(act_n)
        print(obs_n[0].shape)
        # render all agent views
        env.render()
        stp+=1
