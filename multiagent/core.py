from typing import List
from enum import Enum

import numpy as np
import csv
from scipy.integrate import solve_ivp

from multiagent.config import DoubleIntegratorConfig, UnicycleVehicleConfig
# function to check for team or single agent scenarios
def is_list_of_lists(lst):
    if isinstance(lst, list) and lst:  # Check if it's a non-empty list
        return all(isinstance(item, list) for item in lst)
    return False

class EntityDynamicsType(Enum):
    DoubleIntegratorXY = 0
    UnicycleVehicleXY = 1
# Base Class for entity state.
# physical/external base state of all entites
class BaseEntityState(object):
    def __init__(self, state_dim):
        self.state_dim = state_dim
        # state values
        self.values = np.zeros(4)
        self.max_speed = None        
        # travel distance
        self.p_dist = 0.0
        # travel time
        self.time = 0.0
        # communication state (only used when entity is agent)

        self.c = None

    @property
    def p_pos(self):
        pass

    @p_pos.setter
    def p_pos(self, val):
        pass
    
    @property
    def p_vel(self):
        pass

    @p_vel.setter
    def p_vel(self, val):
        pass
    
    @property
    def speed(self):
        pass
    
    @staticmethod
    def dstate(state, action):
        pass
    
    def update_state(self, action, dt):
        # state space equation shoul be specified here in child class.
        pass
    
    def stop(self):
        # stop the vehicle (set vehicle speed states to 0)
        pass
    
    def reset_velocity(self, theta=None):
        # reset vehicle velocity. the default value can be zero or random values
        pass 


class UnicycleVehicleXYState(BaseEntityState):
    def __init__(self, v_min, v_max):
        # state_dim = 4
        # p_x, p_y, theta, v
        super(UnicycleVehicleXYState, self).__init__(4)
        self.min_speed = v_min
        self.max_speed = v_max

    @property
    def p_pos(self):
        return self.values[:2]

    @p_pos.setter
    def p_pos(self, val):
        self.values[:2] = val

    @property
    def speed(self):
        return self.values[3]
    
    @speed.setter
    def speed(self, val):
        self.values[3] = val

    @property
    def theta(self):
        return self.values[2]

    @theta.setter
    def theta(self, val):
        self.values[2] = val

    @property
    def p_vel(self):
        return np.array([self.speed * np.cos(self.theta),
                         self.speed * np.sin(self.theta)])
    
    @staticmethod
    def dstate(state, action):
        dp_x = state[3] * np.cos(state[2])
        dp_y = state[3] * np.sin(state[2])
        dtheta = action[0]
        dv = action[1]
        return np.array([dp_x, dp_y, dtheta, dv])

    def update_state(self, action, dt):
        def ode(t, y):
            return self.dstate(y, action)
        y0 = self.values
        sol = solve_ivp(ode, [0, dt], y0, method='RK45')
        self.values = sol.y[:, -1]
        
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < self.min_speed:
            self.speed = self.min_speed
        # update traveled time and distance.
        # TODO integration here is inaccurate. Need to fix this.
        self.p_dist += self.speed * dt
        self.time += dt
    
    def stop(self):
        self.theta = 0
        self.speed = 0
    
    def reset_velocity(self, theta=None):
        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.random.uniform(0, 2 * np.pi)
        self.speed = self.min_speed
    
    def __getitem__(self, idx):
        return self.values[idx]
    
class DoubleIntegratorXYState(BaseEntityState):
    def __init__(self):
        # state_dim = 4
        # p_x, p_y, v_x, v_y
        super(DoubleIntegratorXYState, self).__init__(4)
        # double integrator can stop.
        self.min_speed = 0.0
        self.max_speed = DoubleIntegratorConfig().VX_MAX
        self.min_vel_x = DoubleIntegratorConfig().VX_MIN
        self.max_vel_x = DoubleIntegratorConfig().VX_MAX
        self.min_accel_x = DoubleIntegratorConfig.ACCELX_MIN
        self.max_accel_x = DoubleIntegratorConfig.ACCELX_MAX
        self.min_accel_y = DoubleIntegratorConfig.ACCELY_MIN
        self.max_accel_y = DoubleIntegratorConfig.ACCELY_MAX

    @property
    def p_pos(self):
        return self.values[:2]

    @p_pos.setter
    def p_pos(self, val):
        self.values[:2] = val


    @property
    def speed(self):
        return np.sqrt(self.values[2] ** 2 + self.values[3] ** 2)
    

    @property
    def theta(self):
        return np.arctan2(self.values[3], self.values[2])

    @property
    def p_vel(self):
        return self.values[2:]

    @p_vel.setter
    def p_vel(self, val):
        self.values[2:] = val

    @staticmethod
    def dstate(state, action):
        dp_x = state[2]
        dp_y = state[3]
        dv_x = action[0]
        dv_y = action[1]
        return np.array([dp_x, dp_y, dv_x, dv_y])
        
    def update_state(self, action, dt): # check
        def ode(t, y):
            return self.dstate(y, action)
        y0 = self.values
        sol = solve_ivp(ode, [0, dt], y0, method='RK45')
        self.values = sol.y[:, -1]
        if self.speed > self.max_speed:
            # adjust magnitude to self.max_speed
            self.p_vel = self.max_speed * self.p_vel / self.speed
        # update traveled time and distance.
        self.p_dist += self.speed * dt
        self.time += dt
        
    def stop(self):
        self.p_vel = np.zeros(2)        

    def reset_velocity(self, theta=None):
        """ theta is unused but needed to match the interface of UnicycleVehicleXYState """
        self.p_vel = np.zeros(2)        
    
    def __getitem__(self, idx):
        return self.values[idx]


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties of wall entities
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                hard=True):
        self.id = None
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        self.size = self.width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])
        self.state = DoubleIntegratorXYState()
        # commu channel
        self.channel = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # id
        self.id = None
        self.global_id = None
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = np.array([0.20, 0.20, 0.20])
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = DoubleIntegratorXYState()
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self, dynamics_type: EntityDynamicsType):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        self.min_speed = None
        # state & dynamics
        self.dynamics_type = dynamics_type
        if dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
            self.config_class = DoubleIntegratorConfig
            self.state = DoubleIntegratorXYState()
            self.min_speed = self.state.min_speed
        elif dynamics_type == EntityDynamicsType.UnicycleVehicleXY:
            self.config_class = UnicycleVehicleConfig
            self.state = UnicycleVehicleXYState(v_min=UnicycleVehicleConfig.V_MIN, v_max=UnicycleVehicleConfig.V_MAX)
            self.min_speed = self.state.min_speed
        else:
            raise NotImplementedError("Dynamics type not implemented")
        self.max_speed = self.state.max_speed
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # min time required to get to its allocated goal
        self.goal_min_time = np.inf
        # time passed for each agent
        self.t = 0.0
        self.status = None

# multi-agent world
class World(object):
    def __init__(self, dynamics_type: EntityDynamicsType, 
                separation_distance=None,  total_actions: int = 5):
        assert dynamics_type in EntityDynamicsType, "Invalid dynamics type"
        self.dynamics_type = dynamics_type
        # if we want to construct graphs with the entities 
        self.graph_mode = False
        self.edge_list = None
        self.graph_feat_type = None
        self.edge_weight = None
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.scripted_agents = []
        self.scripted_agents_goals = []
        self.obstacles, self.walls = [], []
        self.wall_obstacles = []
        self.belief_targets = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        if dynamics_type == EntityDynamicsType.DoubleIntegratorXY:
            self.config_class = DoubleIntegratorConfig
        elif dynamics_type == EntityDynamicsType.UnicycleVehicleXY:
            self.config_class = UnicycleVehicleConfig
        else:
            raise NotImplementedError("Dynamics type not implemented")
        self.dt = self.config_class.DT
        self.simulation_time = 0.0
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 3e+2
        self.wall_contact_force = 2.2e+2

        self.contact_margin = 2e-2
        self.wall_contact_margin =2.4e-2
        # self.contact_force = 1.1e+2
        # self.wall_contact_force = 2.9e+2

        # self.contact_margin = 1.7e-3
        # self.wall_contact_margin =2.9e-2
        #         # contact response parameters
        # self.contact_force = 2e+2
        # self.wall_contact_force = 3.0e+2

        # self.contact_margin = 3e-2
        # self.wall_contact_margin = 5e-2
        # cache distances between all agents (not calculated by default)
        self.cache_dists = True
        self.cached_dist_vect = None
        self.cached_dist_mag = None

        self.coordination_range = self.config_class.COMMUNICATION_RANGE
        self.min_dist_thresh = self.config_class.DISTANCE_TO_GOAL_THRESHOLD


        #############
		## determine the number of actions from arguments
        self.total_actions = total_actions # default is 5 actions  [None, ←, →, ↓, ↑]
		#############

    # return all entities in the world
    @property
    def entities(self):
        if is_list_of_lists(self.agents):
            ## flatten the list of lists into a single list
            flattened_agents = [agent for team in self.agents for agent in team]
            return flattened_agents + self.landmarks + self.obstacles + self.wall_obstacles + self.walls
        if not is_list_of_lists(self.agents):
            return self.agents + self.landmarks + self.obstacles + self.wall_obstacles + self.walls

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        if is_list_of_lists(self.agents):
            ## flatten the list of lists into a single list
            flattened_agents = [agent for team in self.agents for agent in team]
            return flattened_agents
        if not is_list_of_lists(self.agents):
            return self.agents
        # return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def get_scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                            len(self.entities),
                                            self.dim_p))
            # calculate minimum distance for a collision between all entities
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    # get the entity given the id and type
    def get_entity(self, entity_type: str, id:int) -> Entity:
        # TODO make this more elegant instead of iterating through everything
        if entity_type == 'agent':
            for agent in self.agents:
                if agent.name == f'agent {id}':
                    return agent
            raise ValueError(f"Agent with id: {id} doesn't exist in the world")
        if entity_type == 'landmark':
            for landmark in self.landmarks:
                if landmark.name == f'landmark {id}':
                    return landmark
            raise ValueError(f"Landmark with id: {id} doesn't exist in the world")
        if entity_type == 'obstacle':
            for obstacle in self.obstacles:
                if obstacle.name == f'obstacle {id}':
                    return obstacle
            raise ValueError(f"Obstacle with id: {id} doesn't exist in the world")

    # update state of the world
    def step(self):

        raw_action_list = self.get_action()
        safe_action_list = raw_action_list

        # integrate physical state
        self.update_agent_state(safe_action_list)

        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.t += self.dt
            agent.action = agent.action_callback(agent, self)
        # # gather forces applied to entities
        # p_force = [None] * len(self.entities)
        # # apply agent physical controls
        # p_force = self.apply_action_force(p_force)
        # # apply environment forces
        # p_force = self.apply_environment_force(p_force)
        # # integrate physical state
        # self.integrate_state(p_force)
        # update agent state
        if is_list_of_lists(self.agents):
            for team in self.agents:
                for agent in team:
                    agent.t += self.dt
                    self.update_agent_communication_state(agent)
        else:
            for agent in self.agents:
                agent.t += self.dt
                self.update_agent_communication_state(agent)
        if self.cache_dists:
            self.calculate_distances()

        self.update_agent_min_relative_distance()
        
        self.simulation_time += self.dt

    # gather agent action forces
    def get_action(self):
        # set applied forces
        ## agent action has an linear acceleration term and an angular acceleration term
        action_list = []
        for i,agent in enumerate(self.agents):
            if agent.u_noise:
                # Jason's temporary fix
                raise NotImplementedError

            action_i = np.array([agent.action.u[0], agent.action.u[1]])
            action_list.append(action_i)
       
        return action_list

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        if is_list_of_lists(self.agents):
            flattened_agents = [agent for team in self.agents for agent in team]
            for i,agent in enumerate(flattened_agents):
                if agent.movable:
                    noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                    p_force[i] = (
                                agent.mass * agent.accel if agent.accel is not None 
                                else agent.mass) * agent.action.u + noise
        else:
            for i,agent in enumerate(self.agents):
                if agent.movable:
                    noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                    p_force[i] = (
                                agent.mass * agent.accel if agent.accel is not None 
                                else agent.mass) * agent.action.u + noise
                # if agent.id == 8:
                #     print("agent force",p_force[i])
                # print("mass",agent.mass,"accel",agent.accel,"action",agent.action.u) 
                # print("force",p_force[i])                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if(b <= a): continue
                # print("a",a,"b",b, entity_a, entity_b)
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                # [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                # if entity_a.id == 8 or entity_b.id == 8:
                    # print("entitya",entity_a.id,"entity_b.id", entity_b.id,"force",f_a,f_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]    
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    # print("wall force",wf)
                    if wf is not None:
                        # print("p_force",p_force[a])
                        if p_force[a] is None: p_force[a] = 0.0
                        csv_data = [wf[0],wf[1],p_force[a][0], p_force[0][0],(p_force[a] + wf)[0],(p_force[a] + wf)[1]]

                        p_force[a] = p_force[a] + wf

                        # with open('/Users/jasmine/Jasmine/MIT/MARL/Codes/Team-Fair-MARL/totalforces.csv', 'a', newline="") as f:
                        #     # create the csv writer
                        #     writer = csv.writer(f)

                        #     # write a row to the csv file
                        #     writer.writerow(csv_data)
                                    # print("p_force",p_force[a])
        return p_force

    # integrate physical state

    # integrate physical state
    def update_agent_state(self, action_list: List):
        # TODO: Change entities to agents
        for i, agent in enumerate(self.agents):
            action_i = action_list[i]
            if not agent.movable: continue
            if agent.status:
                continue
            agent.state.update_state(action_i, self.dt)
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            # print("vel",entity.id,entity.state.p_vel)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                # print("force",entity.id,np.around(p_force[i], 3))
                # print("vel",entity.id,entity.state.p_vel)
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + 
                                np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                        np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt
            # print("vel",entity.state.p_vel)
            entity.state.p_dist += np.linalg.norm(entity.state.p_vel * self.dt)
            entity.state.time += self.dt
    def update_agent_min_relative_distance(self):
        agents_positions = [agent.state.p_pos for agent in self.agents]
        agent_relative_distance_matrix = np.inf * np.ones((len(self.agents), len(self.agents)))
        for i, agent in enumerate(self.agents):
            if agent.status:
                continue
            for j in range(len(self.agents)):
                if i == j:
                    continue
                if not self.agents[j].status:
                    continue
                agent_relative_distance_matrix[i, j] = np.linalg.norm(agents_positions[i] - agents_positions[j])
        for i, agent in enumerate(self.agents):
            agent.min_relative_distance = np.min(agent_relative_distance_matrix[i, :])
    def update_agent_communication_state(self, agent:Agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * \
                    agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    # NOTE: this is better than using get_collision_force() since 
    # it takes into account if the entity is movable or not
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]  # neither entity moves
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        if (self.cache_dists) and (self.cached_dist_vect is not None):
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = 1.1*entity_a.size + 1.1*entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        # print("force",force)
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force if entity_a.status != True else None
            force_b = -(1 / force_ratio) * force if entity_b.status != True else None
            # print("COLLISION force agent.a",entity_a.id, force_a,"agent.b",entity_b.id,force_b)
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        # print("Entity collision forces: ", force_a, force_b)
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
                ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
                ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.wall_contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force_mag = self.wall_contact_force * delta_pos / dist * penetration
        # force_mag = self.wall_contact_force * delta_pos / dist

        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        # if dist < dist_min:
        #     print("dist",dist, dist_min, dist - dist_min)

        #     print("penetration",penetration)

        #     print("force_mag",force_mag)

        #     print("force",force)
        # csv_data = [dist,dist_min,dist - dist_min,penetration,force_mag,force]

        # with open('/Users/jasmine/Jasmine/MIT/MARL/Codes/Team-Fair-MARL/forces.csv', 'a', newline="") as f:
        #     # create the csv writer
        #     writer = csv.writer(f)

        #     # write a row to the csv file
        #     writer.writerow(csv_data)
        return force

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        # print("Collision forces: ", force_a, force_b)
        return [force_a, force_b]
    
    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # r g b
        dummy_colors = [(0.25, 0.75, 0.25)] * n_dummies
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])