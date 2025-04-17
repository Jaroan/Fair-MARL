import numpy as np

class UnicycleVehicleConfig():
    V_MIN = 0.5
    V_MAX = 1.5
    V_NOMINAL = 1.0
    ACCEL_MIN = -1  
    ACCEL_MAX = 1
    ANGULAR_RATE_MAX = 1
    MOTION_PRIM_ACCEL_OPTIONS = 5 ## choices are 3 and 5
    MOTION_PRIM_ANGRATE_OPTIONS = 5 ## total motion primitive choices are MOTION_PRIM_ACCEL_OPTIONS * MOTION_PRIM_ANGRATE_OPTIONS

    # simulation timestep
    DT = 0.1
    # agent within this distance to the landmark is considered to have reached the goal.
    DISTANCE_TO_GOAL_THRESHOLD = 0.3
    # separation distance between agents for safety
    COLLISION_DISTANCE = 0.5
    # communication distance (entities within this distance are considered in each agent's observations)
    COMMUNICATION_RANGE = 5

class DoubleIntegratorConfig():
    VX_MIN = -1.0
    VX_MAX = 1.0
    VY_MIN = -1.0  
    VY_MAX = 1.0
    # Only used for goal point target speed.
    V_MIN = 0.1
    V_NOMINAL = 0.5
    V_MAX = np.sqrt(VX_MAX**2 + VY_MAX**2)
    ACCELX_MIN = -1.0
    ACCELX_MAX = 1.0
    ACCELY_MIN = -1.0
    ACCELY_MAX = 1.0
    ACCELX_OPTIONS = 3 # double check appropriate value
    ACCELY_OPTIONS = 3 # double check appropriate value
    DT = 0.1
    DISTANCE_TO_GOAL_THRESHOLD = 0.05 # m
    # separation distance between agents for safety
    COLLISION_DISTANCE = 0.5
    # communication distance (entities within this distance are considered in each agent's observations)
    COMMUNICATION_RANGE = 5