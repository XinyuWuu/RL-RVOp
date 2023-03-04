# https://github.com/MengGuo/RVO_Py_MAS

import RVO_Py_MAS.RVO as RVO
from RVO_Py_MAS.RVO import RVO_update, compute_V_des


# ------------------------------
# define workspace model
ws_model = dict()
# robot radius
ws_model['robot_radius'] = 0.2

# circular obstacles, format [x,y,rad]
# rectangular boundary, format [x,y,width/2,heigth/2]

# no obstacles
# ws_model['circular_obstacles'] = []
# ws_model['boundary'] = []

# with obstacles
ws_model['circular_obstacles'] = [[-0.3, 2.5, 0.3],
                                  [1.5, 2.5, 0.3], [3.3, 2.5, 0.3], [5.1, 2.5, 0.3]]
ws_model['boundary'] = []

# ------------------------------
# initialization for robot
# position of [x,y]
X = [[-0.5 + 1.0 * i, 0.0]
     for i in range(7)] + [[-0.5 + 1.0 * i, 5.0] for i in range(7)]
# velocity of [vx,vy]
V = [[0, 0] for i in range(len(X))]
# maximal velocity norm
V_max = [1.0 for i in range(len(X))]
# goal of [x,y]
goal = [[5.5 - 1.0 * i, 5.0]
        for i in range(7)] + [[5.5 - 1.0 * i, 0.0] for i in range(7)]

# ------------------------------
# simulation setup
# total simulation time (s)
total_time = 15
# simulation step
step = 0.01

# ------------------------------
# simulation starts
t = 0
while t * step < total_time:
    # compute desired vel to goal
    V_des = compute_V_des(X, goal, V_max)
    # compute the optimal vel to avoid collision
    # VO or RVO or HRVO
    V = RVO_update(X, V_des, V, ws_model, 'VO')
    # update position
    for i in range(len(X)):
        X[i][0] += V[i][0] * step
        X[i][1] += V[i][1] * step
    t += 1
