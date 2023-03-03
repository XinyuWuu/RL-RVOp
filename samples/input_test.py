from matplotlib.lines import Line2D
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import envCreator
import contourGenerator
import RVOcalculator
import canvas
import numpy as np
from numpy import array, cross, dot, deg2rad, rad2deg, pi, arctan2, sin, cos
from numpy.linalg import norm
import random

import importlib
importlib.reload(envCreator)
importlib.reload(contourGenerator)
importlib.reload(RVOcalculator)
importlib.reload(canvas)

# seed = 103
# np.random.seed(seed)
# random.seed(seed)

robot_r = 0.17
dmax = 3.0
ec = envCreator.EnvCreator()
cg = contourGenerator.ContourGenrator(robot_r)
rvo = RVOcalculator.RVO(dmax)


n = 11
robot_text = ec.gate_robot(n)
# obs_text, obs = ec.line_obstacle(10)
obs_text, obs = ec.circle_obstacle(10)
# obs_text, obs = ec.gate_obstacle(True)
# obs_text, obs = ec.bigObs_obstacle(False, 1)
actuator_text = ec.actuator(n)
text = ec.env_text(robot_text, obs_text, actuator_text)
with open("assets/test.xml", 'w') as fp:
    fp.write(text)

contours = cg.generate_contour(obs)


# fig, ax = canvas.create_canvas(1)
CV=canvas.Canvas(w=32,h=16)
# canvas.draw_contour(contours, ax)
CV.draw_contour(contours)

xr = array([3.0, 2.0])
for con in contours:
    rvop, points = rvo.RVOplus(*con, xr=xr,
                               vr=array([0, 0]), vo=array([0, 0]))

    # points = []
    # for line in contours[2][0]:
    #     points += rvo.LineP(line, xr)
    # for arc in contours[2][1]:
    #     points += rvo.ArcP(arc, xr)

    # canvas.draw_rvop(rvop, xr, ax)
    CV.draw_rvop(rvop,xr)

# canvas.draw_dmax(xr, ax, dmax)
CV.draw_dmax(xr,dmax)
# fig.savefig("assets/draw.png")
# fig.clear()
CV.img.show()
