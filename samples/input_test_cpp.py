import importlib
import random
from numpy.linalg import norm
from numpy import array, cross, dot, deg2rad, rad2deg, pi, arctan2, sin, cos
import canvas
import contourGenerator
import envCreator
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
from CppClass.RVOcalculator import RVOcalculator
from numpy import array, sqrt
from numpy.linalg import norm
import numpy as np

importlib.reload(envCreator)
importlib.reload(contourGenerator)
importlib.reload(canvas)
# importlib.reload(RVOcalculator)

# seed = 103
# np.random.seed(seed)
# random.seed(seed)

robot_r = 0.17
dmax = 3.0
ec = envCreator.EnvCreator()
cg = contourGenerator.ContourGenrator(robot_r)


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
# for i in range(contours.__len__()):
#     contours[i]=(contours[i][0].tolist(),contours[i][1].tolist())
# print(contours[0])
# print(type(contours[0][0]))
rvo = RVOcalculator(dmax, robot_r, contours, target=np.zeros((n, 2)))


# fig, ax = canvas.create_canvas(1)
CV = canvas.Canvas(w=32, h=16)
# canvas.draw_contour(contours, ax)
CV.draw_contour(contours)
# print(contours)
xr = array([3.0, 2.0])
points = []
for con in contours:
    # print("python veiw:")
    # print(con[0])
    # print(con[1])
    # print(xr)
    rvop = rvo.RVOplus(*con, xr=xr,
                        vr=array([0, 0]), vo=array([0, 0]))

    if rvop[-1] > dmax:
        continue
    CV.draw_rvop(array(rvop), xr)
    # for line in con[0]:
    #     points += rvo.LineP(line, xr)
    # for arc in con[1]:
    #     points += rvo.ArcP(arc, xr)



for point in points:
    # print(point)
    CV.draw_line(array(point), xr)
# canvas.draw_dmax(xr, ax, dmax)
CV.draw_dmax(xr, dmax)
# fig.savefig("assets/draw.png")
# fig.clear()
# CV.img.show()
CV.img.save("./assets/draw.png")
