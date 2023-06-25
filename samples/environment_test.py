# export LD_LIBRARY_PATH="CppClass/mujoco/lib;CppClass/glfw/src"
import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
from CppClass.Environment import Environment
import envCreator
import contourGenerator
import numpy as np
from numpy import zeros
from PIL import Image
from matplotlib import pyplot as plt

robot_r = 0.2
EC = envCreator.EnvCreator()
CG = contourGenerator.ContourGenrator(robot_r)

Nrobot, robot_text, obs_text, obs, target_mode, ow, oh, ch, fovy, w, h = EC.env_create3(
    0, 0)
actuator_text = EC.actuator(Nrobot)
modelfile = EC.env_text(robot_text, obs_text,
                        actuator_text, ow, ow, ch, fovy)
contour = CG.generate_contour(obs)
ctrl = [[1.0, 0]]*Nrobot

env = Environment()
env.setSim(modelfile, Nrobot, [[0, 0]]*Nrobot, contour, True, ow, oh)
env.setCtrl(1, 0.5, 0.04, 0.28, 7)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((ow, oh, 3))

env.stepVL(ctrl, 1, 1)
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()
print(posvels)
posvels1 = posvels.copy()

env.stepVL(ctrl, 3000, 5)
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()
print(posvels)
posvels2 = posvels.copy()
print(posvels2-posvels1)

env.setSim(modelfile, Nrobot, [[0, 0]]*Nrobot, contour, True, ow, oh)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((ow, oh, 3))

env.stepVL(ctrl, 1, 1)
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()
print(posvels)
posvels1 = posvels.copy()

env.stepVL(ctrl, 3000, 5)
env.render()


img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()
print(posvels)
posvels2 = posvels.copy()
print(posvels2-posvels1)

env.CloseGLFW()
