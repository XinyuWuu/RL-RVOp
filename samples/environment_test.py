# export LD_LIBRARY_PATH="CppClass/mujoco/lib;CppClass/glfw/src"
import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
from CppClass.Environment import Environment
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

fp = open("assets/test.xml")
modelfile = fp.read()
Nrobot = 30

ctrl = [[1.0, 0]]*Nrobot

env = Environment()
env.setSim(modelfile, Nrobot, True, 1580, 1580)
env.setCtrl(1, 0.5, 0.04, 0.28, 7)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((1580, 1580, 3))

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

env.setSim(modelfile, Nrobot, True, 1580, 1580)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((1580, 1580, 3))

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
