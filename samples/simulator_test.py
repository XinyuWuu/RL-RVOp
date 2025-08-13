# export LD_LIBRARY_PATH="CppClass/mujoco/lib;CppClass/glfw/src"
import sys
if sys.path[0] != '':
    sys.path = [''] + sys.path
from CppClass.Simulator import Simulator
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

fp = open("assets/test.xml")
modelfile = fp.read()
Nrobot = 30

ctrl = [100.0, 100.0]*Nrobot

env = Simulator(modelfile, Nrobot, True, 1580, 1580)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((1580, 1580, 3))

env.step(ctrl, 1)
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()
print(posvels)

env.step(ctrl, 1000)
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()
print(posvels)

env = Simulator(modelfile, Nrobot, True, 1580, 1580)
rgb = env.get_rgb()
posvels = np.frombuffer(
    env.get_posvels(), dtype=np.float64).reshape((Nrobot, 6))
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((1580, 1580, 3))

env.step(ctrl, 1)
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()
print(posvels)

env.step(ctrl, 1000)
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()
print(posvels)

env.CloseGLFW()
