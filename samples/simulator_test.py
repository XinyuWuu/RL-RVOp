# export LD_LIBRARY_PATH="CppClass/mujoco/lib;CppClass/glfw/src"
import sys
from time import strptime
if sys.path[0] != '':
    sys.path = [''] + sys.path
from CppClass.Simulator import Simulator
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

fp=open("assets/test.xml")
modelfile=fp.read()


env = Simulator(True, 1580, 1580, modelfile)
rgb = env.get_rgb()
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((1580, 1580, 3))

env.step()
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()

for i in range(1000):
    env.step()
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()


env = Simulator(True, 1580, 1580, modelfile)
rgb = env.get_rgb()
img_arr = np.frombuffer(
    rgb, dtype=np.uint8).reshape((1580, 1580, 3))

env.step()
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()

for i in range(1000):
    env.step()
env.render()

img = Image.fromarray(img_arr, "RGB")
img.show()
plt.imshow(img_arr)
plt.show()

env.CloseGLFW()
