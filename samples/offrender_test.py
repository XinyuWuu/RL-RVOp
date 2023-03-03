import os
os.environ["DISPLAY"] = ":1"  # for remote access
import mujoco as mj
from mujoco import MjModel
from mujoco import MjData
from mujoco import MjvCamera
from mujoco import MjvOption
from mujoco import MjvScene
from mujoco import MjrContext
import glfw
import subprocess
import numpy as np
import matplotlib.pyplot as plt


model = MjModel.from_xml_path("assets/test.xml")
data = MjData(model)
camera = MjvCamera()
option = MjvOption()
scene = MjvScene()
context = MjrContext()

viewport_width = 1200
viewport_height = 900
framerate = 60
viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
rgb = np.zeros((viewport_width, viewport_height, 3), dtype=np.uint8)
depth = np.zeros((viewport_width, viewport_height, 1), dtype=np.float32)
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)
windows = glfw.create_window(
    viewport_width, viewport_height, "invisible", None, None)
glfw.make_context_current(windows)

mj.mjv_defaultCamera(camera)
mj.mjv_defaultOption(option)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, context)

rgb_raw_file = "assets/rgbbuffer.out"
video_file = "assets/video.mp4"
fp = open(rgb_raw_file, 'bw+')

total_sim_time = 12
flag = False
# TODO set controller callback
data.ctrl[0:2] = np.array([1, 1])*50
# angle = np.deg2rad(90)
# data.joint("car").qpos = np.array(
#     [0.01, 0.01, 0, np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
data.joint("robot_0").qpos
data.joint("robot_0").qvel

camera.fixedcamid = 0
camera.type = mj.mjtCamera.mjCAMERA_FIXED.value

while True:
    time_prev = data.time
    while (data.time - time_prev < 1.0 / 60.0):  # 60fps
        mj.mj_step(model, data)
    # Update scene and render
    mj.mjv_updateScene(model, data, option, None, camera,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    mj.mjr_readPixels(rgb, depth, viewport, context)
    rgb_ = np.flipud(rgb)
    # plt.imshow(rgb_)
    # plt.show()
    rgb_.reshape((-1)).tofile(fp)
    if (data.time > total_sim_time):
        break

glfw.terminate()
fp.close()

subprocess.run(["ffmpeg",
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{int(viewport_height)}x{int(viewport_width)}",
                "-framerate", f"{int(framerate)}",
                "-i", rgb_raw_file,
                video_file,
                "-y",])
# ffmpeg -f rawvideo -pixel_format rgb24 -video_size 600x600 -framerate 60 -i rgbbuffer.out video.mp4
