import render
import importlib
import numpy as np
from mujoco import MjData
from mujoco import MjModel
import mujoco as mj
import os
os.environ["DISPLAY"] = ":1"  # for remote access
importlib.reload(render)

model = MjModel.from_xml_path("assets/test.xml")
data = MjData(model)

viewport_width = 1920
viewport_height = 1080
framerate = 50
Render = render.Render(model, data, framerate,
                       "assets/video.mp4", "h264", "yuv420p", viewport_width, viewport_height)
Render.switchCam()

step_time = 1 / framerate
step_num = int(round(step_time / 0.002))
step_time = step_num * 0.002
total_sim_time = 2
total_step = int(round(total_sim_time / step_time))

# TODO set controller callback
data.ctrl[0:2] = np.array([-1, -1]) * 50.0
data.ctrl[2:4] = np.array([1, 1]) * 50.0
data.ctrl[4:6] = np.array([-1, -1]) * 50.0
data.ctrl[6:8] = np.array([-1, -1]) * 50.0
data.ctrl[8:10] = np.array([1, 1]) * 50.0
# angle = np.deg2rad(90)
# data.joint("car").qpos = np.array(
#     [0.01, 0.01, 0, np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
# data.joint("robot_0").qpos[0:2] = np.array([4, 0])
# data.joint("robot_0").qvel


for stepn in range(total_step):
    time_prev = data.time
    mj.mj_step(model, data, step_num)
    Render.render()

Render.close()
