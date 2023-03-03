import os
import mujoco as mj
from mujoco import MjModel
from mujoco import MjData
from mujoco import MjvCamera
from mujoco import MjvOption
from mujoco import MjvScene
from mujoco import MjrContext
import glfw
import numpy as np
import matplotlib.pyplot as plt

model = MjModel.from_xml_path("assets/test.xml")
data = MjData(model)
camera = MjvCamera()
option = MjvOption()
scene = MjvScene()
context = MjrContext()

glfw.init()
window = glfw.create_window(600, 600, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(camera)
mj.mjv_defaultOption(option)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
total_sim_time = 6
frames = []
# TODO set controller callback
omegal = 5.882352941
omegar = 5.882352941
#1.0081678033941741
# angle = np.deg2rad(90)
# data.joint("robot_0").qpos = np.array(
#     [0.01, 0.01, 0, np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

flag = False
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0 / 60.0):  # 60fps
        mj.mj_step(model, data)


    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    camera.fixedcamid = 0
    camera.type = mj.mjtCamera.mjCAMERA_FIXED.value
    mj.mjv_updateScene(model, data, option, None, camera,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

    if (data.time > total_sim_time):
        break

glfw.terminate()
