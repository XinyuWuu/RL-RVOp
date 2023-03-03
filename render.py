
import matplotlib.pyplot as plt
import numpy as np
import glfw
from mujoco import MjrContext
from mujoco import MjvScene
from mujoco import MjvOption
from mujoco import MjvCamera
from mujoco import MjData
from mujoco import MjModel
import mujoco as mj
import os
import importlib
import videoIO
importlib.reload(videoIO)


class Render():
    def __init__(self, viewport_width=1920, viewport_height=1080) -> None:
        self.camera = MjvCamera()
        self.option = MjvOption()
        self.scene = MjvScene()
        self.context = MjrContext()
        self.viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        self.rgb = np.zeros(
            (viewport_height, viewport_width, 3), dtype=np.uint8)
        self.depth = np.zeros(
            (viewport_height, viewport_width, 1), dtype=np.float32)

        os.environ["DISPLAY"] = ":1"  # for remote access
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)
        self.windows = glfw.create_window(
            viewport_width, viewport_height, "invisible", None, None)
        glfw.make_context_current(self.windows)

        mj.mjv_defaultCamera(self.camera)
        mj.mjv_defaultOption(self.option)

    def set_model(self, model: MjModel, data: MjData):
        self.mjMODEL = model
        self.mjDATA = data
        self.scene = mj.MjvScene(self.mjMODEL, maxgeom=10000)
        self.context = mj.MjrContext(
            self.mjMODEL, mj.mjtFontScale.mjFONTSCALE_150.value)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, self.context)

    def render(self):
        mj.mjv_updateScene(self.mjMODEL, self.mjDATA, self.option, None, self.camera,
                           mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(self.viewport, self.scene, self.context)
        mj.mjr_readPixels(self.rgb, self.depth, self.viewport, self.context)

    def switchCam(self, id=0, type="fixed"):
        if type == "fixed":
            self.camera.fixedcamid = id
            self.camera.type = mj.mjtCamera.mjCAMERA_FIXED.value

    def close(self):
        glfw.destroy_window(self.windows)
        glfw.terminate()
