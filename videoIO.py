import av
import numpy as np
from time import strftime, localtime


class VideoIO():
    def __init__(self, vf_name="", framerate=50, codec="hevc", w=1920, h=1080, pix_fmt="yuv420p", vf_end="glfw") -> None:
        if vf_name == "":
            self.vf_name = strftime(
                f"./assets/%H:%M:%S_%d_%m_{vf_end}.mp4", localtime())
            pass
        else:
            self.vf_name = vf_name
        self.container = av.open(self.vf_name, mode="w")
        self.stream = self.container.add_stream(codec, rate=framerate)
        self.stream.width = w
        self.stream.height = h
        self.stream.pix_fmt = pix_fmt

    def write_frame(self, img: np.ndarray, format="rgb24"):
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(img, format=format)):
            self.container.mux(packet)

    def close(self):
        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
