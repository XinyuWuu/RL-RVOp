import numpy as np
from numpy import rad2deg, pi, arctan2, array
from PIL import Image, ImageDraw, ImageColor, ImageFont


class Canvas():
    def __init__(self, w=32, h=16, dpi=100, bg_color="lightgrey", raw_bg_color=False) -> None:
        self.w = w
        self.h = h
        self.dpi = dpi
        self.bg_color = bg_color
        self.raw_bg_color = raw_bg_color
        self.newCanvas()

    def newCanvas(self):
        if self.raw_bg_color:
            self.img = Image.new("RGB", (self.w * self.dpi, self.h * self.dpi),
                                 color=self.bg_color)
        else:
            self.img = Image.new("RGB", (self.w * self.dpi, self.h * self.dpi),
                                 color=ImageColor.getcolor(self.bg_color, "RGB"))
        self.imgDraw = ImageDraw.Draw(self.img)

    def point_convert(self, point: np.ndarray):
        point = point * self.dpi * \
            array([1, -1]) + array([self.w / 2 * self.dpi, self.h / 2 * self.dpi])
        return (int(round(point[0])), int(round(point[1])))

    def box(self, point1, point2):
        return (min(point1[0], point2[0]), min(point1[1], point2[1]), max(point1[0], point2[0]), max(point1[1], point2[1]))

    def draw_line(self, point_1, point_2, color='red', linewidth=4):
        self.imgDraw.line((self.point_convert(point_1), self.point_convert(
            point_2)), fill=ImageColor.getcolor(color, "RGB"), width=linewidth)

    def draw_arc(self, c, r, v1, v2, vm, color="red", linewidth=4):
        theta1 = arctan2((v1 - c)[1], (v1 - c)[0])
        theta2 = arctan2((v2 - c)[1], (v2 - c)[0])
        thetam = arctan2((vm - c)[1], (vm - c)[0])

        theta1 = theta1 if abs(theta1 - thetam) < abs(theta1 +
                                                      2 * pi - thetam) else theta1 + 2 * pi
        theta1 = theta1 if abs(theta1 - thetam) < abs(theta1 -
                                                      2 * pi - thetam) else theta1 - 2 * pi
        theta2 = theta2 if abs(theta2 - thetam) < abs(theta2 +
                                                      2 * pi - thetam) else theta2 + 2 * pi
        theta2 = theta2 if abs(theta2 - thetam) < abs(theta2 -
                                                      2 * pi - thetam) else theta2 - 2 * pi
        self.imgDraw.arc(self.box(self.point_convert(c - r), self.point_convert(c + r)), -rad2deg(max(theta1, theta2)),
                         -rad2deg(min(theta1, theta2)), fill=ImageColor.getcolor(color, "RGB"), width=linewidth)

    def draw_contour(self, contours: list, color="red", linewidth=2, draw_arc_cross=False):
        for con in contours:
            for line in con[0]:
                if len(line) == 0:
                    continue
                self.draw_line(line[2:4], line[4:6],
                               color=color, linewidth=linewidth)
            for arc in con[1]:
                self.draw_arc(arc[0:2], arc[2], arc[3:5], arc[5:7],
                              arc[7:9], color=color, linewidth=linewidth)
                if not draw_arc_cross:
                    continue
                self.draw_line(arc[3:5], arc[5:7],
                               color=color, linewidth=linewidth)
                self.draw_line(arc[0:2], arc[5:7],
                               color=color, linewidth=linewidth)
                self.draw_line(arc[0:2], arc[3:5],
                               color=color, linewidth=linewidth)
                self.draw_line(arc[0:2], arc[7:9],
                               color=color, linewidth=linewidth)

    def draw_rvop(self, rvop: np.ndarray, xr: np.ndarray, color_v1_v2="blue", color_vnear="orange", linewidth=2, draw_points=False, points_color='green', points=[]):
        self.draw_line(xr, xr + rvop[2:4], color_v1_v2, linewidth)
        self.draw_line(xr, xr + rvop[4:6], color_v1_v2, linewidth)
        self.draw_line(xr, xr + rvop[6:8], color_vnear, linewidth)
        if draw_points:
            for p in points:
                self.draw_line(p, xr, points_color, linewidth)

    def draw_dmax(self, xr: np.ndarray, dmax, color='green', linewidth=2):
        self.imgDraw.arc(self.box(self.point_convert(xr - dmax), self.point_convert(xr + dmax)), 0,
                         360, fill=ImageColor.getcolor(color, "RGB"), width=linewidth)

    def draw_text(self, point, text, color="red", font=ImageFont.truetype(
                    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",25), fontsize = 10):
        self.imgDraw.text(self.point_convert(point), text,
                          fill = ImageColor.getcolor(color, "RGB"), font = font, fontsize = fontsize, align = 'center')
