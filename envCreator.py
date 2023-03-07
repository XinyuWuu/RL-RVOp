import re
import functools
import numpy as np
from numpy.linalg import norm, det
from numpy import dot, array
import random

# obs
# line: two vertex (4,)
# cylinder: center and radius (3,)
# polygon: center and vertex (8,) or (10,)


class EnvCreator():
    def __init__(self) -> None:
        self.half_width = 0.000005
        self.half_height = 0.25
        self.kv = 0.05
        self.circle_split_num = 16
        self.obstacle_rgba = "0.5 0 0 1"
        self.obs_r = 0.25  # radius of circumcircle of small obstacles, half for sylinder
        self.gate_ratio = 3 / 5
        with open("assets/robot.xml", 'r') as fp:
            self.robot_base = fp.read()

    def add_id(self, match_obj: re.Match, id: int):
        return match_obj.group(0)[0:-1] + f'_{id}\"'

    def robot(self, id: int = 0, c: np.ndarray = array([0, 0]), theta: float = 0):
        text = self.robot_base
        text = re.sub(r'name=\".*?\"',
                      functools.partial(self.add_id, id=id), text)
        text = re.sub(r'pos=\".*?\"', f'pos="{c[0]} {c[1]} 0"', text, 1)
        text = re.sub(r'euler=\".*?\"',
                      f'euler="0 0 {np.rad2deg(theta)}"', text, 1)
        return '\n' + text + '\n'

    def actuator(self, n: int = 0):
        text = '<actuator>\n'
        for id in range(n):
            text += f'\t<velocity name="left-velocity-servo_{id}" joint="left-wheel_{id}" kv="{self.kv}" />\n'
            text += f'\t<velocity name="right-velocity-servo_{id}" joint="right-wheel_{id}" kv="{self.kv}" />\n'
        text += '</actuator>'
        return '\n' + text + '\n'

    def polygon(self, c: np.ndarray = array([0, 0]),
                x: list = [array([np.cos(0), np.sin(0)]),
                           array([np.cos(np.pi / 3), np.sin(np.pi / 3)]),
                           array([np.cos(np.pi), np.sin(np.pi)])]):

        text = f'<body pos="{c[0]} {c[1]} {self.half_height}">\n'
        xuint = array([1, 0])
        for i in range(len(x)):
            a = x[i]
            b = x[0 if i == len(x) - 1 else i + 1]
            length = norm(b - a)
            angle = - \
                np.arctan2(det([b - a, xuint]), dot(b - a, xuint))
            pos = (a + b) / 2 - (a + b) / norm(a + b) * self.half_width
            text += f'\t<geom rgba="{self.obstacle_rgba}" type="box" pos="{pos[0]} {pos[1]} 0" size="{length/2} {self.half_width} {self.half_height}" euler="0 0 {np.rad2deg(angle)}" />\n'
        text += "</body>"
        return '\n' + text + '\n'

    def cylinder(self, c: np.ndarray = array([0, 0]), r: float = 1):
        text = f'<body pos="{c[0]} {c[1]} {self.half_height}">\n'
        text += f'\t<geom rgba="{self.obstacle_rgba}" type="cylinder" size="{r} {self.half_height}" />\n'
        text += "</body>"
        return '\n' + text + '\n'

    def circle_robot(self, n: int = 1, size: str = 'l', start_id=0):
        # max 60 robots for size large, 36 for size small
        r_l = 5
        r_s = 3
        n_l = 60
        theta_l = np.deg2rad(360 / n_l)
        n_s = 36
        theta_s = np.deg2rad(360 / n_s)

        if (size == 'l'):
            theta = array(random.sample(range(0, n_l), n)) * \
                theta_l + np.random.rand() * 2 * np.pi
            center = array([[np.cos(e), np.sin(e)] for e in theta]) * r_l
        else:
            theta = array(random.sample(range(0, n_s), n)) * \
                theta_s + np.random.rand() * 2 * np.pi
            center = array([[np.cos(e), np.sin(e)] for e in theta]) * r_s

        id = start_id
        text = ""
        for c in center:
            text += self.robot(id, c, np.random.rand() * 2 * np.pi)
            id += 1
        return text

    def obstacle(self, center: np.ndarray = array([[1, 1], [-1, -1]]), t: bool = False):
        text = ""
        obs = []
        for c in center:
            if np.random.rand() < 0.5:
                text += self.cylinder(c, self.obs_r * 0.5)
                obs.append(array([c[0], c[1], self.obs_r * 0.5]))
            else:
                if t and np.random.rand() < 0.5:
                    es = array(sorted(random.sample(
                        range(0, self.circle_split_num), 4))) / self.circle_split_num * 2 * np.pi
                    vs = array([[np.cos(e), np.sin(e)]
                                for e in es]) * self.obs_r
                else:
                    es = array(sorted(random.sample(
                        range(0, self.circle_split_num), 3))) / self.circle_split_num * 2 * np.pi
                    vs = array([[np.cos(e), np.sin(e)]
                                for e in es]) * self.obs_r
                text += self.polygon(c, vs)
                obs.append(np.hstack((c, (vs + c).reshape((-1,)))))
        return text, obs

    def circle_obstacle(self, n: int = 1, size: str = 'l', t: bool = False):
        # max 60 obstacles for size large, 36 for size small
        r_l = 5
        r_s = 3
        n_l = 60
        theta_l = np.deg2rad(360 / n_l)
        n_s = 36
        theta_s = np.deg2rad(360 / n_s)
        if (size == 'l'):
            theta = array(random.sample(
                range(0, int(n_l / 4)), n)) * theta_l * 4 + np.random.rand() * 2 * np.pi
            center = array([[np.cos(e), np.sin(e)] for e in theta]) * r_l / 2
        else:
            theta = array(random.sample(
                range(0, int(n_s / 6)), n)) * theta_s * 6 + np.random.rand() * 2 * np.pi
            center = array([[np.cos(e), np.sin(e)] for e in theta]) * r_s / 3

        return self.obstacle(center, t)

    def line_robot(self, n: int = 1, size: str = 'l', start_id=0):
        # max 20 robots for size large, 10 for size small
        len_l = 10
        len_s = 5
        n_l = 20
        n_s = 10
        d = 0.5
        if size == 'l':
            center1 = array([[-len_l / 2, (x + np.random.rand() / 8) * d]
                            for x in random.sample(range(-int(n_l / 2), int(n_l / 2)), int(np.floor(n / 2)))])
            center2 = array([[len_l / 2, (x + np.random.rand() / 8) * d]
                            for x in random.sample(range(-int(n_l / 2), int(n_l / 2)), int(np.ceil(n / 2)))])
            center = np.vstack((center1, center2))
        else:
            center1 = array([[-len_s / 2, (x + np.random.rand() / 8) * d]
                            for x in random.sample(range(-int(n_s / 2), int(n_s / 2)), int(np.floor(n / 2)))])
            center2 = array([[len_s / 2, (x + np.random.rand() / 8) * d]
                            for x in random.sample(range(-int(n_s / 2), int(n_s / 2)), int(np.ceil(n / 2)))])
            center = np.vstack((center1, center2))

        id = start_id
        text = ""
        for c in center:
            text += self.robot(id, c, np.random.rand() * 2 * np.pi)
            id += 1
        return text

    def line_obstacle(self, n: int = 1, size: str = 'l', t: bool = False):
        # max 20 obstacles for size large, 10 for size small
        len_l = 15
        len_s = 7.5
        n_l = 20
        n_s = 10
        d = 0.75
        if size == 'l':
            center1 = array([[-len_l / 4, (x + np.random.rand() / 8) * d]
                            for x in random.sample(range(-int(n_l / 2), int(n_l / 2)), int(np.floor(n / 2)))])
            center2 = array([[len_l / 4, (x + np.random.rand() / 8) * d]
                            for x in random.sample(range(-int(n_l / 2), int(n_l / 2)), int(np.ceil(n / 2)))])
            center = np.vstack((center1, center2))
        else:
            center1 = array([[-len_s / 6, (x + np.random.rand() / 8) * d]
                            for x in random.sample(range(-int(n_s / 2), int(n_s / 2)), int(np.floor(n / 2)))])
            center2 = array([[len_s / 6, (x + np.random.rand() / 8) * d]
                            for x in random.sample(range(-int(n_s / 2), int(n_s / 2)), int(np.ceil(n / 2)))])
            center = np.vstack((center1, center2))

        return self.obstacle(center, t)

    def gate_robot(self, n: int = 1, start_id=0):
        # max 15 robots
        n_max = 15
        d = 0.5
        length = 15
        center1 = array([[-length / 2, (x + np.random.rand() / 8) * d]
                        for x in random.sample(range(-int(n_max / 2), int(n_max / 2)), int(np.floor(n / 2)))])
        center2 = array([[length / 2, (x + np.random.rand() / 8) * d]
                        for x in random.sample(range(-int(n_max / 2), int(n_max / 2)), int(np.ceil(n / 2)))])
        center = np.vstack((center1, center2))
        text = ""
        id = start_id
        for c in center:
            text += self.robot(id, c, np.random.rand() * 2 * np.pi)
            id += 1
        return text

    def gate_obstacle(self, t: bool = False):
        width = 8
        length = 40
        half_r_len = 7.5
        obs = []
        text = f'<body  pos="0 0 {self.half_height}">\n'
        text += f'\t<geom rgba="{self.obstacle_rgba}" type="box" pos="0 {-width/2} 0" size="{length/2} {self.half_width} {self.half_height}" />\n'
        obs.append(array([length / 2, -width / 2, -length / 2, -width / 2]))
        text += f'\t<geom rgba="{self.obstacle_rgba}" type="box" pos="0 {width/2} 0" size="{length/2} {self.half_width} {self.half_height}" />\n'
        obs.append(array([length / 2, width / 2, -length / 2, width / 2]))

        if not t:
            if np.random.rand() < 0.5:
                text += f'\t<geom rgba="{self.obstacle_rgba}" type="box" pos="0 {width/2*(1-self.gate_ratio)} 0" size="{self.half_width} {width*self.gate_ratio/2}  {self.half_height}" />\n'
                obs.append(
                    array([0, width / 2 * (1 - self.gate_ratio) + width * self.gate_ratio / 2, 0, width / 2 * (1 - self.gate_ratio) - width * self.gate_ratio / 2]))
            else:
                text += f'\t<geom rgba="{self.obstacle_rgba}" type="box" pos="0 -{width/2*(1-self.gate_ratio)} 0" size="{self.half_width} {width*self.gate_ratio/2}  {self.half_height}" />\n'
                obs.append(
                    array([0, -width / 2 * (1 - self.gate_ratio) + width * self.gate_ratio / 2, 0, -width / 2 * (1 - self.gate_ratio) - width * self.gate_ratio / 2]))
        else:
            offset = 0.3 if np.random.rand() < 0.5 else -0.3

            text += f'\t<geom rgba="{self.obstacle_rgba}" type="box" pos="{offset*half_r_len} {width/2*(1-self.gate_ratio)} 0" size="{self.half_width} {width*self.gate_ratio/2}  {self.half_height}" />\n'
            obs.append(
                array([offset * half_r_len, width / 2 * (1 - self.gate_ratio) + width * self.gate_ratio / 2, offset * half_r_len, width / 2 * (1 - self.gate_ratio) - width * self.gate_ratio / 2]))
            text += f'\t<geom rgba="{self.obstacle_rgba}" type="box" pos="{-offset*half_r_len} -{width/2*(1-self.gate_ratio)} 0" size="{self.half_width} {width*self.gate_ratio/2}  {self.half_height}" />\n'
            obs.append(
                array([-offset * half_r_len, -width / 2 * (1 - self.gate_ratio) + width * self.gate_ratio / 2, -offset * half_r_len, -width / 2 * (1 - self.gate_ratio) - width * self.gate_ratio / 2]))
        text += "</body>"

        return text, obs

    def bigObs_obstacle(self, t: bool = False, mode=0):
        width = 8
        length = 40
        half_r_len = 7.5
        obs = []
        text = f'<body  pos="0 0 {self.half_height}">\n'
        text += f'\t<geom rgba="{self.obstacle_rgba}" type="box" pos="0 {-width/2} 0" size="{length/2} {self.half_width} {self.half_height}" />\n'
        obs.append(array([length / 2, -width / 2, -length / 2, -width / 2]))
        text += f'\t<geom rgba="{self.obstacle_rgba}" type="box" pos="0 {width/2} 0" size="{length/2} {self.half_width} {self.half_height}" />\n'
        obs.append(array([length / 2, width / 2, -length / 2, width / 2]))
        text += "</body>"

        obs_r = self.gate_ratio * width * 2 / 3
        yoffset = width / 2 - self.gate_ratio * width / 3
        xoffset = half_r_len / 3 if np.random.rand() < 0.5 else -half_r_len / 3
        center1 = array([xoffset, yoffset])
        center2 = array([-xoffset, -yoffset])

        if not t:
            if np.random.rand() < 0.5:
                es = array(
                    sorted([np.deg2rad(-90), np.deg2rad(30), np.deg2rad(150)]))
                vs = array([[np.cos(e), np.sin(e)]
                            for e in es]) * obs_r
                text += self.polygon(center1, vs)
                obs.append(np.hstack((center1, (vs + center1).reshape((-1,)))))
                es = array(
                    sorted([np.deg2rad(90), np.deg2rad(-30), np.deg2rad(-150)]))
                vs = array([[np.cos(e), np.sin(e)]
                            for e in es]) * obs_r
                text += self.polygon(center2, vs)
                obs.append(np.hstack((center2, (vs + center2).reshape((-1,)))))
            else:
                es = array(
                    sorted(random.sample(range(0, self.circle_split_num), 3))) / self.circle_split_num * 2 * np.pi
                vs = array([[np.cos(e), np.sin(e)]
                            for e in es]) * obs_r
                text += self.polygon(np.zeros((2,)), vs)
                obs.append(
                    np.hstack((np.zeros((2,)), (vs + np.zeros((2,))).reshape((-1,)))))
        else:
            if mode == 0:
                text += self.cylinder(np.zeros((2,)), obs_r * 0.75)
                obs.append(array([0, 0, obs_r * 0.75]))
            else:
                es = array(sorted(random.sample(
                    range(0, self.circle_split_num), 4))) / self.circle_split_num * 2 * np.pi
                vs = array([[np.cos(e), np.sin(e)]
                            for e in es]) * obs_r
                text += self.polygon(np.zeros((2,)), vs)
                obs.append(
                    np.hstack((np.zeros((2,)), (vs + np.zeros((2,))).reshape((-1,)))))

        return text, obs

    def env_text(self, robot_text, obs_text, actuator_text):
        text = '<mujoco>\n'
        text += '<default>\n'
        text += '\t<geom friction="0.1 0.005 0.0001"/>\n'
        text += '</default>\n'
        text += '<visual>\n\t <global offwidth="1920" offheight="1080" />\n</visual>\n'
        text += '<worldbody>\n'
        text += '<light name="uplight_0" diffuse="0.4 0.4 0.4" pos="0 0 20" dir="0 0 -1" />\n'
        text += '<light name="uplight_1" diffuse="0.4 0.4 0.4" pos="10 10 20" dir="0 0 -1" />\n'
        text += '<light name="uplight_2" diffuse="0.4 0.4 0.4" pos="-10 -10 20" dir="0 0 -1" />\n'
        text += '<light name="uplight_3" diffuse="0.4 0.4 0.4" pos="10 -10 20" dir="0 0 -1" />\n'
        text += '<light name="uplight_4" diffuse="0.4 0.4 0.4" pos="-10 10 20" dir="0 0 -1" />\n'
        text += '<geom contype="2" friction="1 0.005 0.0001" pos="0 0 0" euler="0 0 0" type="plane" size="20 20 0.01" rgba="0.5 0.5 0.5 1"/>\n'
        text += '<camera mode="fixed" pos="0 0 15" ></camera>\n'
        text += robot_text
        text += obs_text
        text += '</worldbody>\n'
        text += actuator_text
        text += '</mujoco>'
        return text
