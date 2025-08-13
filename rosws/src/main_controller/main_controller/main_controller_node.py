import sys
sys.path = sys.path + ['env/lib/python3.10/site-packages']
if sys.path[0] != '':
    sys.path = [''] + sys.path
from CppClass.Environment import Environment
import envCreator
import contourGenerator
import numpy as np
from numpy import zeros
from PIL import Image
from torch import rand
from matplotlib import pyplot as plt
import random
from base_config import PARAMs
import torch
from std_msgs.msg import Int16MultiArray, Float64MultiArray
from rclpy.publisher import Publisher
from rclpy.node import Node
import rclpy
from numpy import int16
import nornnsac
import numpy as np

torch.manual_seed(PARAMs["seed"])
np.random.seed(PARAMs["seed"])
random.seed(PARAMs["seed"])
torch.set_num_threads(torch.get_num_threads())

PARAMs["framerate"] = 25
PARAMs["max_ep_len"] = int(PARAMs["max_simu_second"] * PARAMs["framerate"])
PARAMs["hidden_sizes"] = [1024] * 4

model_file = "module_saves/nornn31/232h_54min_5719999steps_24636760updates_policy.ptd"


class MainController(Node):
    def __init__(self):
        super().__init__('MainController')
        self.declare_parameter('Nrobot', rclpy.Parameter.Type.INTEGER)
        self.Nrobot: int = self.get_parameter("Nrobot").value
        self.posvels = np.zeros((self.Nrobot, 6))
        self.action = np.zeros((self.Nrobot, 2))
        self.model_setted = False
        self.ctrl_publisher = self.create_publisher(Float64MultiArray,
                                                    "epuck2_connector/ctrl_cmd", 10)
        self.posvels_subscription = self.create_subscription(
            Float64MultiArray, "/vicon_connector/posvels", self.posvels_callback, 10)
        PARAMs["wheel_d"] = 0.053
        PARAMs["wheel_d"] = 0.002
        self.EC = envCreator.EnvCreator(PARAMs["robot_r"])
        self.env = Environment()
        PARAMs["rmax"] = self.env.setCtrl(vmax=PARAMs["vmax"], tau=PARAMs["tau"],
                                          wheel_d=PARAMs["wheel_d"], wheel_r=PARAMs["wheel_r"],
                                          gain=PARAMs["gain"])
        self.env.setRvop(dmax=PARAMs["dmax"], robot_r=PARAMs["robot_r"])
        self.env.setRwd(robot_r=PARAMs["robot_r"], vmax=PARAMs["vmax"], rmax=PARAMs["rmax"], tolerance=PARAMs["tolerance"], dreach=PARAMs["dreach"], tb=PARAMs["tb"],
                        a=PARAMs["a"], b=PARAMs["b"], c=PARAMs["c"], d=PARAMs["d"], e=PARAMs["e"],
                        f=PARAMs["f"], g=PARAMs["g"], eta=PARAMs["eta"],
                        h=PARAMs["h"], mu=PARAMs["mu"], rreach=PARAMs["rreach"],
                        remix=PARAMs["remix"], rm_middle=PARAMs["rm_middle"], dmax=PARAMs["dmax"], w=PARAMs["w"])
        self.Pi = nornnsac.nornncore.Policy(obs_dim=PARAMs["obs_dim"], act_dim=PARAMs["act_dim"],
                                            act_limit=PARAMs["act_limit"], hidden_sizes=PARAMs["hidden_sizes"])
        self.Pi.load_state_dict(torch.load(
            model_file, map_location=torch.device(PARAMs["device"])))
        self.Pi.to(device=PARAMs["device"])
        self.Pi.act_limit = self.Pi.act_limit.to(device=PARAMs["device"])

        for p in self.Pi.parameters():
            p.requires_grad = False

        self.Pi.eval()

    def setModel(self):
        pos = self.posvels.reshape((self.Nrobot, 6))[:, 0:2]
        xmean = pos[:, 0].mean()
        ymean = pos[:, 1].mean()
        pos[:, 0] = pos[:, 0] - xmean
        pos[:, 1] = pos[:, 1] - ymean
        self.target = -pos
        self.target[:, 0] = self.target[:, 0] + xmean
        self.target[:, 1] = self.target[:, 1] + ymean
        self.env.setReal(self.Nrobot, self.target, [])
        self.NNinput1 = np.frombuffer(
            self.env.get_NNinput1(), dtype=np.float64
        ).reshape(self.Nrobot, 180)
        self.reward = np.frombuffer(self.env.get_r(), dtype=np.float64)
        self.reward_mix = np.frombuffer(self.env.get_rm(), dtype=np.float64)
        self.death = np.frombuffer(self.env.get_d(), dtype=np.int32)
        self.ctrl = np.frombuffer(self.env.get_ctrl(), dtype=np.float64)
        self.network_timer = self.create_timer(
            0.04, self.network_step_callback)
        self.control_timer = self.create_timer(
            0.01, self.control_step_callback)
        pass

    def network_step_callback(self):
        self.env.setposvels(list(self.posvels))
        self.env.cal_obs(PARAMs["avevel"])
        self.env.cal_NNinput1(PARAMs["nullfill"])
        o = torch.as_tensor(
            self.NNinput1, dtype=torch.float32, device=PARAMs["device"])
        with torch.no_grad():
            a, logp = self.Pi(o, True, with_logprob=False)
        self.action = a.cpu().detach().numpy()
        for i in range(self.Nrobot):
            if self.death[i] == 1:
                self.action[i] = [0, 0]
        self.env.setvsG(self.action)

    def control_step_callback(self):
        self.env.cal_ctrl_vsG()
        msg = Float64MultiArray()
        msg.data = self.ctrl
        self.ctrl_publisher.publish(msg)
        pass

    def posvels_callback(self, msg: Float64MultiArray):
        # print(msg.data)
        self.posvels = np.array(msg.data)
        if not self.model_setted:
            self.setModel()
            self.model_setted = True


def main(args=None):
    rclpy.init(args=args)
    main_controller = MainController()
    rclpy.spin(main_controller)
    main_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
