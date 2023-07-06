from std_msgs.msg import Int16MultiArray, Float64MultiArray
from rclpy.publisher import Publisher
from rclpy.node import Node
import rclpy
from numpy import int16


class Epuck2Connector(Node):

    def __init__(self):
        super().__init__('Epuck2Connector')
        self.declare_parameter('id_list', rclpy.Parameter.Type.INTEGER_ARRAY)
        self.ids: list[float] = self.get_parameter("id_list").value
        self.ctrl_publishers: list[Publisher] = []
        self.gain = 6.0
        for id in self.ids:
            self.ctrl_publishers.append(self.create_publisher(Int16MultiArray,
                                                              "epuck2_robot_" + str(id) + "/mobile_base/cmd_ctrl", 10))
        self.ctrl_subscription = self.create_subscription(
            Float64MultiArray,
            'epuck2_connector/ctrl_cmd',
            self.ctrl_cmd_callback,
            10)

    def ctrl_cmd_callback(self, msg: Float64MultiArray):
        out_msg = Int16MultiArray()
        for i in range(self.ids.__len__()):
            out_msg.data = [int16(msg.data[i * 2] * self.gain),
                            int16(msg.data[i * 2 + 1] * self.gain)]
            self.ctrl_publishers[i].publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    epuck2_connector = Epuck2Connector()
    rclpy.spin(epuck2_connector)
    epuck2_connector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
