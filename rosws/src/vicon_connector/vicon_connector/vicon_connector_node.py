from std_msgs.msg import Int16MultiArray, Float64MultiArray
from rclpy.publisher import Publisher
from rclpy.node import Node
import rclpy
from numpy import int16
import numpy as np
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg._transform_stamped import TransformStamped


class ViconConnector(Node):

    def __init__(self):
        super().__init__('ViconConnector')
        self.declare_parameter('id_list', rclpy.Parameter.Type.INTEGER_ARRAY)
        self.ids: list[float] = self.get_parameter("id_list").value
        self.posvels = np.zeros((self.ids.__len__(), 6), dtype=np.float64)
        self.posvels_publisher = self.create_publisher(Float64MultiArray,
                                                       "vicon_connector/posvels", 10)
        self.first: int = 2
        self.time_stamp = np.zeros((self.ids.__len__(), 2), dtype=np.int64)
        self.rho = 1.0
        self.oneminrho = 1.0 - self.rho
        self.vicon_subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.vicon_callback_first,
            10)

    def vicon_callback_first(self, msg: TFMessage):
        if self.first == 2:
            self.vicon_id_list = []
            for m in msg.transforms:
                m: TransformStamped
                self.vicon_id_list.append(int(m.child_frame_id[-4:]))

            self.vicon_id_order = []
            for id in self.vicon_id_list:
                self.vicon_id_order.append(self.ids.index(id))

            i = 0
            for m in msg.transforms:
                m: TransformStamped
                vo = self.vicon_id_order[i]
                x_old = self.posvels[vo][0]
                y_old = self.posvels[vo][1]
                r_old = self.posvels[vo][2]
                self.posvels[vo][0] = m.transform.translation.x
                self.posvels[vo][1] = m.transform.translation.y
                # this->posvels[i * 6 + 2] = atan2(2 * this->d->qpos[qposadr[i] + 3] * this->d->qpos[qposadr[i] + 6],
                #                      1 - 2 * this->d->qpos[qposadr[i] + 6] * this->d->qpos[qposadr[i] + 6]);
                self.posvels[vo][2] = np.arctan2(2 * m.transform.rotation.w * m.transform.rotation.z,
                                                 1 - 2 * m.transform.rotation.z**2)

                self.time_stamp[vo][0] = m.header.stamp.sec
                self.time_stamp[vo][1] = m.header.stamp.nanosec

                i += 1
            self.first -= 1

        elif self.first == 1:
            i = 0
            for m in msg.transforms:
                m: TransformStamped
                vo = self.vicon_id_order[i]
                x_old = self.posvels[vo][0]
                y_old = self.posvels[vo][1]
                r_old = self.posvels[vo][2]
                self.posvels[vo][0] = m.transform.translation.x
                self.posvels[vo][1] = m.transform.translation.y
                # this->posvels[i * 6 + 2] = atan2(2 * this->d->qpos[qposadr[i] + 3] * this->d->qpos[qposadr[i] + 6],
                #                      1 - 2 * this->d->qpos[qposadr[i] + 6] * this->d->qpos[qposadr[i] + 6]);
                self.posvels[vo][2] = np.arctan2(2 * m.transform.rotation.w * m.transform.rotation.z,
                                                 1 - 2 * m.transform.rotation.z**2)

                self.posvels[vo][0] = self.posvels[vo][0] * self.rho \
                    + self.oneminrho * x_old
                self.posvels[vo][1] = self.posvels[vo][1] * self.rho \
                    + self.oneminrho * y_old
                self.posvels[vo][2] = self.posvels[vo][2] * self.rho \
                    + self.oneminrho * r_old

                delta_t = float(m.header.stamp.sec - self.time_stamp[vo][0]) +\
                    float(m.header.stamp.nanosec - self.time_stamp[vo][1]) / 1e9 \

                self.time_stamp[vo][0] = m.header.stamp.sec
                self.time_stamp[vo][1] = m.header.stamp.nanosec
                self.posvels[vo][3] = (self.posvels[vo][0] - x_old) / delta_t
                self.posvels[vo][4] = (self.posvels[vo][1] - y_old) / delta_t
                self.posvels[vo][5] = (self.posvels[vo][2] - r_old) / delta_t

                i += 1
            self.vicon_subscription.callback = self.vicon_callback
            # self.vicon_subscription = self.create_subscription(
            #     TFMessage,
            #     '/tf',
            #     self.vicon_callback,
            #     10)

    def vicon_callback(self, msg: TFMessage):
        out_msg = Float64MultiArray()
        i = 0
        for m in msg.transforms:
            m: TransformStamped
            vo = self.vicon_id_order[i]
            x_old = self.posvels[vo][0]
            y_old = self.posvels[vo][1]
            r_old = self.posvels[vo][2]
            dx_old = self.posvels[vo][3]
            dy_old = self.posvels[vo][4]
            dr_old = self.posvels[vo][5]
            self.posvels[vo][0] = m.transform.translation.x
            self.posvels[vo][1] = m.transform.translation.y
            # this->posvels[i * 6 + 2] = atan2(2 * this->d->qpos[qposadr[i] + 3] * this->d->qpos[qposadr[i] + 6],
            #                      1 - 2 * this->d->qpos[qposadr[i] + 6] * this->d->qpos[qposadr[i] + 6]);
            self.posvels[vo][2] = np.arctan2(2 * m.transform.rotation.w * m.transform.rotation.z,
                                             1 - 2 * m.transform.rotation.z**2)

            self.posvels[vo][0] = self.posvels[vo][0] * self.rho \
                + self.oneminrho * x_old
            self.posvels[vo][1] = self.posvels[vo][1] * self.rho \
                + self.oneminrho * y_old
            self.posvels[vo][2] = self.posvels[vo][2] * self.rho \
                + self.oneminrho * r_old

            delta_t = float(m.header.stamp.sec - self.time_stamp[vo][0]) +\
                float(m.header.stamp.nanosec - self.time_stamp[vo][1]) / 1e9 \

            self.time_stamp[vo][0] = m.header.stamp.sec
            self.time_stamp[vo][1] = m.header.stamp.nanosec

            self.posvels[vo][3] = (self.posvels[vo][0] - x_old) / delta_t
            self.posvels[vo][4] = (self.posvels[vo][1] - y_old) / delta_t
            self.posvels[vo][5] = (self.posvels[vo][2] - r_old) / delta_t

            self.posvels[vo][3] = self.posvels[vo][3] * self.rho \
                + self.oneminrho * dx_old
            self.posvels[vo][4] = self.posvels[vo][4] * self.rho \
                + self.oneminrho * dy_old
            self.posvels[vo][5] = self.posvels[vo][5] * self.rho \
                + self.oneminrho * dr_old
            i += 1
        out_msg.data = self.posvels.flatten()
        self.posvels_publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    vicon_connector = ViconConnector()
    rclpy.spin(vicon_connector)
    vicon_connector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
