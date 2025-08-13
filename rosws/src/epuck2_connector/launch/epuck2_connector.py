from launch import LaunchDescription
from launch_ros.actions import Node
# id2ip = {
#     4857: "192.168.28.66",
#     4847: "192.168.28.146",
#     4791: "192.168.28.250",
#     5082: "192.168.28.42",
# }
# id2ip = {
#     4857: "192.168.12.151",
#     4847: "192.168.12.6",
#     4791: "192.168.12.44",
#     5082: "192.168.12.127",
# }
id2ip = {
    4857: "10.42.0.77",
    4847: "10.42.0.176",
    4791: "10.42.0.203",
    5082: "10.42.0.53",
}
id_list = []
ip_list = []
for (key, value) in id2ip.items():
    id_list.append(key)
id_list.sort()
for id in id_list:
    ip_list.append(id2ip[id])
# init pose x y z, enable cam ,enable ground
init_list = [[0.0, 0.0, 0.0, False, False]]*ip_list.__len__()

if ip_list.__len__() != id_list.__len__() or ip_list.__len__() != init_list.__len__():
    exit(1)
is_single_robot = 1 if ip_list.__len__() == 1 else 0


def generate_launch_description():
    return LaunchDescription([
        Node(
            # namespace="epuck2_connector",
            package="epuck2_connector",
            executable="epuck2_connector_node",
            name="epuck2_connector_node",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"id_list": id_list},
            ]
        )
    ])
