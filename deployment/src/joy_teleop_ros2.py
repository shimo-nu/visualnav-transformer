import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from topic_names import JOY_BUMPER_TOPIC

CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
VEL_TOPIC = robot_config["vel_teleop_topic"]
JOY_CONFIG_PATH = "../config/joystick.yaml"
with open(JOY_CONFIG_PATH, "r") as f:
    joy_config = yaml.safe_load(f)
DEADMAN_SWITCH = joy_config["deadman_switch"]
LIN_VEL_BUTTON = joy_config["lin_vel_button"]
ANG_VEL_BUTTON = joy_config["ang_vel_button"]
MAX_V = 0.4
MAX_W = 0.8
RATE = 9


class JoyTeleopNode(Node):
    def __init__(self):
        super().__init__('joy2locobot')
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, 1)
        self.bumper_pub = self.create_publisher(Bool, JOY_BUMPER_TOPIC, 1)
        self.create_subscription(Joy, 'joy', self.callback_joy, 1)
        self.timer = self.create_timer(1.0 / RATE, self.publish)
        self.vel_msg = Twist()
        self.button = False
        self.bumper = False

    def callback_joy(self, data: Joy):
        self.button = data.buttons[DEADMAN_SWITCH]
        bumper_button = data.buttons[DEADMAN_SWITCH - 1]
        if self.button:
            self.vel_msg.linear.x = MAX_V * data.axes[LIN_VEL_BUTTON]
            self.vel_msg.angular.z = MAX_W * data.axes[ANG_VEL_BUTTON]
        else:
            self.vel_msg = Twist()
        self.bumper = bool(bumper_button)

    def publish(self):
        if self.button:
            self.vel_pub.publish(self.vel_msg)
        bumper_msg = Bool()
        bumper_msg.data = self.bumper
        self.bumper_pub.publish(bumper_msg)
        if self.bumper:
            self.get_logger().info('Bumper pressed!')


def main():
    rclpy.init()
    node = JoyTeleopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
