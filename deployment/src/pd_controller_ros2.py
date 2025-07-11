import time
from typing import Tuple
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import WAYPOINT_TOPIC, REACHED_GOAL_TOPIC

CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1 / robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1


def clip_angle(theta: float) -> float:
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi


class ROSData:
    def __init__(self, timeout: int = 3, name: str = ""):
        self.timeout = timeout
        self.last_time = -float('inf')
        self.data = None
        self.name = name

    def set(self, data):
        self.data = data
        self.last_time = time.time()

    def get(self):
        return self.data

    def is_valid(self, verbose: bool = False) -> bool:
        elapsed = time.time() - self.last_time
        valid = elapsed < self.timeout
        if verbose and not valid:
            print(f"Not receiving {self.name} data for {elapsed} seconds (timeout: {self.timeout} seconds)")
        return valid


class PDControllerNode(Node):
    def __init__(self):
        super().__init__('pd_controller_node')
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, 1)
        self.create_subscription(Float32MultiArray, WAYPOINT_TOPIC, self.cb_waypoint, 1)
        self.create_subscription(Bool, REACHED_GOAL_TOPIC, self.cb_goal, 1)
        self.waypoint = ROSData(WAYPOINT_TIMEOUT, name='waypoint')
        self.reached_goal = False
        self.reverse_mode = False
        self.timer = self.create_timer(1.0 / RATE, self.loop)

    def cb_waypoint(self, msg: Float32MultiArray):
        self.waypoint.set(msg.data)

    def cb_goal(self, msg: Bool):
        self.reached_goal = msg.data

    def pd_controller(self, waypoint: np.ndarray) -> Tuple[float, float]:
        assert len(waypoint) in (2, 4)
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint
        if len(waypoint) == 4 and abs(dx) < EPS and abs(dy) < EPS:
            v = 0.0
            w = clip_angle(np.arctan2(hy, hx)) / DT
        elif abs(dx) < EPS:
            v = 0.0
            w = np.sign(dy) * np.pi / (2 * DT)
        else:
            v = dx / DT
            w = np.arctan(dy / dx) / DT
        v = float(np.clip(v, 0.0, MAX_V))
        w = float(np.clip(w, -MAX_W, MAX_W))
        return v, w

    def loop(self):
        vel_msg = Twist()
        if self.reached_goal:
            self.vel_pub.publish(vel_msg)
            self.get_logger().info('Reached goal! Stopping...')
            return
        if self.waypoint.is_valid(verbose=True):
            v, w = self.pd_controller(np.array(self.waypoint.get()))
            if self.reverse_mode:
                v *= -1
            vel_msg.linear.x = v
            vel_msg.angular.z = w
            self.get_logger().info(f'publishing new vel: {v}, {w}')
        self.vel_pub.publish(vel_msg)


def main():
    rclpy.init()
    node = PDControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
