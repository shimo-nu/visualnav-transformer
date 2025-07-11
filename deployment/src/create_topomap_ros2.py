import argparse
import os
import shutil
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy

from utils import msg_to_pil
from topic_names import IMAGE_TOPIC

TOPOMAP_IMAGES_DIR = "../topomaps/images"


def remove_files_in_dir(dir_path: str) -> None:
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


class CreateTopomapNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('create_topomap_node')
        self.args = args
        self.obs_img = None
        self.last_msg_time = time.time()
        self.idx = 0
        self.topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            self.get_logger().info(
                f"{self.topomap_name_dir} already exists. Removing previous images...")
            remove_files_in_dir(self.topomap_name_dir)
        self.create_subscription(Image, IMAGE_TOPIC, self.cb_image, 1)
        self.create_subscription(Joy, 'joy', self.cb_joy, 1)
        assert args.dt > 0, 'dt must be positive'
        self.timer = self.create_timer(args.dt, self.loop)

    def cb_image(self, msg: Image):
        self.obs_img = msg_to_pil(msg)
        self.last_msg_time = time.time()

    def cb_joy(self, msg: Joy):
        if msg.buttons and msg.buttons[0]:
            self.get_logger().info('Shutting down...')
            rclpy.shutdown()

    def loop(self):
        if self.obs_img is not None:
            path = os.path.join(self.topomap_name_dir, f"{self.idx}.png")
            self.obs_img.save(path)
            self.get_logger().info(f'published image {self.idx}')
            self.idx += 1
            self.obs_img = None
        elif time.time() - self.last_msg_time > 2 * self.args.dt:
            self.get_logger().info(
                f'Topic {IMAGE_TOPIC} not publishing anymore. Shutting down...')
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description=f'Code to generate topomaps from the {IMAGE_TOPIC} topic')
    parser.add_argument('--dir', '-d', default='topomap', type=str,
                        help='path to topological map images in ../topomaps/images directory (default: topomap)')
    parser.add_argument('--dt', '-t', default=1., type=float,
                        help=f'time between images sampled from the {IMAGE_TOPIC} topic (default: 1.0)')
    args = parser.parse_args()

    rclpy.init()
    node = CreateTopomapNode(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
