import os
import time
from typing import Optional
import yaml

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC

MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_queue = []
context_size: Optional[int] = None


class ExplorerNode(Node):
    def __init__(self, args, model, model_params, noise_scheduler):
        super().__init__('explorer_node')
        self.args = args
        self.model = model
        self.model_params = model_params
        self.noise_scheduler = noise_scheduler
        self.context_size = model_params["context_size"]
        global context_size
        context_size = self.context_size
        self.create_subscription(Image, IMAGE_TOPIC, self.callback_obs, 1)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)
        self.timer = self.create_timer(1.0 / RATE, self.loop)

    def callback_obs(self, msg: Image):
        obs_img = msg_to_pil(msg)
        if len(context_queue) < self.context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

    def loop(self):
        if len(context_queue) <= self.context_size:
            return
        waypoint_msg = Float32MultiArray()
        obs_images = transform_images(context_queue, self.model_params["image_size"], center_crop=False)
        obs_images = obs_images.to(device)
        fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(device)
        mask = torch.ones(1).long().to(device)

        with torch.no_grad():
            obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)
            noisy_action = torch.randn((self.args.num_samples, self.model_params["len_traj_pred"], 2), device=device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        naction = to_numpy(get_action(naction))
        sampled_actions_msg = Float32MultiArray()
        sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
        self.sampled_actions_pub.publish(sampled_actions_msg)
        naction = naction[0]
        chosen_waypoint = naction[self.args.waypoint]
        if self.model_params["normalize"]:
            chosen_waypoint *= (MAX_V / RATE)
        waypoint_msg.data = chosen_waypoint
        self.waypoint_pub.publish(waypoint_msg)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GNM diffusion exploration using ROS2")
    parser.add_argument("--model", "-m", default="nomad", type=str,
                        help="model name (hint: check ../config/models.yaml) (default: nomad)")
    parser.add_argument("--waypoint", "-w", default=2, type=int,
                        help="index of the waypoint used for navigation")
    parser.add_argument("--num-samples", "-n", default=8, type=int,
                        help="number of actions sampled from the exploration model")
    args = parser.parse_args()

    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if not os.path.exists(ckpth_path):
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(ckpth_path, model_params, device)
    model = model.to(device)
    model.eval()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    rclpy.init()
    node = ExplorerNode(args, model, model_params, noise_scheduler)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
