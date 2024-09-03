# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate on single GPU.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb", action="store_true", default=False, help="Use wandb to track the training progress")

# extra parallel reinforcement learning related hyperparameters
parser.add_argument("--tuning_param", type=str, default='mb', choices=['mb', 'me', 'np', 'ne'],
                    help="Select the parameter to be tuned: 'mb' for minibatch size, 'me' for mini epochs, 'np' for number of GPUs per node, or 'ne' for number of environments per GPU.")
parser.add_argument("--minibatch_size", type=int, default=None, help="The size of each minibatch during training.")
parser.add_argument("--mini_epochs", type=int, default=None, help="The number of mini epochs to run during training.")

# extra argument
parser.add_argument("--pre-ex", action="store_true", default=False, help="Flag to indicate if the run is a pre-experiment for exploring hyperparameter influence.")
parser.add_argument("--max_time", type=int, default=None, help="Maximum time for this task to train")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from utilis import WandbManager, adjust_gpu_parameters


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    "Parameter Management with overriding"
    # override configurations with non-hydra CLI arguments
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None
    
    # override configurations with PRL related arguments
    if args_cli.num_envs is not None:
        env_cfg = adjust_gpu_parameters(env_cfg, args_cli.num_envs)
        env_cfg.scene.num_envs = args_cli.num_envs
    args_cli.num_envs = env_cfg.scene.num_envs
    agent_cfg["params"]["config"]["minibatch_size"] = (
        args_cli.minibatch_size if args_cli.minibatch_size is not None else agent_cfg["params"]["config"]["minibatch_size"]
    )
    args_cli.minibatch_size = agent_cfg["params"]["config"]["minibatch_size"]
    agent_cfg["params"]["config"]["mini_epochs"] = (
        args_cli.mini_epochs if args_cli.mini_epochs is not None else agent_cfg["params"]["config"]["mini_epochs"]
    )
    args_cli.mini_epochs = agent_cfg["params"]["config"]["mini_epochs"]
    args_cli.num_nodes = int(os.getenv("NUM_NODES", "1"))
    args_cli.nproc_per_node = int(int(os.getenv("WORLD_SIZE", "1"))/args_cli.num_nodes)
    agent_cfg["params"]["config"]["max_time"] = args_cli.max_time if args_cli.max_time is not None else 10e6
    
    """Initialize wandb if necessary"""
    if int(os.getenv("RANK", "0")) == 0 and args_cli.wandb:
        wandb_manager = WandbManager()
        wandb_manager.initialize(args_cli)

    """Train with RL-Games agent."""
    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # set seed of the env
    env.seed(agent_cfg["params"]["seed"])
    # reset the agent and env
    runner.reset()
    # train the agent
    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()
    
    # close wandb
    if int(os.getenv("RANK", "0")) == 0 and args_cli.wandb:
        wandb_manager.finish()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    time.sleep(2)  # delay to ensure all resources are released
    simulation_app.close()
