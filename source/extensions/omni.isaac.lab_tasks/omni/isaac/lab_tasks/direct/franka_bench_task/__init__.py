# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .franka_reach import FrankaReachEnv, FrankaReachEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-BenchTask-Reach-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_bench_task:FrankaReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaReachEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:reach_ppo_cfg.yaml",
    },
)
