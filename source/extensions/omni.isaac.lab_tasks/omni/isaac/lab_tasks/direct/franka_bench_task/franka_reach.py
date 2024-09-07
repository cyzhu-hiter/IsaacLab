from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform

import numpy as np
from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate



@configclass
class FrankaReachEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    num_actions = 9
    num_observations = 15
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/franka",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FactoryFranka/factory_franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.2,
                "panda_joint3": 0.0,
                "panda_joint4": -2.0,
                "panda_joint5": 0.0,
                "panda_joint6": 2.2,
                "panda_joint7": 0.8,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )
    
    # target object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )

    # target position for the reach task
    goal_pos = torch.tensor([0.5, 0.0, 0.25])

        # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 1
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 2.0
    action_penalty_scale = 0.05


class FrankaReachEnv(DirectRLEnv):
    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Create auxiliary variables for computing applied action, observations, and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        # Set hard limit on the last revolute joint
        self.robot_dof_lower_limits[6], self.robot_dof_upper_limits[6] = -torch.pi, torch.pi
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Extract the index of the useful part of the robot
        self.arm_fingertip_index = self._robot.find_bodies("panda_fingertip_centered")[0][0]

        # Initialize robot grasp positions
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0

        # Initialize goal position to a default value
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = self.cfg.goal_pos.to(self.device)
        self.goal_pos_init = self.goal_pos.clone()

        # Initialize goal marker for visualization
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # Unit vectors for randomizing goal orientation
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        # Add Franka robot to the scene
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.cfg.dof_velocity_scale * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        fingertip_pos = self._robot.data.body_pos_w[:, self.arm_fingertip_index]
        distance_to_target = torch.norm(fingertip_pos - self.goal_pos, dim=-1)

        terminated = distance_to_target < 0.05  # Consider task done if the fingertip is within 5cm of the target
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        fingertip_pos = self._robot.data.body_pos_w[:, self.arm_fingertip_index]
        distance_to_target = torch.norm(fingertip_pos - self.goal_pos, dim=-1)

        # dist_reward = self.cfg.dist_reward_scale * (1.0 / (1.0 + distance_to_target ** 2))
        # action_penalty = self.cfg.action_penalty_scale * torch.sum(self.actions ** 2, dim=-1)

        reward = -distance_to_target
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # Reset robot to random joint positions
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.1, 0.1, (len(env_ids), self._robot.num_joints), self.device
        )

        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Randomize new goal positions within the reach range
        rand_x = sample_uniform(-0.25, 0.1, (len(env_ids),), device=self.device)
        rand_y = sample_uniform(-0.2, 0.2, (len(env_ids),), device=self.device)
        rand_z = sample_uniform(-0.2, 0.1, (len(env_ids),), device=self.device)

        new_pos = torch.stack((rand_x, rand_y, rand_z), dim=-1)
        self.goal_pos[env_ids] = new_pos + self.goal_pos_init[env_ids, :] + self.scene.env_origins[env_ids, :]
        # self.goal_pos[env_ids] = self.goal_pos_init + self.scene.env_origins[env_ids, :]

        # Randomize goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        self.goal_rot[env_ids] = new_rot

        # Visualize the new goal position and orientation using the markers
        self.goal_markers.visualize(self.goal_pos, self.goal_rot)


    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        fingertip_pos_w = self._robot.data.body_pos_w[:, self.arm_fingertip_index]
        # to_target = self.goal_pos - fingertip_pos_w
        goal_pos = self.goal_pos - self.scene.env_origins
        fingertip_pos = fingertip_pos_w - self.scene.env_origins

        obs = torch.cat((dof_pos_scaled,
                         goal_pos,
                         fingertip_pos),
                         dim=-1)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )