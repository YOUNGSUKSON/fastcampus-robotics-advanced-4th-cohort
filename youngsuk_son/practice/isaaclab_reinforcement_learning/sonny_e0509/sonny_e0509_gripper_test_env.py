# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

USD_DIR = Path(__file__).parent / "USD"
GRIPPER_JOINT_NAMES = ("rh_l1", "rh_r1", "rh_l2", "rh_r2")


@configclass
class SysE0509GripperTestEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 4.0
    decimation = 2
    # shared gripper(1) only
    action_space = 1
    # dof_pos(10) + dof_vel(10) + close_ratio + grip_set + grip_read
    observation_space = 23
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_DIR / "e0509_sonny.usd"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": float(torch.pi / 2),
                "joint_4": 0.0,
                "joint_5": float(torch.pi / 2),
                "joint_6": 0.0,
                "rh_l1": 0.0,
                "rh_r1": 0.0,
                "rh_l2": 0.0,
                "rh_r2": 0.0,
            }
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit_sim=330.0,
                stiffness=200.0,
                damping=20.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[*GRIPPER_JOINT_NAMES],
                effort_limit_sim=220.0,
                stiffness=240.0,
                damping=16.0,
            ),
        },
    )

    action_scale = 6.0
    dof_velocity_scale = 0.1
    gripper_joint_speed_scale = 1.5

    # gripper close test reward
    gripper_pos_open = 0.0
    gripper_pos_close = 1.1
    gripper_close_reward_scale = +8.0
    arm_motion_penalty_scale = -2.0

    success_close_ratio = 0.95


class SysE0509GripperTestEnv(DirectRLEnv):
    cfg: SysE0509GripperTestEnvCfg

    def __init__(self, cfg: SysE0509GripperTestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_targets = self._robot.data.default_joint_pos.clone()

        self.arm_joint_ids = self._robot.find_joints("joint_[1-6]")[0]
        self.gripper_joint_ids = torch.tensor(
            [self._robot.find_joints(name)[0][0] for name in GRIPPER_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self.gripper_joint_ids] = self.cfg.gripper_joint_speed_scale

        self.arm_home_joint_pos = torch.tensor(
            [0.0, 0.0, float(torch.pi / 2), 0.0, float(torch.pi / 2), 0.0],
            dtype=torch.float32,
            device=self.device,
        )

        self.gripper_pos_open = float(self.cfg.gripper_pos_open)
        self.gripper_pos_close = float(self.cfg.gripper_pos_close)
        self.gripper_pos_span = self.gripper_pos_close - self.gripper_pos_open
        if abs(self.gripper_pos_span) < 1.0e-6:
            raise ValueError("gripper_pos_open and gripper_pos_close must be different.")

        self.joint_1_idx = self._robot.find_joints("joint_1")[0][0]
        self.joint_2_idx = self._robot.find_joints("joint_2")[0][0]
        self.joint_3_idx = self._robot.find_joints("joint_3")[0][0]
        self.joint_4_idx = self._robot.find_joints("joint_4")[0][0]
        self.joint_5_idx = self._robot.find_joints("joint_5")[0][0]
        self.joint_6_idx = self._robot.find_joints("joint_6")[0][0]

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

        full_actions = torch.zeros_like(self.robot_dof_targets)
        # keep arm fixed at home pose; policy controls only gripper
        self.robot_dof_targets[:, self.arm_joint_ids] = self.arm_home_joint_pos
        shared_gripper_action = self.actions[:, :1]
        full_actions[:, self.gripper_joint_ids] = shared_gripper_action.expand(-1, len(self.gripper_joint_ids))

        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.cfg.action_scale * full_actions
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        gripper_read_mean = torch.mean(self._robot.data.joint_pos[:, self.gripper_joint_ids], dim=-1)
        close_ratio = torch.clamp((gripper_read_mean - self.gripper_pos_open) / self.gripper_pos_span, 0.0, 1.0)
        success = close_ratio > self.cfg.success_close_ratio
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return success, truncated

    def _get_rewards(self) -> torch.Tensor:
        gripper_set = self.robot_dof_targets[:, self.gripper_joint_ids]
        gripper_read = self._robot.data.joint_pos[:, self.gripper_joint_ids]
        gripper_set_mean = torch.mean(gripper_set, dim=-1)
        gripper_read_mean = torch.mean(gripper_read, dim=-1)
        gripper_abs_err = torch.abs(gripper_set - gripper_read).mean(dim=-1)
        close_ratio = torch.clamp((gripper_read_mean - self.gripper_pos_open) / self.gripper_pos_span, 0.0, 1.0)

        close_reward = self.cfg.gripper_close_reward_scale * close_ratio
        arm_motion = torch.sum(torch.abs(self._robot.data.joint_pos[:, self.arm_joint_ids] - self.arm_home_joint_pos), dim=-1)
        arm_motion_penalty = self.cfg.arm_motion_penalty_scale * arm_motion

        rewards = close_reward + arm_motion_penalty

        if "log" not in self.extras:
            self.extras["log"] = {}

        self.extras["log"].update(
            {
                "KPI/01_close_ratio": close_ratio.mean(),
                "KPI/02_close_reward": close_reward.mean(),
                "KPI/03_arm_motion_pen": arm_motion_penalty.mean(),
                "Diag/grip_set_mean": gripper_set_mean.mean(),
                "Diag/grip_read_mean": gripper_read_mean.mean(),
                "Diag/grip_abs_err_mean": gripper_abs_err.mean(),
                "Diag/grip_set_rh_l1": gripper_set[0, 0],
                "Diag/grip_set_rh_r1": gripper_set[0, 1],
                "Diag/grip_set_rh_l2": gripper_set[0, 2],
                "Diag/grip_set_rh_r2": gripper_set[0, 3],
                "Diag/grip_read_rh_l1": gripper_read[0, 0],
                "Diag/grip_read_rh_r1": gripper_read[0, 1],
                "Diag/grip_read_rh_l2": gripper_read[0, 2],
                "Diag/grip_read_rh_r2": gripper_read[0, 3],
                "Diag/arm_motion": arm_motion.mean(),
                "Ref/close_ratio_max": close_ratio.max(),
                "Ref/close_ratio_min": close_ratio.min(),
                "Ref/reward_max": rewards.max(),
                "Ref/reward_min": rewards.min(),
            }
        )

        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        joint_pos[:, self.joint_1_idx] = 0.0
        joint_pos[:, self.joint_2_idx] = 0.0
        joint_pos[:, self.joint_3_idx] = float(torch.pi / 2)
        joint_pos[:, self.joint_4_idx] = 0.0
        joint_pos[:, self.joint_5_idx] = float(torch.pi / 2)
        joint_pos[:, self.joint_6_idx] = 0.0
        joint_pos[:, self.gripper_joint_ids] = 0.0

        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        gripper_set_mean = torch.mean(self.robot_dof_targets[:, self.gripper_joint_ids], dim=-1, keepdim=True)
        gripper_read_mean = torch.mean(self._robot.data.joint_pos[:, self.gripper_joint_ids], dim=-1, keepdim=True)
        close_ratio = torch.clamp((gripper_read_mean - self.gripper_pos_open) / self.gripper_pos_span, 0.0, 1.0)

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                close_ratio,
                gripper_set_mean,
                gripper_read_mean,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
