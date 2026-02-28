# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

USD_DIR = Path(__file__).parent / "USD"
GRIPPER_JOINT_NAMES = ("rh_l1", "rh_r1", "rh_l2", "rh_r2")


@configclass
class SysE0509PickEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6.0
    decimation = 2
    # 6 arm joints + 1 shared gripper command
    action_space = 7
    observation_space = 33
    state_space = 0
    viewer = ViewerCfg(
        eye=(0.95, 0.75, 1.05),
        lookat=(0.35, 0.0, 0.78),
        origin_type="env",
        env_index=0,
    )

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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # e0509 + table + snack box are all inside this single USD
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_DIR / "e0509_sonny.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
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
                effort_limit_sim=120.0,
                stiffness=80.0,
                damping=8.0,
            ),
        },
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/e0509/.*",
        history_length=3,
        track_pose=True,
    )

    # snack box already exists inside e0509_sonny.usd
    snack_box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot/_03_cracker_box",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.39285892, -0.00069323, 0.72404516),
            rot=(0.99987829, 0.01559304, 0.00044949, -0.00020867),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    action_scale = 4.0
    dof_velocity_scale = 0.1

    # reward weights: good terms (+), bad terms (-)
    dist_reward_scale = +2.5
    lift_reward_scale = +38.0
    success_reward = +40.0
    axis_align_reward_scale = +13.0
    # staged gripper shaping (soft-gated by pose+distance)
    gripper_close_action_reward_scale = +8.0
    # proportional gripper-position tracking as alignment/distance become ready
    gripper_target_track_reward_scale = +10.0
    gripper_target_progress_reward_scale = +12.0
    gripper_target_pos_tolerance = 0.025
    gripper_target_progress_norm = 0.02
    # keep gripper policy alive from early phase via dense width-progress + motion rewards
    gripper_width_progress_reward_scale = +18.0
    gripper_width_progress_norm_m = 0.004
    gripper_motion_reward_scale = +1.0
    gripper_motion_reward_norm_m = 0.006
    grasp_ready_reward_scale = +2.0
    grasp_enter_reward_scale = +3.0
    grasp_enter_gate_threshold = 0.55
    grip_gate_pose_center = 0.90
    grip_gate_pose_k = 20.0
    grip_gate_dist_center = 0.04
    grip_gate_dist_k = 60.0
    # +1 means positive shared gripper action closes fingers, -1 means opposite
    gripper_close_action_sign = +1.0

    # width-fit reward (0.0~1.1 -> 109mm~0mm)
    # pre-ready term makes width target shaping available from episode start
    gripper_width_match_pre_reward_scale = +16.0
    gripper_width_match_reward_scale = +6.0
    gripper_pos_open = 0.0
    gripper_pos_close = 1.1
    gripper_open_width_m = 0.109
    gripper_close_width_m = 0.0
    # width target margin by stage:
    # pre-ready: easier pre-grasp width, ready: tighter grasp width
    gripper_width_target_pre_margin_m = 0.0126
    gripper_width_target_ready_margin_m = -0.0002
    gripper_width_match_tolerance_m = 0.008

    arm_action_penalty_scale = -0.003

    # distance reward is gated by pose quality
    dist_pose_gate_min = 0.07
    pose_reward_proximity_k = 4.0
    pose_reward_far_scale = 0.25

    collision_penalty_scale = -0.2
    collision_force_threshold = 5.0
    object_collision_ignore_margin = 0.02

    object_size_xyz = (0.0574126, 0.07470295, 0.05026)
    # object xy size (m): short side axis is selected automatically (x if x<=y else y)
    object_xy_size = (0.0574126, 0.07470295)

    # virtual gripper-center frame in rh_p12_rn_base frame (same orientation as base link)
    gripper_center_offset_b = (0.0, 0.0, 0.115)
    debug_frame_overlay = True

    lift_height_success = 0.10
    success_gripper_dist = 0.06
    # hold robot for a few RL steps right after reset to avoid reset-impact dip
    reset_hold_steps = 15


class SysE0509PickEnv(DirectRLEnv):
    cfg: SysE0509PickEnvCfg

    def __init__(self, cfg: SysE0509PickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.gripper_joint_ids = torch.tensor(
            [self._robot.find_joints(name)[0][0] for name in GRIPPER_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.arm_joint_ids = self._robot.find_joints("joint_[1-6]")[0]

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        # revert to original gripper step-size setting:
        # dt*(1/60) * action_scale(4.0) * speed_scale(0.6) => max 0.04 joint-step at |action|=1
        self.robot_dof_speed_scales[self.gripper_joint_ids] = 0.6
        self.robot_dof_targets = self._robot.data.default_joint_pos.clone()
        self._applied_gripper_action = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_grasp_ready = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._reset_hold_steps_left = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self.gripper_base_body_idx = self._robot.find_bodies("rh_p12_rn_base")[0][0]
        self.gripper_center_offset_b = torch.tensor(self.cfg.gripper_center_offset_b, dtype=torch.float32, device=self.device)

        body_name_to_idx = {name: i for i, name in enumerate(self._robot.body_names)}
        self.gripper_left_body_idx = body_name_to_idx.get("rh_l1", None)
        self.gripper_right_body_idx = body_name_to_idx.get("rh_r1", None)
        self.use_finger_center = (self.gripper_left_body_idx is not None) and (self.gripper_right_body_idx is not None)

        self.local_x_axis_b = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.local_x_axis_b = self.local_x_axis_b.expand(self.num_envs, -1)
        self.local_y_axis_b = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.local_y_axis_b = self.local_y_axis_b.expand(self.num_envs, -1)
        self.local_z_axis_b = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.local_z_axis_b = self.local_z_axis_b.expand(self.num_envs, -1)

        if self.cfg.object_xy_size[0] <= self.cfg.object_xy_size[1]:
            short_axis_local = (1.0, 0.0, 0.0)
            long_axis_local = (0.0, 1.0, 0.0)
        else:
            short_axis_local = (0.0, 1.0, 0.0)
            long_axis_local = (1.0, 0.0, 0.0)
        self.object_short_axis_b = torch.tensor(short_axis_local, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.object_short_axis_b = self.object_short_axis_b.expand(self.num_envs, -1)
        self.object_long_axis_b = torch.tensor(long_axis_local, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.object_long_axis_b = self.object_long_axis_b.expand(self.num_envs, -1)

        self.object_short_size_m = float(min(self.cfg.object_xy_size))
        self.gripper_width_target_pre_m = self.object_short_size_m + float(self.cfg.gripper_width_target_pre_margin_m)
        self.gripper_width_target_ready_m = self.object_short_size_m + float(self.cfg.gripper_width_target_ready_margin_m)

        self.gripper_pos_open = float(self.cfg.gripper_pos_open)
        self.gripper_pos_close = float(self.cfg.gripper_pos_close)
        if self.cfg.gripper_close_action_sign < 0.0:
            self.gripper_pos_open, self.gripper_pos_close = self.gripper_pos_close, self.gripper_pos_open
        self.gripper_pos_span = self.gripper_pos_close - self.gripper_pos_open
        if abs(self.gripper_pos_span) < 1.0e-6:
            raise ValueError("gripper_pos_open and gripper_pos_close must be different for width mapping.")

        width_span = self.cfg.gripper_close_width_m - self.cfg.gripper_open_width_m
        if abs(width_span) < 1.0e-6:
            raise ValueError("gripper_open_width_m and gripper_close_width_m must be different.")
        pre_target_ratio = torch.clamp(
            torch.tensor(
                (self.gripper_width_target_pre_m - self.cfg.gripper_open_width_m) / width_span,
                dtype=torch.float32,
                device=self.device,
            ),
            0.0,
            1.0,
        )
        self.gripper_target_pre_pos = float(self.gripper_pos_open + pre_target_ratio.item() * self.gripper_pos_span)

        init_gripper_pos_mean = torch.mean(self._robot.data.joint_pos[:, self.gripper_joint_ids], dim=-1)
        init_grip_close_ratio = torch.clamp(
            (init_gripper_pos_mean - self.gripper_pos_open) / self.gripper_pos_span, min=0.0, max=1.0
        )
        init_gripper_width_m = self.cfg.gripper_open_width_m + init_grip_close_ratio * (
            self.cfg.gripper_close_width_m - self.cfg.gripper_open_width_m
        )
        init_width_error_m = torch.abs(init_gripper_width_m - self.gripper_width_target_pre_m)
        self._prev_gripper_pos_mean = init_gripper_pos_mean.detach().clone()
        self._prev_gripper_width_error_m = init_width_error_m.detach().clone()
        init_target_err = torch.abs(init_gripper_pos_mean - self.gripper_pos_open)
        self._prev_gripper_target_err = init_target_err.detach().clone()

        self.gripper_center_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.gripper_center_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        self.joint_1_idx = self._robot.find_joints("joint_1")[0][0]
        self.joint_2_idx = self._robot.find_joints("joint_2")[0][0]
        self.joint_3_idx = self._robot.find_joints("joint_3")[0][0]
        self.joint_4_idx = self._robot.find_joints("joint_4")[0][0]
        self.joint_5_idx = self._robot.find_joints("joint_5")[0][0]
        self.joint_6_idx = self._robot.find_joints("joint_6")[0][0]

        # track non-gripper and gripper contact body ids
        contact_body_names = self._contact_sensor.body_names
        robot_body_name_set = set(self._robot.body_names)
        penalty_ids = [
            i for i, name in enumerate(contact_body_names) if (name in robot_body_name_set) and (not name.startswith("rh_"))
        ]
        gripper_ids = [i for i, name in enumerate(contact_body_names) if (name in robot_body_name_set) and name.startswith("rh_")]
        self.collision_penalty_body_ids = torch.tensor(penalty_ids, dtype=torch.long, device=self.device)
        self.gripper_collision_body_ids = torch.tensor(gripper_ids, dtype=torch.long, device=self.device)

        self.object_half_extents_b = 0.5 * torch.tensor(self.cfg.object_size_xyz, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.object_init_height = float(self.cfg.snack_box.init_state.pos[2])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._snack_box = RigidObject(self.cfg.snack_box)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["snack_box"] = self._snack_box
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.cfg.debug_frame_overlay:
            frame_cfg = FRAME_MARKER_CFG.copy()
            frame_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
            self._gripper_frame_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/Debug/gripper_center"))
            self._object_frame_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/Debug/object_frame"))
        else:
            self._gripper_frame_marker = None
            self._object_frame_marker = None

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # map policy action to full joint action: [arm(6), shared_gripper(1)]
        full_actions = torch.zeros_like(self.robot_dof_targets)
        full_actions[:, self.arm_joint_ids] = self.actions[:, : len(self.arm_joint_ids)]

        shared_gripper_action = self.actions[:, -1:]
        self._applied_gripper_action = shared_gripper_action.squeeze(-1)
        full_actions[:, self.gripper_joint_ids] = shared_gripper_action.expand(-1, len(self.gripper_joint_ids))

        # freeze policy action for a short settle window after reset
        hold_mask = self._reset_hold_steps_left > 0
        if torch.any(hold_mask):
            full_actions[hold_mask] = 0.0
            self.actions[hold_mask] = 0.0
            self._applied_gripper_action[hold_mask] = 0.0
            self._reset_hold_steps_left[hold_mask] -= 1

        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.cfg.action_scale * full_actions
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        gripper_pos, box_pos, _, _ = self._compute_intermediate_values()

        distance = torch.norm(gripper_pos - box_pos, dim=-1)
        lifted = box_pos[:, 2] > (self.object_init_height + self.cfg.lift_height_success)
        close_to_gripper = distance < self.cfg.success_gripper_dist
        success = lifted & close_to_gripper

        dropped = box_pos[:, 2] < (self.object_init_height - 0.08)
        # any non-object collision on robot links is treated as failure
        arm_collision_fail = self._has_non_object_collision(self.collision_penalty_body_ids)
        gripper_collision_fail = self._has_non_object_collision(self.gripper_collision_body_ids)
        terminated = success | dropped | arm_collision_fail | gripper_collision_fail

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _has_non_object_collision(self, body_ids: torch.Tensor) -> torch.Tensor:
        if body_ids.numel() == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, :, body_ids]
        contact_force_mag = torch.norm(net_contact_forces, dim=-1)
        is_collision = torch.max(contact_force_mag, dim=1)[0] > self.cfg.collision_force_threshold

        body_pos_w = self._contact_sensor.data.pos_w[:, body_ids]
        box_pos_w = self._snack_box.data.root_pos_w
        box_quat_w = self._snack_box.data.root_quat_w
        body_to_box_w = body_pos_w - box_pos_w.unsqueeze(1)

        num_bodies = body_ids.numel()
        box_quat_repeat = box_quat_w.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
        body_to_box_b = quat_apply_inverse(box_quat_repeat, body_to_box_w.reshape(-1, 3)).reshape(
            self.num_envs, num_bodies, 3
        )
        inside_object_obb = torch.all(
            torch.abs(body_to_box_b) <= (self.object_half_extents_b + self.cfg.object_collision_ignore_margin), dim=-1
        )

        is_collision = is_collision & (~inside_object_obb)
        return torch.any(is_collision, dim=1)

    def _get_rewards(self) -> torch.Tensor:
        gripper_pos, box_pos, box_lin_vel, box_yaw_deg = self._compute_intermediate_values()

        distance = torch.norm(gripper_pos - box_pos, dim=-1)
        dist_reward_raw = 1.0 / (1.0 + distance * distance)

        lift_amount = torch.clamp(box_pos[:, 2] - self.object_init_height, min=0.0, max=0.25)

        lifted = box_pos[:, 2] > (self.object_init_height + self.cfg.lift_height_success)
        close_to_gripper = distance < self.cfg.success_gripper_dist
        success = lifted & close_to_gripper

        # reward when gripper/object local Z axes face each other (target: 180deg, dot -> -1)
        gripper_z_w = quat_apply(self.gripper_center_quat_w, self.local_z_axis_b)
        object_z_w = quat_apply(self._snack_box.data.root_quat_w, self.local_z_axis_b)
        z_axis_dot = torch.sum(gripper_z_w * object_z_w, dim=-1).clamp(-1.0, 1.0)
        z_opposition = 0.5 * (1.0 - z_axis_dot)
        pose_proximity = self.cfg.pose_reward_far_scale + (1.0 - self.cfg.pose_reward_far_scale) * torch.exp(
            -self.cfg.pose_reward_proximity_k * distance
        )
        z_axis_align_reward = z_opposition * pose_proximity

        # reward when gripper local Y axis and object's short-side axis point in the same direction
        gripper_y_w = quat_apply(self.gripper_center_quat_w, self.local_y_axis_b)
        object_short_axis_w = quat_apply(self._snack_box.data.root_quat_w, self.object_short_axis_b)
        short_axis_dot = torch.sum(gripper_y_w * object_short_axis_w, dim=-1).clamp(-1.0, 1.0)
        short_axis_align = 0.5 * (short_axis_dot + 1.0)
        y_axis_align_reward = short_axis_align * pose_proximity

        # reward when gripper local X axis and object's long-side axis point in the same direction
        gripper_x_w = quat_apply(self.gripper_center_quat_w, self.local_x_axis_b)
        object_long_axis_w = quat_apply(self._snack_box.data.root_quat_w, self.object_long_axis_b)
        long_axis_dot = torch.sum(gripper_x_w * object_long_axis_w, dim=-1).clamp(-1.0, 1.0)
        long_axis_align = 0.5 * (long_axis_dot + 1.0)
        x_axis_align_reward = long_axis_align * pose_proximity

        # unified 3-axis alignment term (X-long, Y-short, Z-opposed)
        axis_align_raw = (z_opposition + short_axis_align + long_axis_align) / 3.0
        axis_align_reward = (z_axis_align_reward + y_axis_align_reward + x_axis_align_reward) / 3.0

        # approach reward is strengthened only when pose is aligned
        pose_align = axis_align_raw
        dist_pose_gate = self.cfg.dist_pose_gate_min + (1.0 - self.cfg.dist_pose_gate_min) * pose_align
        dist_reward = dist_reward_raw * dist_pose_gate

        # reward uses applied gripper command
        close_action = torch.clamp(self.cfg.gripper_close_action_sign * self._applied_gripper_action, min=0.0)
        open_action = torch.clamp(-self.cfg.gripper_close_action_sign * self._applied_gripper_action, min=0.0)

        # smooth grasp-ready gate: alignment + near distance
        gate_pose = torch.sigmoid(self.cfg.grip_gate_pose_k * (pose_align - self.cfg.grip_gate_pose_center))
        gate_dist = torch.sigmoid(self.cfg.grip_gate_dist_k * (self.cfg.grip_gate_dist_center - distance))
        grasp_ready_soft = gate_pose * gate_dist

        gripper_close_action_reward = grasp_ready_soft * close_action
        grasp_ready_reward = grasp_ready_soft

        grasp_ready_now = (grasp_ready_soft > self.cfg.grasp_enter_gate_threshold).float()

        # reward when gripper opening matches object's short-side width
        # pre-ready uses looser target (+margin), ready uses tighter target.
        gripper_readback_pos = self._robot.data.joint_pos[:, self.gripper_joint_ids]
        gripper_pos_mean = torch.mean(gripper_readback_pos, dim=-1)
        # target gripper position moves proportionally from open -> pre-grasp target
        # as pose alignment improves and remaining distance decreases
        distance_ready = torch.clamp(
            1.0 - distance / max(self.cfg.grip_gate_dist_center, 1.0e-6), min=0.0, max=1.0
        )
        align_ready = gate_pose * distance_ready
        gripper_target_pos = self.gripper_pos_open + align_ready * (self.gripper_target_pre_pos - self.gripper_pos_open)
        gripper_target_err = torch.abs(gripper_pos_mean - gripper_target_pos)
        target_tol = max(self.cfg.gripper_target_pos_tolerance, 1.0e-6)
        gripper_target_track_reward = align_ready * torch.exp(-torch.square(gripper_target_err / target_tol))
        target_err_delta = self._prev_gripper_target_err - gripper_target_err
        target_prog_norm = max(self.cfg.gripper_target_progress_norm, 1.0e-6)
        gripper_target_progress_reward = align_ready * torch.clamp(target_err_delta / target_prog_norm, min=-1.0, max=1.0)
        grip_close_ratio = torch.clamp((gripper_pos_mean - self.gripper_pos_open) / self.gripper_pos_span, 0.0, 1.0)
        gripper_width_m = self.cfg.gripper_open_width_m + grip_close_ratio * (
            self.cfg.gripper_close_width_m - self.cfg.gripper_open_width_m
        )
        width_target_pre_m = torch.full_like(gripper_width_m, self.gripper_width_target_pre_m)
        width_target_ready_m = torch.full_like(gripper_width_m, self.gripper_width_target_ready_m)
        width_target_m = torch.where(grasp_ready_now > 0.5, width_target_ready_m, width_target_pre_m)
        width_error_m = torch.abs(gripper_width_m - width_target_m)
        width_tol_m = max(self.cfg.gripper_width_match_tolerance_m, 1.0e-6)
        width_match_raw = torch.exp(-torch.square(width_error_m / width_tol_m))
        gripper_width_match_pre_reward = width_match_raw * (1.0 - grasp_ready_now)
        gripper_width_match_reward = width_match_raw * grasp_ready_now

        # dense reward from episode start: reward reduction in width error toward current target
        width_error_delta_m = self._prev_gripper_width_error_m - width_error_m
        width_progress_norm_m = max(self.cfg.gripper_width_progress_norm_m, 1.0e-6)
        gripper_width_progress_reward = torch.clamp(width_error_delta_m / width_progress_norm_m, min=-1.0, max=1.0)

        # keep gripper policy active: reward actual finger motion (fades out near width target)
        gripper_motion_m = torch.abs(gripper_pos_mean - self._prev_gripper_pos_mean)
        motion_norm_m = max(self.cfg.gripper_motion_reward_norm_m, 1.0e-6)
        gripper_motion_reward = torch.clamp(gripper_motion_m / motion_norm_m, min=0.0, max=1.0) * (1.0 - width_match_raw)

        self._prev_gripper_width_error_m = width_error_m.detach()
        self._prev_gripper_pos_mean = gripper_pos_mean.detach()
        self._prev_gripper_target_err = gripper_target_err.detach()

        grasp_enter_reward = torch.clamp(grasp_ready_now - self._prev_grasp_ready, min=0.0)
        self._prev_grasp_ready = grasp_ready_now.detach()

        # undesired collision penalty on non-gripper robot links
        # robot-object contact is excluded with an object-oriented bounding box check
        if self.collision_penalty_body_ids.numel() > 0:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, :, self.collision_penalty_body_ids]
            contact_force_mag = torch.norm(net_contact_forces, dim=-1)
            is_collision = torch.max(contact_force_mag, dim=1)[0] > self.cfg.collision_force_threshold

            penalty_body_pos_w = self._contact_sensor.data.pos_w[:, self.collision_penalty_body_ids]
            box_pos_w = self._snack_box.data.root_pos_w
            box_quat_w = self._snack_box.data.root_quat_w
            body_to_box_w = penalty_body_pos_w - box_pos_w.unsqueeze(1)

            num_penalty_bodies = self.collision_penalty_body_ids.numel()
            box_quat_repeat = box_quat_w.unsqueeze(1).expand(-1, num_penalty_bodies, -1).reshape(-1, 4)
            body_to_box_b = quat_apply_inverse(box_quat_repeat, body_to_box_w.reshape(-1, 3)).reshape(
                self.num_envs, num_penalty_bodies, 3
            )
            inside_object_obb = torch.all(
                torch.abs(body_to_box_b) <= (self.object_half_extents_b + self.cfg.object_collision_ignore_margin), dim=-1
            )

            is_collision = is_collision & (~inside_object_obb)
            collision_count = torch.sum(is_collision, dim=1).float()
        else:
            collision_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        arm_action_penalty = torch.sum(self.actions[:, : len(self.arm_joint_ids)] ** 2, dim=-1)
        arm_action_penalty_term = self.cfg.arm_action_penalty_scale * arm_action_penalty

        box_speed = torch.norm(box_lin_vel, dim=-1)
        gripper_set_pos = self.robot_dof_targets[:, self.gripper_joint_ids]
        gripper_abs_err = torch.abs(gripper_set_pos - gripper_readback_pos)

        rewards = (
            self.cfg.dist_reward_scale * dist_reward
            + self.cfg.lift_reward_scale * lift_amount
            + self.cfg.success_reward * success.float()
            + self.cfg.axis_align_reward_scale * axis_align_reward
            + self.cfg.gripper_close_action_reward_scale * gripper_close_action_reward
            + self.cfg.gripper_target_track_reward_scale * gripper_target_track_reward
            + self.cfg.gripper_target_progress_reward_scale * gripper_target_progress_reward
            + self.cfg.grasp_ready_reward_scale * grasp_ready_reward
            + self.cfg.grasp_enter_reward_scale * grasp_enter_reward
            + self.cfg.gripper_width_progress_reward_scale * gripper_width_progress_reward
            + self.cfg.gripper_motion_reward_scale * gripper_motion_reward
            + self.cfg.gripper_width_match_pre_reward_scale * gripper_width_match_pre_reward
            + self.cfg.gripper_width_match_reward_scale * gripper_width_match_reward
            + self.cfg.collision_penalty_scale * collision_count
            + arm_action_penalty_term
        )

        if "log" not in self.extras:
            self.extras["log"] = {}

        self.extras["log"].update(
            {
                # terminal priority order (KPI first)
                "KPI/01_success": success.float().mean(),
                "KPI/02_lift": (self.cfg.lift_reward_scale * lift_amount).mean(),
                "KPI/03_dist": (self.cfg.dist_reward_scale * dist_reward).mean(),
                "KPI/04_axis_align": (self.cfg.axis_align_reward_scale * axis_align_reward).mean(),
                "KPI/06_grip_close": (self.cfg.gripper_close_action_reward_scale * gripper_close_action_reward).mean(),
                "KPI/17_grip_tgt": (self.cfg.gripper_target_track_reward_scale * gripper_target_track_reward).mean(),
                "KPI/18_grip_prog": (
                    self.cfg.gripper_target_progress_reward_scale * gripper_target_progress_reward
                ).mean(),
                "KPI/12_grasp_ready": (self.cfg.grasp_ready_reward_scale * grasp_ready_reward).mean(),
                "KPI/13_grasp_enter": (self.cfg.grasp_enter_reward_scale * grasp_enter_reward).mean(),
                "KPI/07_collision": (self.cfg.collision_penalty_scale * collision_count).mean(),
                "KPI/08_act_arm": arm_action_penalty_term.mean(),
                "KPI/10_act_total": arm_action_penalty_term.mean(),
                "KPI/15_width_prog": (self.cfg.gripper_width_progress_reward_scale * gripper_width_progress_reward).mean(),
                "KPI/16_grip_motion": (self.cfg.gripper_motion_reward_scale * gripper_motion_reward).mean(),
                "KPI/14_width_fit_pre": (
                    self.cfg.gripper_width_match_pre_reward_scale * gripper_width_match_pre_reward
                ).mean(),
                "KPI/11_width_fit": (self.cfg.gripper_width_match_reward_scale * gripper_width_match_reward).mean(),
                # diagnostics
                "Diag/dist_raw": (self.cfg.dist_reward_scale * dist_reward_raw).mean(),
                "Diag/pose_gate": dist_pose_gate.mean(),
                "Diag/pose_align": pose_align.mean(),
                "Diag/pose_near": pose_proximity.mean(),
                "Diag/grasp_ready": grasp_ready_soft.mean(),
                "Diag/grasp_ready_hard": grasp_ready_now.mean(),
                "Diag/grip_close_act": close_action.mean(),
                "Diag/grip_open_act": open_action.mean(),
                "Diag/align_ready": align_ready.mean(),
                "Diag/grip_tgt_pos": gripper_target_pos.mean(),
                "Diag/grip_tgt_err": gripper_target_err.mean(),
                "Diag/grip_tgt_err_delta": target_err_delta.mean(),
                "Diag/grip_gate_pose": gate_pose.mean(),
                "Diag/grip_gate_dist": gate_dist.mean(),
                "Diag/x_dot": long_axis_dot.mean(),
                "Diag/x_deg": torch.rad2deg(torch.acos(long_axis_dot)).mean(),
                "Diag/z_dot": z_axis_dot.mean(),
                "Diag/z_deg": torch.rad2deg(torch.acos(z_axis_dot)).mean(),
                "Diag/y_dot": short_axis_dot.mean(),
                "Diag/y_deg": torch.rad2deg(torch.acos(short_axis_dot)).mean(),
                "Diag/coll_links": collision_count.mean(),
                "Diag/box_speed": box_speed.mean(),
                "Diag/box_yaw_deg": box_yaw_deg.mean(),
                "Diag/grip_x": self.gripper_center_pos_w[0, 0],
                "Diag/grip_y": self.gripper_center_pos_w[0, 1],
                "Diag/grip_z": self.gripper_center_pos_w[0, 2],
                "Diag/grip_width_m": gripper_width_m.mean(),
                "Diag/width_tgt_m": width_target_m.mean(),
                "Diag/width_tgt_pre_m": width_target_pre_m.mean(),
                "Diag/width_tgt_ready_m": width_target_ready_m.mean(),
                "Diag/width_err_mm": (width_error_m * 1000.0).mean(),
                "Diag/width_err_delta_mm": (width_error_delta_m * 1000.0).mean(),
                "Diag/width_fit_raw": width_match_raw.mean(),
                "Diag/width_ready": grasp_ready_soft.mean(),
                "Diag/grip_motion_mm": (gripper_motion_m * 1000.0).mean(),
                "Diag/grip_set_mean": gripper_set_pos.mean(),
                "Diag/grip_read_mean": gripper_readback_pos.mean(),
                # representative gripper position value (single scalar for quick monitoring)
                "Diag/grip_pos_rep": gripper_readback_pos.mean(),
                "Diag/grip_abs_err_mean": gripper_abs_err.mean(),
                "Diag/grip_set_rh_l1": gripper_set_pos[0, 0],
                "Diag/grip_set_rh_r1": gripper_set_pos[0, 1],
                "Diag/grip_set_rh_l2": gripper_set_pos[0, 2],
                "Diag/grip_set_rh_r2": gripper_set_pos[0, 3],
                "Diag/grip_read_rh_l1": gripper_readback_pos[0, 0],
                "Diag/grip_read_rh_r1": gripper_readback_pos[0, 1],
                "Diag/grip_read_rh_l2": gripper_readback_pos[0, 2],
                "Diag/grip_read_rh_r2": gripper_readback_pos[0, 3],
                # outlier references (min/max) for quick anomaly tracking
                "Ref/dist_min": distance.min(),
                "Ref/dist_max": distance.max(),
                "Ref/pose_align_min": pose_align.min(),
                "Ref/pose_align_max": pose_align.max(),
                "Ref/x_dot_min": long_axis_dot.min(),
                "Ref/z_dot_min": z_axis_dot.min(),
                "Ref/y_dot_min": short_axis_dot.min(),
                "Ref/grip_close_act_max": close_action.max(),
                "Ref/coll_links_max": collision_count.max(),
                "Ref/box_speed_max": box_speed.max(),
                "Ref/reward_min": rewards.min(),
                "Ref/reward_max": rewards.max(),
                "Ref/width_err_mm_max": (width_error_m * 1000.0).max(),
            }
        )

        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # 3,5축 90도 / 나머지 0도
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
        self._applied_gripper_action[env_ids] = 0.0
        self._prev_grasp_ready[env_ids] = 0.0
        self._reset_hold_steps_left[env_ids] = int(self.cfg.reset_hold_steps)

        reset_gripper_pos_mean = torch.mean(joint_pos[:, self.gripper_joint_ids], dim=-1)
        reset_grip_close_ratio = torch.clamp(
            (reset_gripper_pos_mean - self.gripper_pos_open) / self.gripper_pos_span, min=0.0, max=1.0
        )
        reset_gripper_width_m = self.cfg.gripper_open_width_m + reset_grip_close_ratio * (
            self.cfg.gripper_close_width_m - self.cfg.gripper_open_width_m
        )
        reset_width_error_m = torch.abs(reset_gripper_width_m - self.gripper_width_target_pre_m)
        self._prev_gripper_pos_mean[env_ids] = reset_gripper_pos_mean
        self._prev_gripper_width_error_m[env_ids] = reset_width_error_m
        self._prev_gripper_target_err[env_ids] = torch.abs(reset_gripper_pos_mean - self.gripper_pos_open)

        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        box_pose = self._snack_box.data.default_root_state[env_ids, :7].clone()
        box_pose[:, :3] += self.scene.env_origins[env_ids]
        box_vel = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device)

        self._snack_box.write_root_pose_to_sim(box_pose, env_ids=env_ids)
        self._snack_box.write_root_velocity_to_sim(box_vel, env_ids=env_ids)

    def _get_observations(self) -> dict:
        gripper_pos, box_pos, box_lin_vel, box_yaw_deg = self._compute_intermediate_values()
        self._update_debug_overlays()

        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        box_yaw_obs = box_yaw_deg / 180.0

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                gripper_pos,
                box_pos,
                box_pos - gripper_pos,
                box_lin_vel,
                box_yaw_obs,
            ),
            dim=-1,
        )

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _update_debug_overlays(self):
        if self._gripper_frame_marker is None or self._object_frame_marker is None:
            return

        # visualize env_0 to avoid clutter when running many envs
        self._gripper_frame_marker.visualize(
            translations=self.gripper_center_pos_w[:1],
            orientations=self.gripper_center_quat_w[:1],
        )
        self._object_frame_marker.visualize(
            translations=self._snack_box.data.root_pos_w[:1],
            orientations=self._snack_box.data.root_quat_w[:1],
        )

    def _compute_intermediate_values(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gripper_base_pos_w = self._robot.data.body_pos_w[:, self.gripper_base_body_idx]
        gripper_base_quat_w = self._robot.data.body_quat_w[:, self.gripper_base_body_idx]

        if self.use_finger_center:
            left_pos_w = self._robot.data.body_pos_w[:, self.gripper_left_body_idx]
            right_pos_w = self._robot.data.body_pos_w[:, self.gripper_right_body_idx]
            gripper_pos_w = 0.5 * (left_pos_w + right_pos_w)
        else:
            offset_b = self.gripper_center_offset_b.unsqueeze(0).expand(self.num_envs, -1)
            gripper_pos_w = gripper_base_pos_w + quat_apply(gripper_base_quat_w, offset_b)

        # cache world-frame pose for later data logging/debug
        self.gripper_center_pos_w = gripper_pos_w
        self.gripper_center_quat_w = gripper_base_quat_w

        box_pos_w = self._snack_box.data.root_pos_w
        box_quat_w = self._snack_box.data.root_quat_w
        box_lin_vel_w = self._snack_box.data.root_lin_vel_w
        _, _, box_yaw_rad = euler_xyz_from_quat(box_quat_w)
        box_yaw_deg = torch.rad2deg(box_yaw_rad).unsqueeze(-1)

        gripper_pos = gripper_pos_w - self.scene.env_origins
        box_pos = box_pos_w - self.scene.env_origins

        return gripper_pos, box_pos, box_lin_vel_w, box_yaw_deg
