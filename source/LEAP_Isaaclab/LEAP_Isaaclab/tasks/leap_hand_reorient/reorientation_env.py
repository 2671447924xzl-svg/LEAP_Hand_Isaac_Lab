# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py
# --------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import sys
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, euler_xyz_from_quat, quat_from_euler_xyz
import time
if TYPE_CHECKING:
    from LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_env_cfg import LeapHandEnvCfg

from LEAP_Isaaclab.utils import adr_utils, obs_utils
from LEAP_Isaaclab.utils.adr import LeapHandADR

class ReorientationEnv(DirectRLEnv):
    cfg: LeapHandEnvCfg

    def __init__(self, cfg: LeapHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = [self.hand.joint_names.index(j) for j in self.cfg.actuated_joint_names]
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)
        
        # joint limits
        joint_pos_limits = self.hand.data.joint_pos_limits.torch.to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = self.object.data.default_root_pose.torch[:, 0:3].clone()
        self.in_hand_pos[:, 2] += 0.01
        
        # continuous z-axis rotation parameters
        self.target_z_angle = torch.full((self.num_envs,), 2 * math.pi / self.cfg.z_rotation_steps, dtype=torch.float, device=self.device)
        
        # default goal positions and rotations
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 3] = 1.0  # Identity quaternion
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successe
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.override_default_joint_pos = torch.tensor([[0.000, 0.500, 0.000, 0.000, 
                                                        -0.750, 1.300, 0.000, 0.750, 
                                                         1.750, 1.500, 1.750, 1.750, 
                                                         0.000, 1.000, 0.000, 0.000]], device=self.device).repeat(self.num_envs, 1)

        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_rot[:, 3] = 1.0 

        # initialize history tensor
        self.obs_hist_buf = torch.zeros((self.num_envs, self.cfg.observation_space // self.cfg.hist_len, self.cfg.hist_len), device=self.device, dtype=torch.double)            
        self.output_obs_hist_buf = torch.zeros(self.cfg.scene.num_envs, self.cfg.observation_space // self.cfg.hist_len, self.cfg.hist_len, device=self.cfg.sim.device, dtype=torch.double)
            
        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Isaac Lab 3.0 indexed write APIs
        self._set_joint_pos_target = self.hand.set_joint_position_target_index
        self._write_obj_root_pose = self.object.write_root_pose_to_sim_index
        self._write_obj_root_vel = self.object.write_root_velocity_to_sim_index
        self._write_hand_joint_pos = self.hand.write_joint_position_to_sim_index
        self._write_hand_joint_vel = self.hand.write_joint_velocity_to_sim_index

        self.randomized_episode_lengths = torch.randint(int(self.cfg.min_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)), self.max_episode_length + 1, (self.num_envs,), dtype=torch.int32, device=self.device)

        # set adr up
        if self.cfg.enable_adr:
            self.leap_adr = LeapHandADR(self.event_manager, 
                                                    self.cfg.adr_cfg_dict, 
                                                    self.cfg.adr_custom_cfg_dict)
            self.step_since_last_dr_change = 0
            self.leap_adr.set_num_increments(self.cfg.starting_adr_increments)
            adr_utils.init_adr_obs_act_noise(self)

            self.obs_hist_buf = torch.zeros(self.num_envs, self.cfg.observation_space // self.cfg.hist_len, self.cfg.hist_len + self.cfg.obs_max_latency, device=cfg.sim.device, dtype=torch.float)
            self.obs_latency = torch.empty((self.num_envs, self.cfg.obs_per_timestep), device =self.cfg.sim.device)
            self.act_latency = torch.empty((self.num_envs, self.cfg.action_space), device =self.cfg.sim.device)
            self.act_hist_buf = torch.zeros(self.num_envs, self.cfg.action_space, self.cfg.act_max_latency + 1, device=self.cfg.sim.device, dtype=torch.float)

            print("starting ranges: ")
            print(self.leap_adr.print_params())
        
        # Initialize extras if not already present
        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

        self.sim_real_indices()

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        if self.cfg.enable_adr:
            hand_noise = self.leap_adr.get_custom_param_value("robot_action_noise", "hand_noise")
            if hand_noise > 0:
                noise = torch.randn_like(actions) * hand_noise
                self.actions = actions + noise
            self.actions = obs_utils.create_action_latency(self, self.actions)

        self.actions = torch.clamp(self.actions, -1.0, 1.0)

    def _apply_action(self) -> None:

        if self.cfg.action_type=="relative":
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.cfg.act_moving_average * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                targets,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        elif self.cfg.action_type=="absolute":
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                self.cur_targets[:, self.actuated_dof_indices],
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        else:
            raise ValueError(f"Unsupported action type: {self.cfg.action_type}. Must be relative or absolute.")

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        if self.cfg.enable_adr:
            adr_utils.apply_object_wrench(self, self.object, "object")

        self._set_joint_pos_target(
            target=self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    def _update_continuous_z_rotation(self, goal_env_ids):        
        # create quaternion for z-axis rotation
        add_rot = quat_from_angle_axis(self.target_z_angle, self.z_unit_tensor)
        self.goal_rot[goal_env_ids] = quat_mul(add_rot[goal_env_ids], self.goal_rot[goal_env_ids])
        
        # update goal markers
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

    def _get_observations(self) -> dict:
        frame = unscale(self.hand_dof_pos,
                    self.hand_dof_lower_limits,
                    self.hand_dof_upper_limits)  
        if self.cfg.store_cur_actions:
            frame = torch.cat((frame, self.cur_targets[:]), dim=-1)  

        self.obs_hist_buf[:, :, :-1] = self.obs_hist_buf[:, :, 1:]
        self.obs_hist_buf[:, :, -1] = frame    
        obs = self.obs_hist_buf.transpose(1, 2).reshape(self.num_envs, -1)   
        return {"policy": obs.float()}

    def _get_rewards(self) -> torch.Tensor:

        pose_diff_penalty = ((self.cur_targets[:, self.actuated_dof_indices] - self.override_default_joint_pos) ** 2).sum(-1)
        torque_penalty = (self.hand.data.computed_torque.torch ** 2).sum(-1)

        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.fingertip_pos,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.object_linvel,
            self.object_angvel,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            pose_diff_penalty, 
            self.cfg.pose_diff_penalty_scale,
            torque_penalty,
            self.cfg.torque_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean() / self.cfg.z_rotation_steps
        self.extras["log"]["pose_diff_penalty"] = pose_diff_penalty.mean() 
        self.extras["log"]["torque_info"] = torque_penalty.mean() 
        self.extras["log"]['object_linvel'] = torch.norm(self.object_linvel, p=1, dim=-1).mean()
        self.extras["log"]['roll'] = self.object_angvel[:, 0].mean()
        self.extras["log"]['pitch'] = self.object_angvel[:, 1].mean()
        self.extras["log"]['yaw'] = self.object_angvel[:, 2].mean()

        # Log episode length statistics
        self.extras["log"]["avg_episode_length_s"] = (self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation).mean()
        self.extras["log"]["min_episode_length_s"] = (self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation).min()
        self.extras["log"]["max_episode_length_s"] = (self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation).max()

        if self.cfg.enable_adr:
            adr_criteria = ((self.consecutive_successes / self.cfg.z_rotation_steps) / (self.randomized_episode_lengths.float().mean() * self.cfg.sim.dt * self.cfg.decimation)).float().mean()
            self.extras["log"]["adr_criteria"] = adr_criteria

        # update goal if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._update_continuous_z_rotation(goal_env_ids)
            self.reset_goal_buf[goal_env_ids] = 0

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist
        time_out = self.episode_length_buf >= self.randomized_episode_lengths - 1

        #get z axis difference to terminate episode if a cube is flipped or not
        obj_z = matrix_from_quat(self.object_rot)[:, :, 2]
        goal_z = matrix_from_quat(self.goal_rot)[:, :, 2]
        diff = torch.sum(obj_z * goal_z, dim=1)
        flipped = (torch.abs(diff) < 0.5)

        out_of_reach = out_of_reach | flipped

        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if self.cfg.enable_adr:
            adr_criteria = ((self.consecutive_successes.float().mean() / self.cfg.z_rotation_steps) / (self.randomized_episode_lengths.float().mean() * self.cfg.sim.dt * self.cfg.decimation)).float().mean()

        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        self.randomized_episode_lengths[env_ids] = torch.randint(
            int(self.cfg.min_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)), 
            self.max_episode_length + 1, 
            (len(env_ids),), 
            dtype=torch.int32, 
            device=self.device
        )

        # reset object
        object_default_pose = self.object.data.default_root_pose.torch.clone()[env_ids]
        object_default_vel = self.object.data.default_root_vel.torch.clone()[env_ids]

        dof_pos = self.override_default_joint_pos[env_ids].clone()
        dof_vel = self.hand.data.default_joint_vel.torch[env_ids].clone()

        object_default_pose[:, 0:3] += self.scene.env_origins[env_ids]
        object_default_vel[:] = 0.0

        if self.cfg.enable_adr:
            x_width = self.leap_adr.get_custom_param_value("object_spawn", "x_width_spawn")
            y_width = self.leap_adr.get_custom_param_value("object_spawn", "y_width_spawn")
            x_rot = self.leap_adr.get_custom_param_value("object_spawn", "x_rotation")
            y_rot = self.leap_adr.get_custom_param_value("object_spawn", "y_rotation")
            z_rot = self.leap_adr.get_custom_param_value("object_spawn", "z_rotation")
            
            # Apply randomization
            if x_width > 0 or y_width > 0:
                pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
                object_default_pose[:, 0] += pos_noise[:, 0] * x_width
                object_default_pose[:, 1] += pos_noise[:, 1] * y_width
            
            if x_rot > 0:
                x_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
                x_rot_quat = quat_from_angle_axis(x_rot_noise * x_rot, self.x_unit_tensor[env_ids])
                object_default_pose[:, 3:7] = quat_mul(x_rot_quat, object_default_pose[:, 3:7])
                
            if y_rot > 0:
                y_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
                y_rot_quat = quat_from_angle_axis(y_rot_noise * y_rot, self.y_unit_tensor[env_ids])
                object_default_pose[:, 3:7] = quat_mul(y_rot_quat, object_default_pose[:, 3:7])
                
            if z_rot > 0:
                z_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
                z_rot_quat = quat_from_angle_axis(z_rot_noise * z_rot, self.z_unit_tensor[env_ids])
                object_default_pose[:, 3:7] = quat_mul(z_rot_quat, object_default_pose[:, 3:7])

            joint_pos_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_pos_noise")
            joint_vel_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_vel_noise")

            if joint_pos_noise_width > 0:
                joint_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
                dof_pos += joint_pos_noise * joint_pos_noise_width
                
            if joint_vel_noise_width > 0:
                joint_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
                dof_vel += joint_vel_noise * joint_vel_noise_width

        self._write_obj_root_pose(
            root_pose=object_default_pose,
            env_ids=env_ids,
        )        
        self._write_obj_root_vel(
            root_velocity=object_default_vel,
            env_ids=env_ids,
        )

        # reset hand
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos
        self.successes[env_ids] = 0

        self._set_joint_pos_target(
            target=dof_pos,
            env_ids=env_ids,
        )
        
        self._write_hand_joint_pos(
            position=dof_pos,
            env_ids=env_ids,
        )
        
        self._write_hand_joint_vel(
            velocity=dof_vel,
            env_ids=env_ids,
        )        

        if self.cfg.enable_adr and len(env_ids) > 0:
            adr_utils.update_adr_obs_act_noise(self, env_ids)

            obs_latency_resets =  self.leap_adr.get_custom_param_value("obs_latency","latency") - torch.randint(0, self.cfg.obs_latency_rand + 1, (len(env_ids),1), device=self.cfg.sim.device)
            obs_latency_resets = torch.maximum(obs_latency_resets, torch.tensor(0))
            self.obs_latency[env_ids, :] = obs_latency_resets.expand(-1, self.cfg.obs_per_timestep)
            
            act_latency_resets = self.leap_adr.get_custom_param_value("action_latency","hand_latency") - torch.randint(0, self.cfg.act_latency_rand + 1, (len(env_ids), 1), device=self.cfg.sim.device)
            act_latency_resets = torch.maximum(act_latency_resets, torch.tensor(0))
            self.act_latency[env_ids, :] = act_latency_resets.expand(-1, self.cfg.action_space)
            
            self.extras["log"]["num_adr_increases"] = self.leap_adr.num_increments()
            
            if self.step_since_last_dr_change >= self.cfg.min_steps_for_dr_change and\
                (adr_criteria  >= self.cfg.min_rot_adr_coeff):
                self.step_since_last_dr_change = 0
                self.leap_adr.increase_ranges()
                self.leap_adr.print_params()
                self.consecutive_successes.fill_(0.0)
            else:
                self.step_since_last_dr_change += 1

            # update whether to apply wrench for the episode
            self.object_mass = self.object.data.body_mass.torch.clone() 
            self.apply_wrench = torch.where(
                torch.rand(self.num_envs, device=self.device) <= self.cfg.wrench_prob_per_rollout,
                True,
                False)

        # initialize goal rotation
        self._compute_intermediate_values()
        r,p,y = euler_xyz_from_quat(self.object_rot[env_ids])
        r[:].fill_(0.0)
        p[:].fill_(0.0)
        self.goal_rot[env_ids] = quat_from_euler_xyz(r,p,y)

        self._update_continuous_z_rotation(env_ids)

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w.torch[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w.torch[:, self.finger_bodies]
        
        self.fingertip_pos -= self.scene.env_origins.repeat(
            (1, self.num_fingertips)
        ).reshape(
            self.num_envs,
            self.num_fingertips,
            3,
        )
        
        self.fingertip_velocities = self.hand.data.body_vel_w.torch[:, self.finger_bodies]
        
        self.hand_dof_pos = self.hand.data.joint_pos.torch
        self.hand_dof_vel = self.hand.data.joint_vel.torch
        
        self.object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w.torch
        self.object_velocities = self.object.data.root_vel_w.torch
        self.object_linvel = self.object.data.root_lin_vel_w.torch
        self.object_angvel = self.object.data.root_ang_vel_w.torch 
            
    def sim_real_indices(self):
        sim2real_idx_16, _ = self.hand.find_joints(self.cfg.actuated_joint_names, preserve_order=True)
        sim2real_idx_16 = torch.tensor(sim2real_idx_16) - min(sim2real_idx_16)
        real2sim_idx_16 = torch.empty_like(sim2real_idx_16)
        real2sim_idx_16[sim2real_idx_16] = torch.arange(len(sim2real_idx_16))

        print(f"sim2real_indices: {sim2real_idx_16}")
        print(f"real2sim_indices: {real2sim_idx_16}")
            
@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))  # changed quat convention

@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    object_linvel: torch.Tensor,
    object_angvel: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    pose_diff_penalty: torch.Tensor,
    pose_diff_penalty_scale: float,
    torque_penalty: torch.Tensor,
    torque_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):

    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)
    pose_diff_penalty = pose_diff_penalty * pose_diff_penalty_scale
    fingertip_dist_penalty = torch.norm(fingertip_pos - object_pos.unsqueeze(1), p=2, dim=-1)
    fingertip_dist_penalty = torch.mean(fingertip_dist_penalty, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty + pose penalty + torque penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale + pose_diff_penalty  + torque_penalty * torque_penalty_scale 

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where((torch.abs(rot_dist) <= success_tolerance) & (goal_dist <= 0.025), torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Stability reward: cube is not spinning too fast to the point where joints are messed up
    reward = torch.where((object_angvel[:, 2] > 0.25) & (object_angvel[:, 2] < 1.5), reward + 1, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes