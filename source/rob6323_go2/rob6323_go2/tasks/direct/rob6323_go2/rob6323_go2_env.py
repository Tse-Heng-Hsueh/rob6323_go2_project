# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import gymnasium as gym
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    """
    RL Environment for training Unitree Go2 quadruped robot locomotion.

    This environment trains the robot to:
    - Follow velocity commands (forward, lateral, and yaw rotation)
    - Maintain stable walking gait
    - Avoid falling or base contact with ground
    """

    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the environment and all state variables.

        This method sets up:
        - Action buffers (current and previous actions)
        - Velocity command buffers
        - Logging dictionaries for tracking rewards
        - Body part indices for contact detection
        - Debug visualization markers
        """
        super().__init__(cfg, render_mode, **kwargs)

        # --- Action Buffers ---
        # Current action from the policy (12D: 3 joints per leg × 4 legs)
        # These are OFFSETS from default standing pose, not absolute joint angles
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Previous action - stored in observation space so policy knows what it did last step
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # --- PD Controller Parameters (Part 2: IMPLEMENTED) ---
        # Manual implementation of low-level PD controller for torque control
        # Instead of using the physics engine's built-in PD controller,
        # we compute torques explicitly for better control and debugging.

        # Kp (Proportional gain): Shape (num_envs, 12)
        # Controls how strongly joints push toward target position
        # Higher Kp → stiffer joints, faster response (but may oscillate)
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Kd (Derivative gain): Shape (num_envs, 12)
        # Provides damping to reduce oscillations and overshoot
        # Higher Kd → more damping, smoother motion (but slower response)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Motor offsets: Not used in baseline, but reserved for calibration
        self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)

        # Torque limits: Maximum torque per joint (safety constraint)
        # Prevents unrealistic forces and simulation instability
        self.torque_limits = cfg.torque_limits

        # Torque buffer: Store applied torques for reward calculation
        # TODO: Re-enable after baseline validation
        # self._torques = torch.zeros(self.num_envs, 12, device=self.device)

        # --- Velocity Commands ---
        # Shape: (num_envs, 3) where 3 = [vx, vy, yaw_rate]
        # vx: forward/backward velocity, vy: left/right velocity, yaw_rate: rotation speed
        # Commands are randomly sampled in range [-1, 1] at each reset
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # --- Reward Logging ---
        # Track cumulative rewards for each component across an episode
        # Used for tensorboard logging and analysis
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",  # Reward for tracking XY velocity commands
                "track_ang_vel_z_exp",  # Reward for tracking yaw rate command
                "rew_action_rate",  # Penalty for jerky/sudden actions (Part 1: IMPLEMENTED)
                "raibert_heuristic",  # Reward for good gait patterns (Part 4: IMPLEMENTED)
                "rew_torque",  # Penalty for high torque usage (course requirement)
                "orient",  # Penalty for body tilt (Part 5: IMPLEMENTED)
                "lin_vel_z",  # Penalty for vertical velocity (Part 5: IMPLEMENTED)
                "dof_vel",  # Penalty for high joint velocities (Part 5: IMPLEMENTED)
                "ang_vel_xy",  # Penalty for body roll/pitch (Part 5: IMPLEMENTED)
                "feet_clearance",  # Penalty for not lifting feet during swing (Part 6: IMPLEMENTED)
                "tracking_contacts_shaped_force",  # Reward for proper contact forces (Part 6: IMPLEMENTED)
                # Debug metrics for Part 6 (output raw values to tensorboard)
                "debug_contact_force_magnitude",  # Raw foot contact force (should be >0 when touching ground)
                "debug_desired_contact_states",  # Should oscillate between 0 and 1 based on gait phase
                "debug_is_in_contact",  # Binary: 1 if force > threshold, 0 otherwise
                "debug_all_body_forces",  # Sum of ALL body forces (verify sensor works at all)
            ]
        }

        # --- Action History for Smoothness Penalty ---
        # Store last 3 actions to compute action rate (how much actions change between steps)
        # Shape: (num_envs, 12, 3) where 3 = history length
        # Used to penalize jerky movements and encourage smooth control
        self.last_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # --- Gait State Variables (Part 4: IMPLEMENTED) ---
        # Track gait phase for Raibert Heuristic foot placement guidance

        # Get foot body indices for position tracking
        self._feet_ids = []
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for name in foot_names:
            id_list, _ = self.robot.find_bodies(name)
            self._feet_ids.append(id_list[0])

        # Gait clock: Tracks current phase of walking cycle [0, 1)
        # Increments each step based on gait frequency (3 Hz)
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Clock inputs: Sine wave representation of each foot's gait phase
        # Shape: (num_envs, 4) for 4 feet
        # Added to observation space so policy knows when each foot should lift/land
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # Desired contact states: Smooth transition between stance and swing phases
        # Shape: (num_envs, 4) for 4 feet
        # Values in [0, 1]: 1 = foot should be on ground, 0 = foot should be in air
        self.desired_contact_states = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Part 6: Advanced Foot Interaction Rewards
        # Find feet indices in the CONTACT SENSOR
        # Note: _feet_ids for kinematics is already defined above (line 124-128)
        # Use exact names (same as robot) because regex ".*_foot" may not match
        self._feet_ids_sensor = []
        for name in foot_names:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(id_list[0])

        # Initialize debug variables for tensorboard logging
        self._debug_contact_force_magnitude = torch.zeros(self.num_envs, device=self.device)
        self._debug_desired_contact_states = torch.zeros(self.num_envs, device=self.device)
        self._debug_is_in_contact = torch.zeros(self.num_envs, device=self.device)
        # Additional debug: track all body forces to verify sensor is working
        self._debug_all_body_forces = torch.zeros(self.num_envs, device=self.device)
        self._debug_num_feet_ids_sensor = len(self._feet_ids_sensor)  # Should be 4

        # --- Body Part Indices ---
        # Get indices of specific body parts for contact detection
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        # Robot's main body/torso
        # All four feet (to be used later)
        # self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        # self._undesired_contact_body_ids, _ =
        # self._contact_sensor.find_bodies(".*thigh")
        # Parts that shouldn't touch ground

        # --- Debug Visualization ---
        # Setup velocity visualization arrows (green=target, blue=actual)
        self.set_debug_vis(self.cfg.debug_vis)

        # --- TensorBoard Text Logging ---
        # Flag to ensure we only write config once
        self._config_logged = False

    def _log_experiment_config_to_tensorboard(self):
        """
        WARNING: TEMPORARY DEBUG CODE - REMOVE AFTER DEBUGGING PHASE

        Log experiment configuration to TensorBoard for tracking parameter changes
        across training runs. This is a temporary debugging feature to diagnose
        performance regressions and should be removed once debugging is complete.

        TODO: Remove this function after identifying and fixing performance issues.
        """
        if self._config_logged:
            return

        # Build clean configuration summary (no emojis, no subjective judgments)
        config_text = "ROB6323 Go2 Training Configuration\n"
        config_text += "=" * 80 + "\n\n"

        # PD Controller Parameters
        config_text += "PD CONTROLLER\n"
        config_text += "-" * 40 + "\n"
        config_text += f"  Kp: {self.cfg.Kp}\n"
        config_text += f"  Kd: {self.cfg.Kd}\n"
        config_text += f"  Torque Limit: {self.cfg.torque_limits} Nm\n\n"

        # Reward Weights
        config_text += "REWARD WEIGHTS\n"
        config_text += "-" * 40 + "\n"
        config_text += f"  lin_vel_reward_scale: {self.cfg.lin_vel_reward_scale}\n"
        config_text += f"  yaw_rate_reward_scale: {self.cfg.yaw_rate_reward_scale}\n"
        config_text += f"  action_rate_reward_scale: {self.cfg.action_rate_reward_scale}\n"
        config_text += f"  raibert_heuristic_reward_scale: {self.cfg.raibert_heuristic_reward_scale}\n"
        config_text += f"  orient_reward_scale: {self.cfg.orient_reward_scale}\n"
        config_text += f"  lin_vel_z_reward_scale: {self.cfg.lin_vel_z_reward_scale}\n"
        config_text += f"  dof_vel_reward_scale: {self.cfg.dof_vel_reward_scale}\n"
        config_text += f"  ang_vel_xy_reward_scale: {self.cfg.ang_vel_xy_reward_scale}\n"
        config_text += f"  feet_clearance_reward_scale: {self.cfg.feet_clearance_reward_scale}\n"
        config_text += "  tracking_contacts_shaped_force_reward_scale: "
        config_text += f"{self.cfg.tracking_contacts_shaped_force_reward_scale}\n\n"

        # Termination Conditions
        config_text += "TERMINATION CONDITIONS\n"
        config_text += "-" * 40 + "\n"
        config_text += f"  base_height_min: {self.cfg.base_height_min} m\n"
        config_text += f"  max_episode_length: {self.max_episode_length} steps\n\n"

        # Environment Settings
        config_text += "ENVIRONMENT\n"
        config_text += "-" * 40 + "\n"
        config_text += f"  num_envs: {self.num_envs}\n"
        config_text += f"  control_frequency: {1.0 / self.step_dt:.0f} Hz\n"
        config_text += f"  observation_space: {self.cfg.observation_space}D\n"
        config_text += f"  action_space: {self.cfg.action_space}D\n"
        config_text += f"  action_scale: {self.cfg.action_scale}\n\n"

        # Implementation Notes
        config_text += "NOTES\n"
        config_text += "-" * 40 + "\n"
        config_text += "  Part 6 reimplemented using IsaacGymEnvs logic with IsaacLab API\n"
        config_text += "  Dynamic target height: 0.02-0.10m based on gait phase\n"
        config_text += "  Contact penalty: Single-sided (F^2/100 shaping)\n"
        config_text += "=" * 80 + "\n"

        # Try to log to TensorBoard if writer is available
        try:
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.add_text("Config/Experiment_Setup", config_text, 0)
                print("[INFO] Experiment configuration logged to TensorBoard")
        except Exception:
            # Silently fail if TensorBoard writer is not available
            pass

        self._config_logged = True

    def log_final_metrics_summary(self, iteration: int):
        """
        WARNING: TEMPORARY DEBUG CODE - REMOVE AFTER DEBUGGING PHASE

        Log training metrics summary to TensorBoard (optional, not called automatically).
        This is a temporary debugging feature for manual inspection.

        TODO: Remove this function after debugging phase is complete.

        Args:
            iteration: Current training iteration number
        """
        # Build clean metrics summary (no emojis, no subjective judgments)
        summary_text = f"Training Metrics Summary (Iteration {iteration})\n"
        summary_text += "=" * 80 + "\n\n"

        # Get metrics from episode_sums (averaged in extras["log"])
        if hasattr(self, "extras") and "log" in self.extras:
            metrics = self.extras["log"]

            summary_text += "PRIMARY TRACKING\n"
            summary_text += "-" * 40 + "\n"
            if "Episode_Reward/track_lin_vel_xy_exp" in metrics:
                val = metrics["Episode_Reward/track_lin_vel_xy_exp"]
                summary_text += f"  track_lin_vel_xy_exp: {val:.2f}\n"
            if "Episode_Reward/track_ang_vel_z_exp" in metrics:
                val = metrics["Episode_Reward/track_ang_vel_z_exp"]
                summary_text += f"  track_ang_vel_z_exp: {val:.2f}\n"

            summary_text += "\nGAIT QUALITY\n"
            summary_text += "-" * 40 + "\n"
            if "Episode_Reward/rew_action_rate" in metrics:
                summary_text += f"  rew_action_rate: {metrics['Episode_Reward/rew_action_rate']:.2f}\n"
            if "Episode_Reward/raibert_heuristic" in metrics:
                summary_text += f"  raibert_heuristic: {metrics['Episode_Reward/raibert_heuristic']:.2f}\n"
            if "Episode_Reward/feet_clearance" in metrics:
                summary_text += f"  feet_clearance: {metrics['Episode_Reward/feet_clearance']:.2f}\n"
            if "Episode_Reward/tracking_contacts_shaped_force" in metrics:
                val = metrics["Episode_Reward/tracking_contacts_shaped_force"]
                summary_text += f"  tracking_contacts_shaped_force: {val:.2f}\n"

            summary_text += "\nSTABILITY\n"
            summary_text += "-" * 40 + "\n"
            if "Episode_Termination/base_contact" in metrics:
                val = metrics["Episode_Termination/base_contact"]
                summary_text += f"  base_contact_terminations: {val:.0f}\n"
            for key in ["orient", "lin_vel_z", "ang_vel_xy"]:
                metric_key = f"Episode_Reward/{key}"
                if metric_key in metrics:
                    summary_text += f"  {key}: {metrics[metric_key]:.4f}\n"

        summary_text += "=" * 80 + "\n"

        # Log to TensorBoard if available
        try:
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.add_text("Summary/Final_Metrics", summary_text, iteration)
                print("[INFO] Final metrics summary logged to TensorBoard")
        except Exception:
            pass

    def _setup_scene(self):
        """
        Build the simulation scene before training starts.

        This method creates:
        - Robot articulation (joints, links, actuators)
        - Contact sensors for detecting ground contact
        - Terrain (flat plane in baseline, can be changed to rough terrain later)
        - Multiple parallel environments (4096 by default)
        - Lighting for visualization

        Called once at the beginning of training.
        """
        # Create robot with all its joints and sensors
        self.robot = Articulation(self.cfg.robot_cfg)

        # Create contact sensor to detect which body parts touch ground
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # CRITICAL: Register sensor in scene so it gets updated each step!
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Create terrain (flat ground plane in baseline)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone the scene to create 4096 parallel environments for faster training
        self.scene.clone_environments(copy_from_source=False)

        # Handle CPU-specific collision filtering
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register robot in the scene
        self.scene.articulations["robot"] = self.robot

        # Add lighting for visualization
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Process actions from the policy before physics simulation runs.

        This converts normalized actions (typically in [-1, 1]) into desired joint positions:
        - Scale actions by action_scale (0.25)
        - Add to default joint positions (standing pose)

        Example: If action = 0.5 for a joint with default position = 0.0
                 Result = 0.25 * 0.5 + 0.0 = 0.125 radians offset

        Args:
            actions: Raw actions from policy, shape (num_envs, 12)
        """
        # Store raw actions for observation and action rate calculation
        self._actions = actions.clone()

        # Compute desired joint positions from policy actions
        # Formula: q_desired = action_scale * action + q_default
        self.desired_joint_pos = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        """
        Apply torque control using manual PD controller (Part 2: IMPLEMENTED).

        This computes joint torques using the classic PD control formula:
            τ = Kp * (q_desired - q_actual) - Kd * q̇_actual

        Where:
        - Kp (proportional gain): Controls how strongly we push toward target position
        - Kd (derivative gain): Provides damping to prevent oscillations
        - q_desired: Target joint positions from _pre_physics_step()
        - q_actual: Current joint positions from simulation
        - q̇_actual: Current joint velocities from simulation

        The computed torques are clipped to [−torque_limits, +torque_limits] for safety.

        Why manual PD control instead of built-in?
        - Full control over gains (Kp, Kd) for fine-tuning
        - Can add advanced features later (friction compensation, gravity compensation)
        - Explicit torque limits prevent unrealistic forces
        """
        # Compute PD torques using the standard formula
        # Proportional term: Kp * position_error
        # Derivative term: -Kd * velocity (acts as damping)
        torques = torch.clip(
            (self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel),
            -self.torque_limits,
            self.torque_limits,
        )

        # Store torques for reward calculation (used by torque penalty)
        self._torques = torques

        # Send computed torques to robot actuators
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        """
        Build the observation vector that the policy network sees.

        The observation contains 52 values (all in robot's body frame):
        - [0:3]   Base linear velocity (vx, vy, vz)
        - [3:6]   Base angular velocity (ωx, ωy, ωz)
        - [6:9]   Projected gravity vector (tells robot its tilt)
        - [9:12]  Velocity commands (target vx, vy, yaw_rate)
        - [12:24] Joint positions (offset from default standing pose)
        - [24:36] Joint velocities
        - [36:48] Previous actions (so policy knows what it just did)
        - [48:52] Clock inputs (gait phase for each of 4 feet) [Part 4: ADDED]

        Returns:
            Dictionary with key "policy" containing the 52D observation tensor
        """
        # Save current action as previous for next step
        self._previous_actions = self._actions.clone()

        # Concatenate all observation components into one vector
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,  # Base velocity in body frame
                    self.robot.data.root_ang_vel_b,  # Base angular velocity
                    self.robot.data.projected_gravity_b,  # Gravity direction (for tilt sensing)
                    self._commands,  # Velocity targets
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # Joint offsets
                    self.robot.data.joint_vel,  # Joint velocities
                    self._actions,  # Previous action
                    self.clock_inputs,  # Add gait phase info
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute rewards for the current step - this shapes what the robot learns!

        Current reward terms (Parts 1-4):
        1. Linear velocity tracking: Rewards robot for matching target vx, vy speed
        2. Yaw rate tracking: Rewards robot for matching target rotation speed
        3. Action rate penalty: Penalizes jerky/sudden action changes (Part 1)
        4. Raibert heuristic penalty: Guides proper foot placement for stable gait (Part 4)

        Reward formula uses exponential mapping for tracking:
        - reward = exp(-error / temperature)
        - When error = 0 → reward = 1.0 (perfect)
        - When error is large → reward ≈ 0 (poor tracking)
        - temperature = 0.25 controls how steep the curve is

        TODO (from tutorial): Add more rewards like:
        - Orientation/height penalties (Part 5)
        - Contact force penalties (Part 6)

        Returns:
            Total reward for each environment, shape (num_envs,)
        """
        # --- Linear Velocity Tracking Reward ---
        # Compare commanded XY velocity vs actual XY velocity
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        # Map error to [0, 1] range using exponential function
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        # --- Yaw Rate Tracking Reward ---
        # Compare commanded yaw rate vs actual yaw rate
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # --- Action Rate Penalization (Part 1) ---
        # Penalize high-frequency action changes to encourage smooth control.
        # This prevents jerky movements and makes the robot's gait more natural.

        # First derivative: Penalize large changes in action (velocity of action change)
        # ||a(t) - a(t-1)||²
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (
            self.cfg.action_scale**2
        )

        # Second derivative: Penalize sudden acceleration in actions (acceleration of action change)
        # ||a(t) - 2*a(t-1) + a(t-2)||²
        # This catches oscillations that first derivative alone might miss
        rew_action_rate += torch.sum(
            torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1
        ) * (self.cfg.action_scale**2)

        # Update the action history buffer (roll and insert new action)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        # --- Raibert Heuristic Reward (Part 4: IMPLEMENTED) ---
        # Update gait clock and calculate ideal foot placements
        self._step_contact_targets()
        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # --- Part 5: Additional Regularization Rewards ---

        # 1. Penalize non-vertical orientation (projected gravity on XY plane)
        # We want the robot to stay upright, so gravity should only project onto Z.
        # Calculate the sum of squares of the X and Y components of projected_gravity_b.
        rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)

        # 2. Penalize vertical velocity (z-component of base linear velocity)
        # Square the Z component of the base linear velocity to reduce bouncing.
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])

        # 3. Penalize high joint velocities
        # Sum the squares of all joint velocities to encourage smooth, natural motion.
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)

        # 4. Penalize angular velocity in XY plane (roll/pitch)
        # Sum the squares of the X and Y components of the base angular velocity
        # to minimize body rocking/swaying.
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

        # 5. Penalize high torque usage (energy efficiency)
        # This encourages the robot to use minimal force for smooth, energy-efficient motion.
        # Formula: sum of squared torques across all 12 joints
        rew_torque = torch.sum(torch.square(self._torques), dim=1)

        # --- Part 6: Advanced Foot Interaction Rewards ---
        # Calculate foot clearance and contact force rewards
        rew_feet_clearance = self._reward_feet_clearance()
        rew_contact_forces = self._reward_tracking_contacts_shaped_force()

        # --- Combine All Rewards ---
        # Scale by reward weights and timestep duration
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,  # Removed step_dt
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,  # Removed step_dt
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            # Note: This reward is negative (penalty) in the config
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            "rew_torque": rew_torque * self.cfg.torque_reward_scale,  # Penalty for high torque
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": rew_contact_forces * self.cfg.tracking_contacts_shaped_force_reward_scale,
            # Debug metrics (scale=1.0 to see raw values in tensorboard)
            "debug_contact_force_magnitude": self._debug_contact_force_magnitude,
            "debug_desired_contact_states": self._debug_desired_contact_states,
            "debug_is_in_contact": self._debug_is_in_contact,
            "debug_all_body_forces": self._debug_all_body_forces,
        }
        reward = torch.sum(
            torch.stack(
                [
                    rewards["track_lin_vel_xy_exp"],
                    rewards["track_ang_vel_z_exp"],
                    rewards["rew_action_rate"],
                    rewards["raibert_heuristic"],
                    rewards["rew_torque"],
                    rewards["orient"],
                    rewards["lin_vel_z"],
                    rewards["dof_vel"],
                    rewards["ang_vel_xy"],
                    rewards["feet_clearance"],
                    rewards["tracking_contacts_shaped_force"],
                ]
            ),
            dim=0,
        )  # Exclude debug metrics from actual reward sum

        # --- Logging ---
        # Accumulate rewards for tensorboard tracking
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Check if episodes should terminate (either failure or timeout).

        Returns two boolean tensors:
        1. died: True if robot failed (fell over or base touched ground)
        2. time_out: True if episode reached maximum length

        Termination conditions (Parts 3):
        - Base contact: Robot's main body touches ground (should walk on feet only!)
        - Upside down: Robot flipped over (gravity points wrong direction)
        - Base too low: Base height < 0.20m (Part 3: ADDED)
        - Time out: Episode reached 20 seconds

        Why terminate on failure?
        - Prevents robot from learning bad behaviors like crawling
        - Encourages staying upright and walking properly
        - Speeds up training by avoiding useless episode continuation

        Returns:
            (died, time_out): Both are boolean tensors of shape (num_envs,)
        """
        # Check if episode reached maximum length (20 seconds = 1000 steps at 50Hz)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Get contact forces from all body parts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # Check if base (torso) is contacting ground with force > 1.0 N
        # torch.any checks if ANY timestep in history had contact
        cstr_termination_contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )

        # Check if robot is upside down
        # When upright, gravity projects as negative Z. If positive → robot flipped!
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0

        # Check if base height is too low (Part 3: IMPLEMENTED)
        # If robot crouches/collapses below 0.20m, terminate episode
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min

        # Combine all failure conditions with OR logic
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset specific environments (when they fail or timeout).

        This function:
        1. Resets robot to standing pose with zero velocity
        2. Samples new random velocity commands
        3. Clears action history
        4. Logs episode statistics (rewards, termination reasons)
        5. Staggers resets to avoid training spikes

        Args:
            env_ids: Which environments to reset. If None, resets all.

        Why reset?
        - Failed episodes: Robot fell, need fresh start
        - Timeout: Episode finished, ready for new commands
        - Initial setup: All environments start with random timing
        """
        # Log experiment configuration on first reset (to TensorBoard)
        if not self._config_logged:
            self._log_experiment_config_to_tensorboard()

        # Handle "reset all" case
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # Reset robot physics state
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Stagger episode lengths to avoid all envs resetting at once
        # This prevents training spikes and makes learning smoother
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Clear action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # --- Sample New Random Velocity Commands ---
        # Each environment gets random target: vx, vy, yaw_rate in [-1, 1]
        # This ensures robot learns to handle all movement directions
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        # --- Reset Robot to Standing Pose ---
        joint_pos = self.robot.data.default_joint_pos[env_ids]  # Standing joint angles
        joint_vel = self.robot.data.default_joint_vel[env_ids]  # Zero velocity
        default_root_state = self.robot.data.default_root_state[env_ids]  # Base position/orientation

        # Place robot at terrain origin (important for terrain randomization later)
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Write state back to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- Log Episode Statistics ---
        extras = dict()
        # Log average reward per second for each reward component
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0  # Clear for next episode

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # Log termination reasons (helps diagnose training issues)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        # Reset action history buffer for action rate calculation
        self.last_actions[env_ids] = 0.0

        # Reset gait clock for Raibert Heuristic (Part 4: IMPLEMENTED)
        self.gait_indices[env_ids] = 0

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """
        Get current positions of all four feet in world frame.

        Used by Raibert Heuristic to calculate foot placement errors.

        Returns:
            Foot positions tensor, shape (num_envs, 4, 3) for [FL, FR, RL, RR]
        """
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _step_contact_targets(self):
        """
        Update gait clock and compute desired contact states for all feet (Part 4: IMPLEMENTED).

        This function implements a trot gait pattern where diagonal feet move together:
        - Frequency: 3 Hz (3 steps per second)
        - Phase offset: 0.5 (50% - diagonal pairs are synchronized)
        - Duration: 50% stance, 50% swing

        Steps:
        1. Increment gait clock based on time and frequency
        2. Calculate phase for each foot (with appropriate offsets for trot gait)
        3. Convert phases to sine waves (clock_inputs) for observation
        4. Compute smooth contact transitions using von Mises distribution

        The clock_inputs are added to observations so the policy can learn
        phase-dependent behaviors (e.g., lift leg during swing phase).
        """
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs])
            )

        self.clock_inputs[:, 0] = torch.sin(2 * math.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * math.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * math.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * math.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(
            0, kappa
        ).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_FR = smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RL = smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RR = smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)
        )

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _reward_raibert_heuristic(self):
        """
        Calculate Raibert Heuristic reward for proper foot placement (Part 4: IMPLEMENTED).

        This reward guides the robot to place its feet at ideal locations based on
        the classic Raibert Heuristic for stable legged locomotion.

        Algorithm:
        1. Get current foot positions in world frame
        2. Transform to body frame (so positions are relative to robot's orientation)
        3. Calculate nominal foot positions (default standing stance)
        4. Apply Raibert offsets based on commanded velocity:
           - Forward velocity → feet should land further forward
           - Yaw velocity → adjust lateral foot positions for turning
        5. Compute error between actual and ideal foot positions
        6. Return squared error as penalty (lower error = better gait)

        Physical intuition:
        - When running forward, you naturally place feet ahead of your body
        - When turning, outer feet sweep wider than inner feet
        - This reward teaches the robot these natural movement patterns

        Returns:
            Penalty (squared error) for each environment, shape (num_envs,)
        """
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w), cur_footsteps_translated[:, i, :]
            )

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            device=self.device,
        ).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [
                desired_stance_length / 2,
                desired_stance_length / 2,
                -desired_stance_length / 2,
                -desired_stance_length / 2,
            ],
            device=self.device,
        ).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    def _reward_feet_clearance(self):
        """
        Penalize feet not lifting high enough during swing phase (Part 6: REIMPLEMENTED).

        Re-implemented based on IsaacGymEnvs logic but using IsaacLab API.
        Reference: https://github.com/Jogima-cyber/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/go2_terrain.py

        Key differences from IsaacGymEnvs:
        - IsaacGymEnvs uses dynamic target_height based on gait phase
        - We adapt the logic to use IsaacLab's ArticulationData API

        Physical intuition:
        - During swing phase, feet should lift higher in the middle of the swing
        - Target height varies: low when entering/exiting swing, high in mid-swing
        - This encourages natural arc-like foot trajectory

        Algorithm:
        1. Calculate phases: dynamic value that peaks in mid-swing
        2. Calculate target_height: 0.08 * phases + 0.02 (foot radius offset)
        3. Get actual foot heights from world frame
        4. Penalize squared error weighted by swing mask

        Returns:
            Penalty (non-negative) for each environment, shape (num_envs,)
        """
        # Get foot heights (Z coordinate) in world frame
        foot_height = self.foot_positions_w[:, :, 2]  # Shape: (num_envs, 4)

        # Calculate phases: peaks at 1.0 during mid-swing, 0.0 at stance
        # This creates a smooth dynamic target that encourages arc-like foot motion
        # foot_indices range: [0, 1], where 0-0.5 is stance, 0.5-1.0 is swing
        phases = 1.0 - torch.abs(1.0 - torch.clamp((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)

        # Dynamic target height: varies from 0.02m (at swing start/end) to 0.10m (mid-swing)
        # 0.02 is foot radius offset, 0.08 is maximum clearance
        target_height = 0.08 * phases + 0.02  # Shape: (num_envs, 4)

        # Swing mask: 1.0 during swing, 0.0 during stance
        # desired_contact_states ≈ 1 means foot should be on ground (stance)
        # desired_contact_states ≈ 0 means foot should be in air (swing)
        swing_mask = 1.0 - self.desired_contact_states  # Shape: (num_envs, 4)

        # Calculate penalty: squared error between target and actual height, weighted by swing mask
        # Only penalizes during swing phase (swing_mask ≈ 1)
        penalty = torch.square(target_height - foot_height) * swing_mask

        # Sum across all 4 feet
        penalty = torch.sum(penalty, dim=1)  # Shape: (num_envs,)

        return penalty

    def _reward_tracking_contacts_shaped_force(self):
        """
        Penalize feet touching ground during swing phase (Part 6: REIMPLEMENTED).

        Re-implemented based on IsaacGymEnvs logic but using IsaacLab API.
        Reference: https://github.com/Jogima-cyber/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/go2_terrain.py

        Key differences from IsaacGymEnvs:
        - IsaacGymEnvs uses single-sided penalty (only penalize swing contact)
        - Uses F²/100 for force shaping (squared force, not linear)
        - We adapt the logic to use IsaacLab's ContactSensorData API

        Physical intuition:
        - During swing phase, feet should NOT touch the ground
        - Any contact force during swing indicates foot dragging or stumbling
        - We don't reward stance contact (robot naturally learns this for stability)

        Algorithm:
        1. Get contact forces from ContactSensor using _feet_ids_sensor
        2. Calculate force magnitude for each foot
        3. Apply exponential shaping: 1 - exp(-F²/100)
        4. Penalize shaped force during swing phase only
        5. Average across 4 feet

        Returns:
            Penalty (negative) for each environment, shape (num_envs,)
            Values closer to 0 = better (less swing contact)
        """
        # Step 1: Get contact forces from sensor
        # CRITICAL: Must use _feet_ids_sensor (not _feet_ids) for ContactSensorData
        forces_w = self._contact_sensor.data.net_forces_w  # Shape: (num_envs, num_bodies, 3)
        foot_forces = forces_w[:, self._feet_ids_sensor, :]  # Shape: (num_envs, 4, 3)

        # Step 2: Calculate force magnitude (L2 norm)
        force_norm = torch.linalg.norm(foot_forces, dim=-1)  # Shape: (num_envs, 4)

        # Step 3: Get desired contact states
        desired_contact = self.desired_contact_states  # Shape: (num_envs, 4)

        # Step 4: Calculate penalty using IsaacGymEnvs formula
        # Key difference: Uses F² (squared) not F (linear), and divided by 100
        # This creates stronger penalty for large forces during swing
        rew_tracking_contacts = torch.zeros(self.num_envs, device=self.device)

        for i in range(4):
            # Swing mask: 1.0 when foot should be in air, 0.0 when on ground
            swing_mask_i = 1.0 - desired_contact[:, i]

            # Force shaping: 1 - exp(-F²/100)
            # F²/100 means: 0N→0, 10N→0.09, 30N→0.59, 100N→0.99
            # Stronger penalty for high forces than Tyler's linear version
            force_shaped_i = 1.0 - torch.exp(-1.0 * force_norm[:, i] ** 2 / 100.0)

            # Accumulate penalty (negative sign because we penalize swing contact)
            rew_tracking_contacts += -swing_mask_i * force_shaped_i

        # Average across 4 feet (IsaacGymEnvs divides by 4)
        rew_tracking_contacts = rew_tracking_contacts / 4.0

        # --- Debug logging (keep Tyler's debug variables for tensorboard) ---
        self._debug_contact_force_magnitude = torch.sum(force_norm, dim=1)
        self._debug_desired_contact_states = torch.sum(desired_contact, dim=1)
        is_in_contact = (force_norm > 1.0).float()
        self._debug_is_in_contact = torch.sum(is_in_contact, dim=1)

        all_forces = self._contact_sensor.data.net_forces_w
        all_force_magnitudes = torch.norm(all_forces, dim=-1)
        self._debug_all_body_forces = torch.sum(all_force_magnitudes, dim=1)

        return rew_tracking_contacts

    def _set_debug_vis_impl(self, debug_vis: bool):
        """
        Toggle visibility of debug visualization markers.

        Shows/hides velocity arrows:
        - Green arrow: Target velocity command
        - Blue arrow: Actual robot velocity

        Useful for debugging why robot isn't following commands correctly.

        Args:
            debug_vis: True to show markers, False to hide
        """
        if debug_vis:
            # Create markers on first use
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # Show markers
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            # Hide markers if they exist
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """
        Update visualization markers every frame (called automatically).

        This updates the position, direction, and length of velocity arrows
        to show current target vs actual velocity in real-time.
        """
        # Safety check: don't access robot data if simulation stopped
        if not self.robot.is_initialized:
            return

        # Position arrows above robot base
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5  # Lift 0.5m above base for visibility

        # Convert velocity vectors to arrow scale and rotation
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        # Update marker visualization
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert 2D velocity vector into arrow visualization (scale and orientation).

        This helper function takes a velocity vector and converts it to:
        - Arrow scale: Length proportional to velocity magnitude
        - Arrow rotation: Points in velocity direction

        Args:
            xy_velocity: 2D velocity vectors, shape (num_envs, 2) - [vx, vy]

        Returns:
            arrow_scale: Arrow length, shape (num_envs, 3)
            arrow_quat: Arrow orientation, shape (num_envs, 4) - quaternion
        """
        # Get default arrow size from config
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale

        # Scale arrow length by velocity magnitude (×3 for visibility)
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        # Calculate arrow direction from velocity vector
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])  # atan2(vy, vx)
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        # Transform from body frame to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
