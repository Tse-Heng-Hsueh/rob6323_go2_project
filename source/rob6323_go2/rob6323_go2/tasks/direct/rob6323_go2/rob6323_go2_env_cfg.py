# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md). # noqa: E501
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    """
    Configuration for the Go2 quadruped locomotion environment.

    This file contains ALL hyperparameters and settings for:
    - Environment timing and control frequency
    - Action/observation space dimensions
    - Physics simulation parameters
    - Robot model and sensors
    - Reward function weights
    - Visualization settings

    IMPORTANT: You can modify this file for the project!
             (Along with rob6323_go2_env.py - these are the only 2 files you should edit)
    """

    # ============================================
    # ENVIRONMENT TIMING
    # ============================================
    decimation = 4  # Physics runs at 200Hz, but policy acts at 200/4 = 50Hz
    # This means robot receives new actions every 4 physics steps (0.02 seconds)

    episode_length_s = 20.0  # Each training episode lasts 20 seconds max
    # At 50Hz control → 20 * 50 = 1000 steps per episode

    # ============================================
    # ACTION AND OBSERVATION SPACES
    # ============================================
    action_scale = 0.25  # Scale factor for policy outputs (prevents extreme joint movements)
    # Policy output (usually [-1, 1]) is multiplied by 0.25

    action_space = 12  # 12D action: 3 joints per leg × 4 legs
    # Actions represent OFFSETS from default standing pose

    observation_space = 52  # 52D observation vector (Part 4: UPDATED from 48 to 52)
    # Contains: velocities, orientation, commands, joint states, previous actions, clock inputs
    # Added 4D clock inputs for gait phase information (Part 4)

    # --- Gait Shaping Rewards (Part 4: IMPLEMENTED) ---
    raibert_heuristic_reward_scale = -10.0  # Penalty for deviation from ideal foot placement
    # Guides robot to place feet at positions suggested by Raibert Heuristic
    # Negative scale = penalty (minimize foot placement error)

    # --- Future Rewards (Part 6: TODO) ---
    feet_clearance_reward_scale = -30.0  # Penalty for not lifting feet during swing
    tracking_contacts_shaped_force_reward_scale = 4.0  # Reward for proper contact forces

    state_space = 0  # Not used in this project (for centralized training in multi-agent RL)

    debug_vis = True  # Show velocity arrows (green=target, blue=actual) in visualization

    # ============================================
    # PHYSICS SIMULATION SETTINGS
    # ============================================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # Physics timestep = 0.005 seconds (200 Hz)
        # With decimation=4, control frequency = 50 Hz
        render_interval=decimation,  # Render every 4th physics step (for visualization)
        # Material properties for robot-ground interaction
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # When two surfaces touch, multiply their friction values
            restitution_combine_mode="multiply",  # Same for bounciness
            static_friction=1.0,  # Friction when not moving (1.0 = normal ground friction)
            dynamic_friction=1.0,  # Friction when sliding (1.0 = normal ground friction)
            restitution=0.0,  # No bounce (robot doesn't bounce when landing)
        ),
    )

    # ============================================
    # TERRAIN CONFIGURATION
    # ============================================
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # Path in simulation scene graph
        terrain_type="plane",  # Flat plane (baseline)
        # Tutorial later may change to "rough" or "stairs"
        collision_group=-1,  # Default collision group
        # Ground material properties (same as robot for consistency)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,  # Don't show terrain debug visuals
    )

    # ============================================
    # ROBOT CONFIGURATION
    # ============================================
    # Load Unitree Go2 robot model (URDF/USD with joints, links, sensors)
    # prim_path uses regex to create one robot per environment: env_0, env_1, ..., env_4095
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # PD control gains
    Kp = 20.0  # Proportional gain
    Kd = 0.5  # Derivative gain
    torque_limits = 100.0  # Max torque

    # "base_legs" is an arbitrary key we use to group these actuators
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # CRITICAL: Set to 0 to disable implicit P-gain (Part 2)
        damping=0.0,  # CRITICAL: Set to 0 to disable implicit D-gain (Part 2)
    )

    # --- Termination Thresholds (Part 3: IMPLEMENTED) ---
    base_height_min = 0.20  # Terminate episode if base drops below 20cm
    # Prevents robot from learning to crawl or collapse

    # ============================================
    # PARALLEL ENVIRONMENT SETTINGS
    # ============================================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,  # Train with 4096 parallel environments (vectorized training)
        # More envs = faster data collection but needs more GPU memory
        env_spacing=4.0,  # Each environment is 4 meters apart (prevents robot collisions)
        replicate_physics=True,  # Each environment has independent physics simulation
    )

    # ============================================
    # CONTACT SENSOR
    # ============================================
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",  # Monitor ALL robot body parts for contact
        history_length=3,  # Keep last 3 timesteps of contact data
        # Used for detecting transient contacts (e.g., base briefly touched)
        update_period=0.005,  # Update every physics step (200 Hz)
        track_air_time=True,  # Track how long each foot is in the air
        # Useful for gait analysis and foot clearance rewards
    )

    # ============================================
    # VISUALIZATION MARKERS
    # ============================================
    # Green arrow: Shows target velocity command
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    # Blue arrow: Shows actual robot velocity
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Make arrows smaller for better visibility (default size is too large)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # ============================================
    # REWARD FUNCTION WEIGHTS (This is what you'll tune!)
    # ============================================

    # --- Velocity Tracking Rewards (Baseline) ---
    lin_vel_reward_scale = 1.0  # Reward for matching target forward/lateral velocity
    # Larger value = robot tries harder to match speed
    # Scale = 1.0 means this is the primary objective

    yaw_rate_reward_scale = 0.5  # Reward for matching target rotation speed
    # Scale = 0.5 means turning is half as important as moving

    # --- Action Smoothness (Part 1: IMPLEMENTED) ---
    action_rate_reward_scale = -0.1  # Penalty for jerky/sudden actions
    # Encourages smooth, continuous movements by penalizing:
    # 1. Large changes between consecutive actions (first derivative)
    # 2. Sudden acceleration in action changes (second derivative)

    # --- Future Reward Scales (Parts 5-6: TODO) ---
    # You will add more reward scales as you progress through tutorial:
    # - Orientation penalty (keep robot level)
    # - Vertical velocity penalty (reduce bouncing)
    # - Joint velocity penalty (don't move joints too fast)
    # - Angular velocity penalty (minimize roll/pitch)
    # - Torque penalty (don't use excessive force)
    # - Contact force penalties (feet should touch gently)

    # Additional reward scales (Part 5: IMPLEMENTED)
    orient_reward_scale = -5.0  # Penalty for body tilt
    lin_vel_z_reward_scale = -0.02  # Penalty for vertical velocity
    dof_vel_reward_scale = -0.0001  # Penalty for high joint velocities
    ang_vel_xy_reward_scale = -0.001  # Penalty for body roll/pitch

    # Advanced Foot Interaction Rewards (Part 6)
    feet_clearance_reward_scale = -30.0  # Penalty for not lifting feet during swing
    tracking_contacts_shaped_force_reward_scale = 4.0  # Reward for proper contact forces
