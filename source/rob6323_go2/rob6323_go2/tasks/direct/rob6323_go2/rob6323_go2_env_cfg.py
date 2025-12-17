# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

# bonus2 -uneven terrain
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.sensors import RayCasterCfg, patterns

# part 2
from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.2
    action_space = 12
    # # part 4
    # observation_space = 48 + 4  # Added 4 for clock inputs
    # bonus2 - uneven terrain
    # --- Height scanner (RayCaster) ---
    _height_scan_resolution = 0.1
    _height_scan_size = (1.6, 1.0)  # (x_size, y_size) in meters

    # grid points count: (size/res) + 1 in each axis
    _num_height_x = int(_height_scan_size[0] / _height_scan_resolution) + 1
    _num_height_y = int(_height_scan_size[1] / _height_scan_resolution) + 1
    num_height_samples = _num_height_x * _num_height_y

    # base obs = 48, plus 4 clock inputs, plus height samples
    observation_space = (48 + 4 + 12) + num_height_samples

    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=_height_scan_resolution,
            size=[_height_scan_size[0], _height_scan_size[1]],
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # bonus2 - uneven terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0, 
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # robot(s)
    # robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Update robot_cfg
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # "base_legs" is an arbitrary key we use to group these actuators
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # CRITICAL: Set to 0 to disable implicit P-gain
        damping=0.0,    # CRITICAL: Set to 0 to disable implicit D-gain
    )

    # scene
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    # bonus2 - uneven terrain - test 
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.3
    action_rate_reward_scale = -0.01
    raibert_heuristic_reward_scale = -0.3

    orient_reward_scale = -1.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.00005
    ang_vel_xy_reward_scale = -0.0002

    # part 6
    feet_clearance_reward_scale = -0.05
    tracking_contacts_shaped_force_reward_scale = 0.05

    # PD control gains
    Kp = 15.0  # Proportional gain
    Kd = 2.0  # Derivative gain
    torque_limits = 23.5  # Max torque

    # part 3 - termination
    base_height_min = 0.10  # Terminate if base is lower than 10cm

    torque_reward_scale = -2e-5