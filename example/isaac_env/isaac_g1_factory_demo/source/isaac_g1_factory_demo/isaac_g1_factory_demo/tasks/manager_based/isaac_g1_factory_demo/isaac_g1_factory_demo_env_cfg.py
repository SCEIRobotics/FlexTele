# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import tempfile
import os
from pathlib import Path

import torch

import carb
from pink.tasks import FrameTask
import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import UnitreeG1RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
import math

from isaaclab.assets import RigidObjectCollection, RigidObjectCollectionCfg
from . import mdp

assets_dir = Path(__file__).parents[9] / "assets" / "isaac"

from .robots.unitree import G1_INSPIRE_FTP_CFG  # isort: skip
G1_INSPIRE_FTP_CFG.spawn.usd_path = assets_dir / "robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd"

# Actuator configuration for arms (stability focused for manipulation)
# Increased damping improves stability of arm movements
G1_INSPIRE_FTP_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_.*_joint",
    ],
    effort_limit=300,
    velocity_limit=100,
    stiffness=3000.0,
    damping=100.0,
    armature={
        ".*_shoulder_.*": 0.001,
        ".*_elbow_.*": 0.001,
        ".*_wrist_.*_joint": 0.001,
    },
)
# Actuator configuration for hands (flexibility focused for grasping)
# Lower stiffness and damping to improve finger flexibility when grasping objects
G1_INSPIRE_FTP_CFG.actuators["hands"] = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_index_.*",
        ".*_middle_.*",
        ".*_thumb_.*",
        ".*_ring_.*",
        ".*_pinky_.*",
    ],
    effort_limit_sim=30.0,
    velocity_limit_sim=10.0,
    stiffness=10.0,
    damping=0.2,
    armature=0.001,
)

ANGLE2RAD = math.pi / 180.0
pick_height = 0.1
pick_max_vel = 0.2

# assets_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# assets_dir = os.path.join(os.path.dirname(assets_dir), "assets")
# print("assets path: {}".format(assets_dir))

##
# Scene definition
##
@configclass
class IsaacG1FactoryDemoSceneCfg(InteractiveSceneCfg):

    room = AssetBaseCfg(
        prim_path="/World/envs/env_.*/room",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd"
        ),
    )
    
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-3.50831, -0.37687, 0.7385], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=assets_dir / "objects/object_0/socket.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )
    
    object_1 = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object_1",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.58523, -0.37401, 0.7385)),
                spawn=UsdFileCfg(
                    usd_path=assets_dir / "objects/object_1/socket.usd",
                    scale=(1.5, 1.5, 2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                ),
            )
    
    object_2 = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object_2",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.63066, -0.23166, 0.7585), rot=[0.70255, 0.58454, 0.25959, 0.55959]),
                spawn=UsdFileCfg(
                    usd_path=assets_dir / "objects/object_2/socket.usd",
                    scale=(3, 3, 1.5),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                ),
            )
    
    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 1.0),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                # -- left/right hand
                ".*_thumb_.*": 0.0,
                ".*_index_.*": 0.0,
                ".*_middle_.*": 0.0,
                ".*_ring_.*": 0.0,
                ".*_pinky_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )
    # robot.init_state.pos = [-3.10357, -0.47986, 0.79286]
    robot.init_state.pos = [-3.30357, -0.47986, 0.79286]
    robot.init_state.rot = [0, 0, 0, 1]

    # # Ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",
    #     spawn=GroundPlaneCfg(),
    # )

    # # Lights
    # light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    # )

    # camera
    head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/head_cam",
        update_period=0.02,   # 60 Hz
        height=960,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=7.6, focus_distance=400.0, horizontal_aperture=20.0, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_yaw_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
        ],
        hand_joint_names=[
            # All the drive and mimic joints, total 24 joints
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "R_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=24,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                        "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
        enable_gravity_compensation=False,
    )
    left_gripper_action_cfg = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "L_index_proximal_joint",
                "L_middle_proximal_joint",
                "L_pinky_proximal_joint",
                "L_ring_proximal_joint",
                "L_thumb_proximal_yaw_joint",
                "L_index_intermediate_joint",
                "L_middle_intermediate_joint",
                "L_pinky_intermediate_joint",
                "L_ring_intermediate_joint",
                "L_thumb_proximal_pitch_joint",
                "L_thumb_intermediate_joint",
                "L_thumb_distal_joint",
            ],
            open_command_expr={
                "L_index_proximal_joint": 0.0,
                "L_middle_proximal_joint": 0.0,
                "L_pinky_proximal_joint": 0.0,
                "L_ring_proximal_joint": 0.0,
                "L_thumb_proximal_yaw_joint": 74.0*ANGLE2RAD,
                "L_index_intermediate_joint": 0.0,
                "L_middle_intermediate_joint": 0.0,
                "L_pinky_intermediate_joint": 0.0,
                "L_ring_intermediate_joint": 0.0,
                "L_thumb_proximal_pitch_joint": -5.0*ANGLE2RAD,
                "L_thumb_intermediate_joint": 0.0,
                "L_thumb_distal_joint": 0.0,
            },
            close_command_expr={
                "L_index_proximal_joint": 70.0*ANGLE2RAD,
                "L_middle_proximal_joint": 70.0*ANGLE2RAD,
                "L_pinky_proximal_joint": 70.0*ANGLE2RAD,
                "L_ring_proximal_joint": 70.0*ANGLE2RAD,
                "L_thumb_proximal_yaw_joint": 74.0*ANGLE2RAD,
                "L_index_intermediate_joint": 0.0*ANGLE2RAD,
                "L_middle_intermediate_joint": 0.0*ANGLE2RAD,
                "L_pinky_intermediate_joint": 0.0*ANGLE2RAD,
                "L_ring_intermediate_joint": 0.0*ANGLE2RAD,
                "L_thumb_proximal_pitch_joint": 14.5*ANGLE2RAD,
                "L_thumb_intermediate_joint": 24.0*ANGLE2RAD,
                "L_thumb_distal_joint": 0.0,
            },        )
    
    right_gripper_action_cfg = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "R_index_proximal_joint",
                "R_middle_proximal_joint",
                "R_pinky_proximal_joint",
                "R_ring_proximal_joint",
                "R_thumb_proximal_yaw_joint",
                "R_index_intermediate_joint",
                "R_middle_intermediate_joint",
                "R_pinky_intermediate_joint",
                "R_ring_intermediate_joint",
                "R_thumb_proximal_pitch_joint",
                "R_thumb_intermediate_joint",
                "R_thumb_distal_joint",
            ],
            open_command_expr={
                "R_index_proximal_joint": 0.0,
                "R_middle_proximal_joint": 0.0,
                "R_pinky_proximal_joint": 0.0,
                "R_ring_proximal_joint": 0.0,
                "R_thumb_proximal_yaw_joint": 74.0*ANGLE2RAD,
                "R_index_intermediate_joint": 0.0,
                "R_middle_intermediate_joint": 0.0,
                "R_pinky_intermediate_joint": 0.0,
                "R_ring_intermediate_joint": 0.0,
                "R_thumb_proximal_pitch_joint": -5.0*ANGLE2RAD,
                "R_thumb_intermediate_joint": 0.0,
                "R_thumb_distal_joint": 0.0,
            },
            close_command_expr={
                "R_index_proximal_joint": 70.0*ANGLE2RAD,
                "R_middle_proximal_joint": 70.0*ANGLE2RAD,
                "R_pinky_proximal_joint": 70.0*ANGLE2RAD,
                "R_ring_proximal_joint": 70.0*ANGLE2RAD,
                "R_thumb_proximal_yaw_joint": 74.0*ANGLE2RAD,
                "R_index_intermediate_joint": 0.0*ANGLE2RAD,
                "R_middle_intermediate_joint": 0.0*ANGLE2RAD,
                "R_pinky_intermediate_joint": 0.0*ANGLE2RAD,
                "R_ring_intermediate_joint": 0.0*ANGLE2RAD,
                "R_thumb_proximal_pitch_joint": 14.5*ANGLE2RAD,
                "R_thumb_intermediate_joint": 24.0*ANGLE2RAD,
                "R_thumb_distal_joint": 0.0,
            },
        )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})

        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})

        object = ObsTerm(
            func=mdp.object_obs,
            params={"left_eef_link_name": "left_wrist_yaw_link", "right_eef_link_name": "right_wrist_yaw_link"},
        )

        # sensors
        head_cam = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("head_cam"), "data_type": "rgb", "normalize": False}
        )
        # head_cam_depth = ObsTerm(
        #     func=mdp.image,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("head_cam"),
        #         "data_type": "distance_to_image_plane",
        #         "normalize": True,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.1, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(
        func=mdp.task_done_pick_place,
        params={"task_link_name": "right_wrist_yaw_link"},
    )
# @configclass
# class EventCfg:
#     """Configuration for events."""

#     reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

#     reset_object = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {
#                 "x": [-0.04, 0.25],
#                 "y": [-0.125, 0.125],
#                 "yaw": [0, 360.],
#             },
#             "velocity_range": {},
#             "asset_cfg": SceneEntityCfg("object"),
#         },
#     )

#     reset_object_scale = EventTerm(
#         func=mdp.randomize_rigid_body_scale,
#         mode="usd",
#         params={
#             "scale_range": [-0.05, 0.1],
#             "asset_cfg": SceneEntityCfg("object"),
#         },
#     )

#     reset_robot = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {
#                 "z": [0, 0.05],
#             },
#             "velocity_range": {},
#             "asset_cfg": SceneEntityCfg("robot"),
#         },
#     )

#     # random texture of the table
#     randomize_table_visual_material = EventTerm(
#         func=mdp.randomize_visual_texture_prim,
#         mode="reset",
#         params={
#             "prim_path": "/World/envs/env_.*/PackingTable/packing_table/SM_CratePacking_Table_A1/SM_HeavyDutyPackingTable_C02_top",
#             "textures": [
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
#             ],
#             "default_texture": (
#                 f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
#             ),
#         },
#     )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    object_height_target_dis = RewTerm(
        func=mdp.object_height_target_dis,
        weight=1.0,
        params={"min_height": pick_height, "min_vel": pick_max_vel}
    )

@configclass
class IsaacG1FactoryDemoEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: IsaacG1FactoryDemoSceneCfg = IsaacG1FactoryDemoSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    # events = EventCfg()
    rewards: RewardsCfg = RewardsCfg()

    # Unused managers
    commands = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (12), right hand joint pos (12)]
    # not used currently
    idle_action = torch.tensor([
        # 14 hand joints for EEF control
        -0.1487,
        0.2038,
        1.0952,
        0.707,
        0.0,
        0.0,
        0.707,
        0.1487,
        0.2038,
        1.0952,
        0.707,
        0.0,
        0.0,
        0.707,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = self.decimation

        # Convert USD to URDF and change revolute joints to fixed
        # Set the URDF and mesh paths for the IK controller
        urdf_file = os.path.join("urdf", G1_INSPIRE_FTP_CFG.spawn.usd_path.split("/")[-1].replace(".usd", ".urdf"))
        if os.path.exists(os.path.join(self.temp_urdf_dir, urdf_file)):
            print("using existing urdf files...")
            temp_urdf_output_path = os.path.join(self.temp_urdf_dir, urdf_file)
            temp_urdf_meshes_output_path = os.path.join(self.temp_urdf_dir, "meshes")
        else:
            print("converting usd to urdf files...")
            temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
                self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
            )

        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=True,
                            # number of joints in both hands
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            # Please confirm that self.actions.pink_ik_cfg.hand_joint_names is consistent with robot.joint_names[-24:]
                            # The order of the joints does matter as it will be used for converting pink_ik actions to final control actions in IsaacLab.
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            },
        )

        self.image_obs_list = ["head_cam"]
        # Set settings for camera rendering
        self.num_rerenders_on_reset = 1
        self.sim.render.antialiasing_mode = "OFF"
        self.sim.render.enable_translucency = True
        self.sim.render.enable_reflections = True
