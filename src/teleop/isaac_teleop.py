from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R

from pico import PicoClient
from pico.common import TeleopState, TeleopRecordState, TeleopCommand


def convert_isaac_pose_to_T(position, orientation):
    # Position is [x, y, z]
    # Orientation is [qw, qx, qy, qz]
    orientation = R.from_quat(orientation, scalar_first=True)
    T = np.eye(4)
    T[:3, :3] = orientation.as_matrix()
    T[:3, 3] = position
    return T


def convert_T_to_isaac_pose(T):
    position = T[:3, 3]
    orientation = R.from_matrix(T[:3, :3])
    orientation = orientation.as_quat(scalar_first=True)
    return position, orientation


class IsaacTeleop:
    def __init__(self, convet_M: np.array):
        self.pico_client = PicoClient()
        self.convert_M = convet_M

    def get_teleop_data(self, left_ee_init_T, right_ee_init_T):
        pico_data = self.pico_client.get_latest()

        if pico_data is None:
            return None

        teleop_data = deepcopy(pico_data["teleop"])

        if teleop_data["teleop_state"] == TeleopState.ENABLE:
            left_controller_delta_T = np.array(teleop_data["left_controller_delta_T"])
            right_controller_delta_T = np.array(teleop_data["right_controller_delta_T"])

            left_ee_delta_T = (
                self.convert_M @ left_controller_delta_T @ np.linalg.inv(self.convert_M)
            )
            right_ee_delta_T = (
                self.convert_M
                @ right_controller_delta_T
                @ np.linalg.inv(self.convert_M)
            )

            left_ee_target_T = left_ee_init_T @ left_ee_delta_T
            left_ee_position, left_ee_quat = convert_T_to_isaac_pose(left_ee_target_T)

            right_ee_target_T = right_ee_init_T @ right_ee_delta_T
            right_ee_position, right_ee_quat = convert_T_to_isaac_pose(
                right_ee_target_T
            )

            teleop_data["left_ee_position"] = left_ee_position
            teleop_data["left_ee_quat"] = left_ee_quat
            teleop_data["right_ee_position"] = right_ee_position
            teleop_data["right_ee_quat"] = right_ee_quat
            return teleop_data
        else:
            return None
