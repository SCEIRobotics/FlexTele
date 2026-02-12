import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_pose_to_T(pose):
    # Pose is [x, y, z, qx, qy, qz, qw]
    position = np.array(pose[:3])
    orientation = R.from_quat(pose[3:], scalar_first=False)
    T = np.eye(4)
    T[:3, :3] = orientation.as_matrix()
    T[:3, 3] = position
    return T

def callculate_delta_T(pre_pose, cur_pose):
    if isinstance(pre_pose, list):
        pre_pose = convert_pose_to_T(pre_pose)
    if isinstance(cur_pose, list):
        cur_pose = convert_pose_to_T(cur_pose)
    delta_T = np.linalg.inv(pre_pose) @ cur_pose
    return delta_T