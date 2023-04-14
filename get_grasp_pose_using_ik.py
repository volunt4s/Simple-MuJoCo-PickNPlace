import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from src.ik_module import solve_IK

def get_z_rot_mat(z_deg):
    z_rad = np.deg2rad(z_deg)
    y_rad = np.deg2rad(3.0)
    x_rad = np.deg2rad(180.0)

    rpy = np.array([z_rad, y_rad, x_rad])

    rot = R.from_euler('zyx', rpy)

    return rot.as_matrix()


def get_q_from_ik(env):
    body_name = 'panda_eef'

    """
    Define grasping pose

    Task sequence loop
        1. pre_grasp
        2. rotate_eef_i0
        3. move_down_1
        4. grasp
        5. pre_grasp
        6. rotate_eef_i1
        7. move_down_2
        8. release (endloop, go to 1)
    """

    pre_grasp_p = np.array([0.78, 0.0, 1.4])
    pre_grasp_R = np.array([
        [6.123e-17, 0.9848, 0.1736],
        [1.0, 0.0, 0.0],
        [0.0, 0.1736, -0.9848]
    ])

    rotate_eef_p_lst = []
    rotate_eef_R_lst = []

    move_down_p_1_lst = []
    move_down_R_1_lst = []

    move_down_p_2_lst = []
    move_down_R_2_lst = []

    to_rot_angle = np.linspace(-90.0, 90.0, num=10, endpoint=True)

    for angle in to_rot_angle:
        rotate_eef_p = pre_grasp_p
        rotate_eef_R = get_z_rot_mat(angle)
        rotate_eef_p_lst.append(rotate_eef_p)
        rotate_eef_R_lst.append(rotate_eef_R)
        
        move_down_p_1 = rotate_eef_p - np.array([-0.02, 0.0, 0.18])
        move_down_R_1 = rotate_eef_R
        move_down_p_1_lst.append(move_down_p_1)
        move_down_R_1_lst.append(move_down_R_1)

        move_down_p_2 = rotate_eef_p - np.array([-0.02, 0.0, 0.15])
        move_down_R_2 = rotate_eef_R
        move_down_p_2_lst.append(move_down_p_2)
        move_down_R_2_lst.append(move_down_R_2)

    # Check grasp pose
    # Render grasp pose arrow
    check_grasp_pose = False
    if check_grasp_pose:
        while True:
            env.viewer.add_marker(
                pos = pre_grasp_p,
                mat = pre_grasp_R,
                type = mujoco.mjtGeom.mjGEOM_ARROW,
                size = [0.01, 0.01, 0.1],
                rgba = [1, 0, 1, 1],
                label = ''
            )
            # Render
            env.render()

    # Solve IK
    is_render = False
    pre_grasp_q = solve_IK(
        env, 
        max_tick=1000, 
        p_trgt=pre_grasp_p, 
        R_trgt=pre_grasp_R, 
        body_name=body_name,
        is_render=is_render,
        VERBOSE=False
    )

    rotate_eef_q_lst = []
    for rotate_eef_p, rotate_eef_R in zip(rotate_eef_p_lst, rotate_eef_R_lst):
        rotate_eef_q = solve_IK(
            env,
            max_tick=1000,
            p_trgt=rotate_eef_p,
            R_trgt=rotate_eef_R,
            body_name=body_name,
            curr_q=pre_grasp_q,
            is_render=is_render,
            VERBOSE=False
        )
        rotate_eef_q_lst.append(rotate_eef_q)
    
    move_down_q_1_lst = []
    for i, (move_down_p_1, move_down_R_1) in enumerate(zip(move_down_p_1_lst, move_down_R_1_lst)):
        move_down_q_1 = solve_IK(
            env,
            max_tick=1000,
            p_trgt=move_down_p_1,
            R_trgt=move_down_R_1,
            body_name=body_name,
            curr_q=rotate_eef_q_lst[i],
            is_render=is_render,
            VERBOSE=False
        )
        move_down_q_1_lst.append(move_down_q_1)

    move_down_q_2_lst = []
    for i, (move_down_p_2, move_down_R_2) in enumerate(zip(move_down_p_2_lst, move_down_R_2_lst)):
        move_down_q_2 = solve_IK(
            env,
            max_tick=1000,
            p_trgt=move_down_p_2,
            R_trgt=move_down_R_2,
            body_name=body_name,
            curr_q=rotate_eef_q_lst[i],
            is_render=is_render,
            VERBOSE=False
        )
        move_down_q_2_lst.append(move_down_q_2)
    
    # Add gripper joint
    pre_grasp_q = np.concatenate([pre_grasp_q, [np.pi, -np.pi]])

    for i, tmp in enumerate(rotate_eef_q_lst):
        tmp = np.concatenate([tmp, [np.pi, -np.pi]])
        rotate_eef_q_lst[i] = tmp

    for i, tmp in enumerate(move_down_q_1_lst):
        tmp = np.concatenate([tmp, [np.pi, -np.pi]])
        move_down_q_1_lst[i] = tmp

    for i, tmp in enumerate(move_down_q_2_lst):
        tmp = np.concatenate([tmp, [np.pi, -np.pi]])
        move_down_q_2_lst[i] = tmp

    env.reset()

    return pre_grasp_q, rotate_eef_q_lst, move_down_q_1_lst, move_down_q_2_lst