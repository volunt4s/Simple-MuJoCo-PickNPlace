import numpy as np

from src.mujoco_parser import MuJoCoParserClass
from src.PID import PID_ControllerClass
from get_grasp_pose_using_ik import get_q_from_ik

def set_gripper(desired_q, option="open"):
    if option == "open":
        desired_q[7], desired_q[8] = np.pi, -np.pi
    elif option == "close":
        desired_q[7], desired_q[8] = 0.0, 0.0
    return desired_q

def main():
    # MuJoCo Panda parsing
    xml_path = 'asset/panda/franka_panda_w_objs.xml'
    env = MuJoCoParserClass(name='Panda', rel_xml_path=xml_path, VERBOSE=False)
    # env.print_info()

    env.forward()

    # Initialize MuJoCo viewer
    env.init_viewer(viewer_title="PNP test", viewer_width=1600, viewer_height=900,
                    viewer_hide_menus=False)
    env.update_viewer(cam_id=0)
    env.reset()
    
    # Initialize PID including gripper joints
    PID = PID_ControllerClass(
        name='PID', dim=env.n_ctrl,
        k_p=800.0,
        k_i=20.0,
        k_d=100.0,
        out_min=env.ctrl_ranges[env.ctrl_joint_idxs, 0],
        out_max=env.ctrl_ranges[env.ctrl_joint_idxs, 1],
        ANTIWU=True
    )
    PID.reset()

    # Get IK solution
    pre_grasp_q, rotate_eef_q_lst, move_down_q_1_lst, move_down_q_2_lst = get_q_from_ik(env)

    # Define env max_tick
    max_tick    = 1000000

    # Define task
    task_sequnce_idx = 0
    task_sequnce = ["pre_grasp",
                    "rotate_eef_i0",
                    "move_down_1",
                    "grasp",
                    "pre_grasp_with_close",
                    "rotate_eef_i1_with_close",
                    "move_down_2_with_close", 
                    "release"]
    rot_idx = 0

    # Control loop
    while env.tick < max_tick:
        # Act every 1500 step
        if env.tick % 1500 == 0:
            # Repeat task
            if task_sequnce_idx >= len(task_sequnce):
                task_sequnce_idx = 0
                if rot_idx < len(rotate_eef_q_lst) - 2:
                    rot_idx = rot_idx + 1
                else: break

            current_task = task_sequnce[task_sequnce_idx]
            task_sequnce_idx = task_sequnce_idx + 1

        if current_task == "pre_grasp":
            desired_q = pre_grasp_q
            desired_q = set_gripper(desired_q, option="open")

        elif current_task == "rotate_eef_i0":
            desired_q = rotate_eef_q_lst[rot_idx]
            desired_q = set_gripper(desired_q, option="open")

        elif current_task == "move_down_1":
            desired_q = move_down_q_1_lst[rot_idx]

        elif current_task == "grasp":
            desired_q = set_gripper(desired_q, option="close")

        elif current_task == "pre_grasp_with_close":
            desired_q = pre_grasp_q
            desired_q = set_gripper(desired_q, option="close")

        elif current_task == "rotate_eef_i1_with_close":
            desired_q = rotate_eef_q_lst[rot_idx+1]
            desired_q = set_gripper(desired_q, option="close")

        elif current_task == "move_down_2_with_close":
            desired_q = move_down_q_2_lst[rot_idx+1]
            desired_q = set_gripper(desired_q, option="close")

        elif current_task == "release":
            desired_q = set_gripper(desired_q, option="open")

        print(f"[{env.tick}] current_task : {current_task}\t task_sequnce_idx : {task_sequnce_idx}\t rot_idx : {rot_idx}")
        PID.update(x_trgt=desired_q)
        PID.update(t_curr=env.get_sim_time(), x_curr=env.get_q(joint_idxs=env.ctrl_joint_idxs), VERBOSE=False)
        torque = PID.out()
        env.step(ctrl=torque, ctrl_idxs=env.ctrl_joint_idxs)

        # Render
        if (env.tick % 3) == 0:
            env.render()

    env.close_viewer()
    print("Done")


if __name__ == "__main__":
    main()