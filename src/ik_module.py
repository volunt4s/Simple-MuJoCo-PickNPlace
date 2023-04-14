'''
Code fully based on https://github.com/sjchoi86/yet-another-mujoco-tutorial-v2
'''

import numpy as np
import mujoco

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()


def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x


def get_J_body(model,data,body_name,rev_joint_idxs=None):
    J_p = np.zeros((3,model.nv)) # nv: nDoF
    J_R = np.zeros((3,model.nv))
    mujoco.mj_jacBody(model,data,J_p,J_R,data.body(body_name).id)
    if rev_joint_idxs is not None:
        J_p = J_p[:,rev_joint_idxs]
        J_R = J_R[:,rev_joint_idxs]
    J_full = np.array(np.vstack([J_p,J_R]))
    return J_p,J_R,J_full


def solve_IK(env, max_tick, p_trgt, R_trgt, body_name,
             curr_q=None, is_render=False, VERBOSE=False):
    # IK in MJ
    q = env.data.qpos[env.rev_joint_idxs] if curr_q is None else curr_q
    p_trgt = p_trgt
    R_trgt = R_trgt
    
    err_eps = 1e-2

    is_render = False

    while env.tick < max_tick:
        J_p, J_R, J_full = get_J_body(env.model, env.data, body_name, rev_joint_idxs=env.rev_joint_idxs)

        # Numerical IK
        p_curr = env.data.body(body_name).xpos
        R_curr = env.data.body(body_name).xmat.reshape([3, 3])
        p_err = (p_trgt - p_curr)
        R_err = np.linalg.solve(R_curr, R_trgt)
        w_err = R_curr @ r2w(R_err)

        # Compute dq
        J = J_full
        err = np.concatenate((p_err, w_err))
        eps = 1e-1  
        dq = np.linalg.solve(a=(J.T@J) + eps*np.eye(J.shape[1]), b=J.T@err)
        dq = trim_scale(x=dq, th=5.0*np.pi/180.0)

        # Update
        q = q + dq
        env.data.qpos[env.rev_joint_idxs] = q

        # FK
        env.forward()
        
        if is_render:
            if (env.tick % 5) == 0:
                env.render()

        err_norm = np.linalg.norm(err)
        if VERBOSE: print(f"err_norm : {err_norm}")

        if err_norm < err_eps:
            print("IK solved")
            break
            
    env.reset()

    return q
