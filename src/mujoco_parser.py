import os
import numpy as np
import mujoco
import mujoco_viewer

from src.util import r2w,rpy2r,trim_scale

class MuJoCoParserClass(object):
    """
        MuJoCo Parser class
    """
    def __init__(self,name='Robot',rel_xml_path=None,USE_MUJOCO_VIEWER=False,VERBOSE=True):
        """
            Initialize MuJoCo parser
        """
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.VERBOSE      = VERBOSE
        # Constants
        self.tick         = 0
        self.render_tick  = 0
        # Parse an xml file
        if self.rel_xml_path is not None:
            self._parse_xml()
        # Viewer
        self.USE_MUJOCO_VIEWER = USE_MUJOCO_VIEWER
        if self.USE_MUJOCO_VIEWER:
            self.init_viewer()
        # Reset
        self.reset()
        # Print
        if self.VERBOSE:
            self.print_info()

    def _parse_xml(self):
        """
            Parse an xml file
        """
        self.full_xml_path    = os.path.abspath(os.path.join(os.getcwd(),self.rel_xml_path))
        self.model            = mujoco.MjModel.from_xml_path(self.full_xml_path)
        self.data             = mujoco.MjData(self.model)
        self.n_geom           = self.model.ngeom # number of geometries
        self.geom_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,x)
                                for x in range(self.model.ngeom)]
        self.n_body           = self.model.nbody # number of bodies
        self.body_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,x)
                                for x in range(self.n_body)]
        self.n_dof            = self.model.nv # degree of freedom
        self.n_joint          = self.model.njnt     # number of joints
        self.joint_names      = [mujoco.mj_id2name(self.model,mujoco.mjtJoint.mjJNT_HINGE,x)
                                 for x in range(self.n_joint)]
        self.joint_types      = self.model.jnt_type # joint types
        self.joint_ranges     = self.model.jnt_range # joint ranges
        self.rev_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names  = [self.joint_names[x] for x in self.rev_joint_idxs]
        self.n_rev_joint      = len(self.rev_joint_idxs)
        self.rev_joint_mins   = self.joint_ranges[self.rev_joint_idxs,0]
        self.rev_joint_maxs   = self.joint_ranges[self.rev_joint_idxs,1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins
        self.pri_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names  = [self.joint_names[x] for x in self.pri_joint_idxs]
        self.pri_joint_mins   = self.joint_ranges[self.pri_joint_idxs,0]
        self.pri_joint_maxs   = self.joint_ranges[self.pri_joint_idxs,1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins
        self.n_pri_joint      = len(self.pri_joint_idxs)
        # Actuator
        self.n_ctrl           = self.model.nu # number of actuators (or controls)
        self.ctrl_names       = []
        for addr in self.model.name_actuatoradr:
            ctrl_name = self.model.names[addr:].decode().split('\x00')[0]
            self.ctrl_names.append(ctrl_name) # get ctrl name
        self.ctrl_joint_idxs = []
        for ctrl_idx in range(self.n_ctrl):
            transmission_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid # transmission index
            joint_idx = self.model.jnt_qposadr[transmission_idx][0] # index of the joint when the actuator acts on a joint
            self.ctrl_joint_idxs.append(joint_idx)
        self.ctrl_ranges      = self.model.actuator_ctrlrange # control range

    def print_info(self):
        """
            Printout model information
        """
        print ("n_body:[%d]"%(self.n_geom))
        print ("geom_names:%s"%(self.geom_names))
        print ("n_body:[%d]"%(self.n_body))
        print ("body_names:%s"%(self.body_names))
        print ("n_joint:[%d]"%(self.n_joint))
        print ("joint_names:%s"%(self.joint_names))
        print ("joint_types:%s"%(self.joint_types))
        print ("joint_ranges:\n%s"%(self.joint_ranges))
        print ("n_rev_joint:[%d]"%(self.n_rev_joint))
        print ("rev_joint_idxs:%s"%(self.rev_joint_idxs))
        print ("rev_joint_names:%s"%(self.rev_joint_names))
        print ("rev_joint_mins:%s"%(self.rev_joint_mins))
        print ("rev_joint_maxs:%s"%(self.rev_joint_maxs))
        print ("rev_joint_ranges:%s"%(self.rev_joint_ranges))
        print ("n_pri_joint:[%d]"%(self.n_pri_joint))
        print ("pri_joint_idxs:%s"%(self.pri_joint_idxs))
        print ("pri_joint_names:%s"%(self.pri_joint_names))
        print ("pri_joint_mins:%s"%(self.pri_joint_mins))
        print ("pri_joint_maxs:%s"%(self.pri_joint_maxs))
        print ("pri_joint_ranges:%s"%(self.pri_joint_ranges))
        print ("n_ctrl:[%d]"%(self.n_ctrl))
        print ("ctrl_names:%s"%(self.ctrl_names))
        print ("ctrl_joint_idxs:%s"%(self.ctrl_joint_idxs))
        print ("ctrl_ranges:\n%s"%(self.ctrl_ranges))


    def init_viewer(self,viewer_title='MuJoCo',viewer_width=1200,viewer_height=800,viewer_hide_menus=True):
        """
            Initialize viewer
        """
        self.USE_MUJOCO_VIEWER = True
        self.viewer = mujoco_viewer.MujocoViewer(
                self.model,self.data,mode='window',title=viewer_title,
                width=viewer_width,height=viewer_height,hide_menus=viewer_hide_menus)

    def update_viewer(self,azimuth=None,distance=None,elevation=None,lookat=None,
                      VIS_TRANSPARENT=None,VIS_CONTACTPOINT=None,
                      contactwidth=None,contactheight=None,contactrgba=None,
                      VIS_JOINT=None,jointlength=None,jointwidth=None,jointrgba=None, cam_id=None):
        """
            Initialize viewer
        """
        if azimuth is not None:
            self.viewer.cam.azimuth = azimuth
        if distance is not None:
            self.viewer.cam.distance = distance
        if elevation is not None:
            self.viewer.cam.elevation = elevation
        if lookat is not None:
            self.viewer.cam.lookat = lookat
        if VIS_TRANSPARENT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = VIS_TRANSPARENT
        if VIS_CONTACTPOINT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = VIS_CONTACTPOINT
        if contactwidth is not None:
            self.model.vis.scale.contactwidth = contactwidth
        if contactheight is not None:
            self.model.vis.scale.contactheight = contactheight
        if contactrgba is not None:
            self.model.vis.rgba.contactpoint = contactrgba
        if VIS_JOINT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = VIS_JOINT
        if jointlength is not None:
            self.model.vis.scale.jointlength = jointlength
        if jointwidth is not None:
            self.model.vis.scale.jointwidth = jointwidth
        if jointrgba is not None:
            self.model.vis.rgba.joint = jointrgba
        if cam_id is not None:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = cam_id
        mujoco.mj_forward(self.model,self.data)

    def get_viewer_cam_info(self,VERBOSE=True):
        """
            Get viewer cam information
        """
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance
        cam_elevation = self.viewer.cam.elevation
        cam_lookat    = self.viewer.cam.lookat
        if VERBOSE:
            print ("cam_azimuth:[%.2f] cam_distance:[%.2f] cam_elevation:[%.2f] cam_lookat:%s]"%
                (cam_azimuth,cam_distance,cam_elevation,cam_lookat))
        return cam_azimuth,cam_distance,cam_elevation,cam_lookat

    def is_viewer_alive(self):
        """
            Check whether a viewer is alive
        """
        return self.viewer.is_alive

    def reset(self):
        """
            Reset
        """
        mujoco.mj_resetData(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        self.tick        = 0
        self.render_tick = 0

    def step(self,ctrl=None,ctrl_idxs=None,nstep=1):
        """
            Forward dynamics
        """
        if ctrl is not None:
            if ctrl_idxs is None:
                self.data.ctrl[:] = ctrl
            else:
                self.data.ctrl[ctrl_idxs] = ctrl
        mujoco.mj_step(self.model,self.data,nstep=nstep)
        self.tick = self.tick + 1

    def forward(self,q=None,joint_idxs=None):
        """
            Forward kinematics
        """
        if q is not None:
            if joint_idxs is not None:
                self.data.qpos[joint_idxs] = q
            else:
                self.data.qpos = q
        mujoco.mj_forward(self.model,self.data)
        self.tick = self.tick + 1

    def get_sim_time(self):
        """
            Get simulation time (sec)
        """
        return self.data.time

    def render(self,render_every=1):
        """
            Render
        """
        if self.USE_MUJOCO_VIEWER:
            if ((self.render_tick % render_every) == 0) or (self.render_tick == 0):
                self.viewer.render()
            self.render_tick = self.render_tick + 1
        else:
            print ("[%s] Viewer NOT initialized."%(self.name))

    def grab_image(self):
        """
            Grab the rendered iamge
        """
        img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)
        mujoco.mjr_readPixels(img, None,self.viewer.viewport,self.viewer.ctx)
        img = np.flipud(img) # flip image
        return img

    def close_viewer(self):
        """
            Close viewer
        """
        self.USE_MUJOCO_VIEWER = False
        self.viewer.close()

    def get_p_body(self,body_name):
        """
            Get body position
        """
        return self.data.body(body_name).xpos

    def get_R_body(self,body_name):
        """
            Get body rotation matrix
        """
        return self.data.body(body_name).xmat.reshape([3,3])

    def get_pR_body(self,body_name):
        """
            Get body position and rotation matrix
        """
        p = self.get_p_body(body_name)
        R = self.get_R_body(body_name)
        return p,R

    def get_q(self,joint_idxs=None):
        """
            Get joint position in (radian)
        """
        if joint_idxs is None:
            q = self.data.qpos
        else:
            q = self.data.qpos[joint_idxs]
        return q

    def get_J_body(self,body_name):
        """
            Get Jocobian matrices of a body
        """
        J_p = np.zeros((3,self.model.nv)) # nv: nDoF
        J_R = np.zeros((3,self.model.nv))
        mujoco.mj_jacBody(self.model,self.data,J_p,J_R,self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_ik_ingredients(self,body_name,p_trgt=None,R_trgt=None,IK_P=True,IK_R=True):
        """
            Get IK ingredients
        """
        J_p,J_R,J_full = self.get_J_body(body_name=body_name)
        p_curr,R_curr = self.get_pR_body(body_name=body_name)
        if (IK_P and IK_R):
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_full
            err   = np.concatenate((p_err,w_err))
        elif (IK_P and not IK_R):
            p_err = (p_trgt-p_curr)
            J     = J_p
            err   = p_err
        elif (not IK_P and IK_R):
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_R
            err   = w_err
        else:
            J   = None
            err = None
        return J,err

    def damped_ls(self,J,err,eps=1e-6,stepsize=1.0,th=5*np.pi/180.0):
        """
            Dampled least square for IK
        """
        dq = stepsize*np.linalg.solve(a=(J.T@J)+eps*np.eye(J.shape[1]),b=J.T@err)
        dq = trim_scale(x=dq,th=th)
        return dq

    def onestep_ik(self,body_name,p_trgt=None,R_trgt=None,IK_P=True,IK_R=True,
                   joint_idxs=None,stepsize=1,eps=1e-1,th=5*np.pi/180.0):
        """
            Solve IK for a single step
        """
        J,err = self.get_ik_ingredients(
            body_name=body_name,p_trgt=p_trgt,R_trgt=R_trgt,IK_P=IK_P,IK_R=IK_R)
        dq = self.damped_ls(J,err,stepsize=stepsize,eps=eps,th=th)
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        q = self.get_q(joint_idxs=joint_idxs)
        q = q + dq[joint_idxs]
        # FK
        self.forward(q=q,joint_idxs=joint_idxs)
        return q, err

    def plot_sphere(self,p,r,rgba=[1,1,1,1],label=''):
        """
            Add sphere
        """
        self.viewer.add_marker(
            pos   = p,
            size  = [r,r,r],
            rgba  = rgba,
            type  = mujoco.mjtGeom.mjGEOM_SPHERE,
            label = label)

    def plot_T(self,p,R,
               PLOT_AXIS=True,axis_len=1.0,axis_width=0.01,
               PLOT_SPHERE=False,sphere_r=0.05,sphere_rgba=[1,0,0,0.5],
               label=None):
        """
            Plot coordinate axes
        """
        if PLOT_AXIS:
            rgba_x = [1.0,0.0,0.0,0.9]
            rgba_y = [0.0,1.0,0.0,0.9]
            rgba_z = [0.0,0.0,1.0,0.9]
            # X axis
            R_x = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([1,0,0]))
            p_x = p + R_x[:,2]*axis_len/2
            self.viewer.add_marker(
                pos   = p_x,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_x,
                rgba  = rgba_x,
                label = ''
            )
            R_y = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,1,0]))
            p_y = p + R_y[:,2]*axis_len/2
            self.viewer.add_marker(
                pos   = p_y,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_y,
                rgba  = rgba_y,
                label = ''
            )
            R_z = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,0,1]))
            p_z = p + R_z[:,2]*axis_len/2
            self.viewer.add_marker(
                pos   = p_z,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_z,
                rgba  = rgba_z,
                label = ''
            )
        if PLOT_SPHERE:
            self.viewer.add_marker(
                pos   = p,
                size  = [sphere_r,sphere_r,sphere_r],
                rgba  = sphere_rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = '')
        if label is not None:
            self.viewer.add_marker(
                pos   = p,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.01],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label)

    def plot_arrow(self,p,uv,r_stem=0.03,len_arrow=0.3,rgba=[1,0,0,1],label=''):
        """
            Plot arrow
        """
        p_a = np.copy(np.array([0,0,1]))
        p_b = np.copy(uv)
        p_a_norm = np.linalg.norm(p_a)
        p_b_norm = np.linalg.norm(p_b)
        if p_a_norm > 1e-9: p_a = p_a/p_a_norm
        if p_b_norm > 1e-9: p_b = p_b/p_b_norm
        v = np.cross(p_a,p_b)
        S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        if np.linalg.norm(v) == 0:
            R = np.eye(3,3)
        else:
            R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))

        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r_stem,r_stem,len_arrow],
            rgba  = rgba,
            label = label
        )

    def get_body_names(self,prefix='obj_'):
        """
            Get body names with prefix
        """
        body_names = [x for x in self.body_names if x[:len(prefix)]==prefix]
        return body_names

    def get_contact_info(self,must_include_prefix=None):
        """
            Get contact information
        """
        p_contacts = []
        f_contacts = []
        geom1s = []
        geom2s = []
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]
            # Contact position and frame orientation
            p_contact = contact.pos # contact position
            R_frame = contact.frame.reshape((3,3))
            # Contact force
            f_contact_local = np.zeros(6,dtype=np.float64)
            mujoco.mj_contactForce(self.model,self.data,0,f_contact_local)
            f_contact = R_frame @ f_contact_local[:3] # in the global coordinate
            # Contacting geoms
            contact_geom1 = self.geom_names[contact.geom1]
            contact_geom2 = self.geom_names[contact.geom2]
            # Append
            if must_include_prefix is not None:
                if (contact_geom1[:len(must_include_prefix)] == must_include_prefix) or (contact_geom2[:len(must_include_prefix)] == must_include_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
            else:
                p_contacts.append(p_contact)
                f_contacts.append(f_contact)
                geom1s.append(contact_geom1)
                geom2s.append(contact_geom2)
        return p_contacts,f_contacts,geom1s,geom2s

    def plot_contact_info(self,must_include_prefix=None):
        """
            Plot contact information
        """
        # Get contact information
        p_contacts,f_contacts,geom1s,geom2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        # Render contact informations
        for (p_contact,f_contact,geom1,geom2) in zip(p_contacts,f_contacts,geom1s,geom2s):
            f_norm = np.linalg.norm(f_contact)
            f_uv = f_contact / (f_norm+1e-8)
            f_len = 0.3 # f_norm*0.05
            self.plot_arrow(p=p_contact,uv=f_uv,r_stem=0.01,len_arrow=f_len,rgba=[1,0,0,1],
                        label='')
            self.plot_arrow(p=p_contact,uv=-f_uv,r_stem=0.01,len_arrow=f_len,rgba=[1,0,0,1],
                        label='')
            label = '' # '[%s]-[%s]'%(geom1,geom2)
            self.plot_sphere(p=p_contact,r=0.02,rgba=[1,0.2,0.2,1],label=label)


