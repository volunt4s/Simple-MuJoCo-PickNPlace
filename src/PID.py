'''
Code fully based on https://github.com/sjchoi86/yet-another-mujoco-tutorial-v2
'''

import numpy as np

# PID Controller            
class PID_ControllerClass(object):
    def __init__(self,
                 name      = 'PID',
                 k_p       = 0.01,
                 k_i       = 0.0,
                 k_d       = 0.001,
                 dt        = 0.01,
                 dim       = 1,
                 dt_min    = 1e-6,
                 out_min   = -np.inf,
                 out_max   = np.inf,
                 ANTIWU    = True,   # anti-windup
                 out_alpha = 0.0    # output EMA (0: no EMA)
                ):
        """
            Initialize PID Controller
        """
        self.name      = name
        self.k_p       = k_p
        self.k_i       = k_i
        self.k_d       = k_d
        self.dt        = dt
        self.dim       = dim
        self.dt_min    = dt_min
        self.out_min   = out_min
        self.out_max   = out_max
        self.ANTIWU    = ANTIWU
        self.out_alpha = out_alpha
        # Buffers
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = 0.0
        self.t_prev   = 0.0
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def reset(self,t_curr=0.0):
        """
            Reset PID Controller
        """
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = t_curr
        self.t_prev   = t_curr
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def update(
        self,
        t_curr  = None,
        x_trgt  = None,
        x_curr  = None,
        VERBOSE = False
        ):
        """
            Update PID controller
            u(t) = K_p e(t) + K_i int e(t) {dt} + K_d {de}/{dt}
        """
        if x_trgt is not None:
            self.x_trgt  = x_trgt
        if t_curr is not None:
            self.t_curr  = t_curr
        if x_curr is not None:
            self.x_curr  = x_curr
            # PID controller updates here
            self.dt       = max(self.t_curr - self.t_prev,self.dt_min)
            self.err_curr = self.x_trgt - self.x_curr     
            self.err_intg = self.err_intg + (self.err_curr*self.dt)
            self.err_diff = self.err_curr - self.err_prev
            
            if self.ANTIWU: # anti-windup
                self.err_out = self.err_curr * self.out_val
                self.err_intg[self.err_out<0.0] = 0.0
            
            if self.dt > self.dt_min:
                self.p_term   = self.k_p * self.err_curr
                self.i_term   = self.k_i * self.err_intg
                self.d_term   = self.k_d * self.err_diff / self.dt
                self.out_val  = np.clip(
                    a     = self.p_term + self.i_term + self.d_term,
                    a_min = self.out_min,
                    a_max = self.out_max)
                # Smooth the output control value using EMA
                self.out_val = self.out_alpha*self.out_val_prev + \
                    (1.0-self.out_alpha)*self.out_val
                self.out_val_prev = self.out_val

                if VERBOSE:
                    print ("cnt:[%d] t_curr:[%.5f] dt:[%.5f]"%
                           (self.cnt,self.t_curr,self.dt))
                    print (" x_trgt:   %s"%(self.x_trgt))
                    print (" x_curr:   %s"%(self.x_curr))
                    print (" err_curr: %s"%(self.err_curr))
                    print (" err_intg: %s"%(self.err_intg))
                    print (" p_term:   %s"%(self.p_term))
                    print (" i_term:   %s"%(self.i_term))
                    print (" d_term:   %s"%(self.d_term))
                    print (" out_val:  %s"%(self.out_val))
                    print (" err_out:  %s"%(self.err_out))
            # Backup
            self.t_prev   = self.t_curr
            self.err_prev = self.err_curr
        # Counter
        if (t_curr is not None) and (x_curr is not None):
            self.cnt = self.cnt + 1
            
    def out(self):
        """
            Get control output
        """
        return self.out_val