import math
from Trajectory_generation.trajectory_utilities import *

class trajGenerator:
    def __init__(self, waypoints, st_vel, ed_vel, max_vel=5, max_acc=2):
        self.waypoints = waypoints
        self.st_vel = st_vel
        self.ed_vel = ed_vel
        self.max_vel = max_vel
        self.max_acc = max_acc
        len, dim = waypoints.shape
        self.order = 10
        self.dim = dim
        self.len = len
        self.TS = [0.0]
        self.yaw = 0
        self.heading = np.zeros(2)
        self.coeffs = []
        self.get_des_pattern()

    def get_des_pattern(self):
        pos_val = np.linalg.norm(self.waypoints[1] - self.waypoints[0])
        acc_val = self.max_acc
        st_vel_val = np.linalg.norm(np.array(self.st_vel))
        ed_vel_val = np.linalg.norm(np.array(self.ed_vel))
        if ed_vel_val > 0:
            t_acc = (ed_vel_val - st_vel_val) / acc_val
            s_acc = st_vel_val * t_acc + 0.5 * acc_val * t_acc**2
            self.TS.append(t_acc)
            self.coeffs.extend([0.0, st_vel_val, 0.5 * acc_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            s_const = pos_val - s_acc
            v_const = ed_vel_val
            t_const = s_const / v_const

            if t_const > 0:
                self.TS.append(t_acc + t_const)
                self.coeffs.extend([s_acc, v_const, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            t_acc = (self.max_vel - st_vel_val) / acc_val
            s_acc = st_vel_val * t_acc + 0.5 * acc_val * t_acc ** 2
            if 2 * s_acc >= pos_val:
                v_move = math.sqrt(acc_val * pos_val)
                t_acc = v_move / acc_val
                s_acc = st_vel_val * t_acc + 0.5 * acc_val * t_acc ** 2
                self.TS.append(t_acc)
                self.TS.append(2 * t_acc)
                self.coeffs.extend([0.0, st_vel_val, 0.5 * acc_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.coeffs.extend([s_acc, v_move, -0.5 * acc_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                s_const = pos_val - 2 * s_acc
                v_const = self.max_vel
                t_const = s_const / v_const
                self.TS.append(t_acc)
                self.TS.append(t_acc + t_const)
                self.TS.append(2 * t_acc + t_const)
                self.coeffs.extend([0.0, st_vel_val, 0.5 * acc_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.coeffs.extend([s_acc, v_const, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.coeffs.extend([s_acc + s_const, v_const, -0.5 * acc_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    def get_des_state(self, t):

        if t > self.TS[-1]:
            t = self.TS[-1] - 0.001

        i = np.where(t >= self.TS)[0][-1]

        t = t - self.TS[i]
        coeff = self.coeffs[self.order * i:self.order * (i + 1)]

        pos_val = np.dot(coeff, polyder(t))
        vel_val = np.dot(coeff, polyder(t, 1))
        accl_val = np.dot(coeff, polyder(t, 2))

        unit_pos_vec = (self.waypoints[1] - self.waypoints[0]) / np.linalg.norm(self.waypoints[1] - self.waypoints[0])
        pos = self.waypoints[0] + pos_val * unit_pos_vec
        vel = vel_val * unit_pos_vec
        accl = accl_val * unit_pos_vec

        jerk = None
        snap = None

        return DesiredState(pos, vel, accl, jerk, snap)