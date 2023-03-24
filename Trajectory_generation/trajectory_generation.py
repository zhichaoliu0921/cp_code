import math

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from Trajectory_generation.trajectory_utilities import *

class trajGenerator:
    def __init__(self, waypoints, st_vel, ed_vel, max_vel=5, gamma=100):
        self.waypoints = waypoints
        self.st_vel = st_vel
        self.ed_vel = ed_vel
        self.max_vel = max_vel
        self.gamma = gamma
        self.order = 10
        len, dim = waypoints.shape
        self.dim = dim
        self.len = len
        self.TS = np.zeros(self.len)

        # for time allocation
        self.max_acc = 4.0

        self.optimize()
        self.yaw = 0
        self.heading = np.zeros(2)

    def get_cost(self, T):
        coeffs, cost = self.MinimizeSnap(T)
        cost = cost + self.gamma * np.sum(T)
        return cost

    def get_cost_from_coeffs(self, T, coeffs):
        Q = Hessian(T)
        Q_mat = np.mat(Q)
        P = np.array(coeffs)
        cost = np.trace(P.T * Q_mat * P)
        return cost

    def optimize(self):
        Tmin = self.time_allocation()
        T = optimize.minimize(self.get_cost, Tmin, method="COBYLA", constraints=({'type': 'ineq',
                                                                                  'fun': lambda T: T-Tmin}))['x']

        # Tmax = ...
        # T = optimize.minimize(self.get_cost, Tmax, method="COBYLA", constraints=({'type': 'ineq',
        #                                                                           'fun': lambda T: -T+Tmax}))['x']

        self.TS[1:] = np.cumsum(T)
        self.coeffs, self.cost = self.MinimizeSnap(T)

    def MinimizeSnap(self,T):
        unkns = 4*(self.len - 2)

        Q = Hessian(T)
        Q_mat = np.mat(Q)

        A, B = self.get_constraints(T)

        # n = self.len - 1
        # M = np.eye(len(A))
        # # M[[2 * n + 1, 4 * n - 1], :] = M[[4 * n - 1, 2 * n + 1], :]
        # M_mat = np.mat(M)

        A_mat = np.mat(A)
        B_mat = np.mat(B)

        # M_B_mat = M_mat * B_mat

        invA = LA.inv(A_mat)

        if unkns != 0:
            # R = M_mat * invA.T * Q_mat * invA * M_mat.T
            R = invA.T * Q_mat * invA

            Rfp = R[:-unkns,-unkns:]
            Rpp = R[-unkns:,-unkns:]

            # M_B_mat[-unkns:,] = -LA.inv(Rpp) * Rfp.T * M_B_mat[:-unkns,]
            B_mat[-unkns:, ] = -LA.inv(Rpp) * Rfp.T * B_mat[:-unkns, ]

        P = invA * B_mat
        cost = np.trace(P.T * Q_mat * P)

        P = np.array(P)
        return P, cost

    def get_constraints(self, T):
        n = self.len - 1
        o = self.order

        A = np.zeros((self.order*n, self.order*n))
        B = np.zeros((self.order*n, self.dim))

        B[:n, :] = self.waypoints[:-1, :]
        B[n:2*n, :] = self.waypoints[1:, :]

        #waypoints contraints
        for i in range(n):
            A[i, o*i : o*(i+1)] = polyder(0)
            A[i + n, o*i : o*(i+1)] = polyder(T[i])

        #continuity contraints
        for i in range(n-1):
            A[2*n + 4*i: 2*n + 4*(i+1), o*i: o*(i+1)] = -polyder(T[i], 'all')
            A[2*n + 4*i: 2*n + 4*(i+1), o*(i+1): o*(i+2)] = polyder(0, 'all')

        #start and end at rest
        A[6*n - 4 : 6*n, : o] = polyder(0, 'all')
        A[6*n: 6*n + 4, -o : ] = polyder(T[-1], 'all')

        B[6*n - 4, :] = self.st_vel
        B[6*n, :] = self.ed_vel

        #free variables
        for i in range(1,n):
            A[6*n + 4*i : 6*n + 4*(i+1), o*i : o*(i+1)] = polyder(0, 'all')

        return A, B

    def get_des_state(self, t):

        if t > self.TS[-1]:
            t = self.TS[-1] - 0.001

        i = np.where(t >= self.TS)[0][-1]

        t = t - self.TS[i]
        coeff = (self.coeffs.T)[:, self.order * i:self.order * (i + 1)]

        pos = np.dot(coeff, polyder(t))
        vel = np.dot(coeff, polyder(t, 1))
        accl = np.dot(coeff, polyder(t, 2))
        jerk = np.dot(coeff, polyder(t, 3))
        snap = np.dot(coeff, polyder(t, 4))

        return DesiredState(pos, vel, accl, jerk, snap)

    def time_allocation(self):
        # diff = self.waypoints[0:-1] - self.waypoints[1:]
        # Tmin = LA.norm(diff, axis=-1, ord=2) / self.max_vel

        # # solution 2: Trapezoidal time allocation
        # diff = self.waypoints[0:-1] - self.waypoints[1:]
        # s_diff = LA.norm(diff, axis=-1, ord=2)
        # Tmin = np.zeros(len(diff))
        # if len(s_diff) == 1:
        #     s_temp = s_diff[0]
        #     st_vel_norm = LA.norm(self.st_vel)
        #     ed_vel_norm = LA.norm(self.ed_vel)
        #     s_acc = (2 * self.max_vel ** 2 - st_vel_norm ** 2 - ed_vel_norm ** 2) / (2 * self.max_acc)
        #     if s_acc >= s_temp:
        #         v_max = math.sqrt((st_vel_norm ** 2 + ed_vel_norm ** 2 + 2 * self.max_acc * s_temp) / 2)
        #         Tmin[0] = (2 * v_max - st_vel_norm - ed_vel_norm) / self.max_acc
        #     else:
        #         v_max = self.max_vel
        #         T_acc = (2 * v_max - st_vel_norm - ed_vel_norm) / self.max_acc
        #         T_const = (s_temp - s_acc) / v_max
        #         Tmin[0] = T_acc + T_const
        # else:
        #     # first segment
        #     s_temp = s_diff[0]
        #     st_vel_norm = LA.norm(self.st_vel)
        #     s_acc = (self.max_vel ** 2 - st_vel_norm ** 2) / (2 * self.max_acc)
        #     if s_acc >= s_temp:
        #         v_max = math.sqrt(st_vel_norm ** 2 + 2 * self.max_acc * s_temp)
        #         Tmin[0] = (v_max - st_vel_norm) / self.max_acc
        #     else:
        #         v_max = self.max_vel
        #         T_acc = (v_max - st_vel_norm) / self.max_acc
        #         T_const = (s_temp - s_acc) / v_max
        #         Tmin[0] = T_acc + T_const
        #
        #     # intermediate segments
        #     for i in range(1, len(diff)-1):
        #         s_temp = s_diff[i]
        #         Tmin[i] = s_temp / v_max
        #
        #     # last segment
        #     s_temp = s_diff[-1]
        #     ed_vel_norm = LA.norm(self.ed_vel)
        #     s_acc = (self.max_vel ** 2 - ed_vel_norm ** 2) / (2 * self.max_acc)
        #     if s_acc >= s_temp:
        #         v_max = math.sqrt(st_vel_norm ** 2 + 2 * self.max_acc * s_temp)
        #         Tmin[-1] = (v_max - ed_vel_norm) / self.max_acc
        #     else:
        #         v_max = self.max_vel
        #         T_acc = (v_max - ed_vel_norm) / self.max_acc
        #         T_const = (s_temp - s_acc) / v_max
        #         Tmin[-1] = T_acc + T_const

        # solution 3: Time allocation with factor
        alpha = 0.15
        diff = self.waypoints[0:-1] - self.waypoints[1:]
        s_diff = LA.norm(diff, axis=-1, ord=2)
        if len(s_diff) == 1:
            Tmin = LA.norm(diff, axis=-1, ord=2) / self.max_vel  # tune the parameter
            factor = alpha * s_diff[0]
            ed_vel_norm = np.linalg.norm(self.ed_vel)
            if ed_vel_norm > 0:
                if factor < 1.0:
                    Tmin[0] = 1.5 * Tmin[0]
                elif factor < 2.0:
                    Tmin[0] = Tmin[0] * (factor + 0.5)
                else:
                    Tmin[0] = Tmin[0] * 2.5  # tune the parameter
            else:
                if factor < 1.0:
                    Tmin[0] = 2.0 * Tmin[0]
                elif factor < 2.0:
                    Tmin[0] = Tmin[0] * (factor + 1.0)
                else:
                    Tmin[0] = Tmin[0] * 3.0  # tune the parameter
        else:
            Tmin = LA.norm(diff, axis=-1, ord=2) / self.max_vel
            factor = alpha * s_diff[0]
            if factor < 1.0:
                Tmin[0] = 1.0 * Tmin[0]
            elif factor < 2.0:
                Tmin[0] = Tmin[0] * factor
            else:
                Tmin[0] = Tmin[0] * 2.0  # tune the parameter

            for i in range(1, len(s_diff) - 1):
                factor = alpha * s_diff[i]
                if factor < 1.0:
                    Tmin[0] = Tmin[0]
                elif factor < 1.5:
                    Tmin[0] = Tmin[0] * factor
                else:
                    Tmin[0] = Tmin[0] * 1.5  # tune the parameter

            factor = alpha * s_diff[-1]
            # print factor
            if factor < 1.0:
                Tmin[-1] = 1.0 * Tmin[-1]
            elif factor < 2.0:
                Tmin[-1] = Tmin[-1] * factor
            else:
                Tmin[-1] = Tmin[-1] * 2.0

        return Tmin