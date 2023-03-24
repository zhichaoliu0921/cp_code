from copy import deepcopy
import numpy as np
import math
class trajectory_pattern_generation:
    def __init__(self, waypoint_list, vel_st=[0.0, 0.0, 0.0],
                 vel_ed=[0.0, 0.0, 0.0]):
        self.waypoint_list = waypoint_list
        self.vel_st = vel_st
        self.vel_ed = vel_ed
        self.trajectory_pattern = {}  # {ith segment: {'waypoints': 'vel_start':  'vel_end':}}

    def generate_traj_pattern(self):
        self.trajectory_pattern[0] = {}
        index = 0
        for i in range(len(self.waypoint_list) - 1):
            # segment for
            self.trajectory_pattern[index] = {}
            self.trajectory_pattern[index]['waypoints'] = self.waypoint_list[i:i+2]
            # start velocity
            self.trajectory_pattern[index]['vel_st'] = [0.0, 0.0, 0.0]
            # end velocity
            self.trajectory_pattern[index]['vel_ed'] = [0.0, 0.0, 0.0]
            # ========================================================================================= #
            index += 1

    # def generate_traj_pattern(self, v_max, a_max):
    #     collision_point_ind = []
    #     for i in range(len(self.collision_states)):
    #         if self.collision_states[i] == 1:
    #             collision_point_ind.append(i)
    #
    #     if len(collision_point_ind) > 0:
    #         self.trajectory_pattern[0] = {}
    #         self.trajectory_pattern[0]['waypoints'] = self.waypoint_list[0:collision_point_ind[0] + 1]
    #         self.trajectory_pattern[0]['vel_st'] = self.vel_st
    #         vec_move = np.array(self.trajectory_pattern[0]['waypoints'][-1]) - \
    #                    np.array(self.trajectory_pattern[0]['waypoints'][-2])
    #         s_move = np.linalg.norm(vec_move)
    #         if v_max**2 / (2 * a_max) <= s_move:
    #             vel_ed_temp = vec_move / s_move * v_max
    #         else:
    #             v_norm = math.sqrt(2 * a_max * s_move)
    #             vel_ed_temp = vec_move / s_move * v_norm
    #         self.trajectory_pattern[0]['vel_ed'] = list(vel_ed_temp)
    #         index = 1
    #         for i in range(len(collision_point_ind) - 1):
    #             # segment for
    #             self.trajectory_pattern[index] = {}
    #             self.trajectory_pattern[index]['waypoints'] = self.waypoint_list[collision_point_ind[i]:
    #                                                                              collision_point_ind[i + 1] + 1]
    #             pos_c = tuple(self.waypoint_list[collision_point_ind[i]])
    #             recover_temp = [self.collision_info_dict[pos_c]['recovery_point'][0],
    #                             self.collision_info_dict[pos_c]['recovery_point'][1],
    #                             0.0]
    #             self.trajectory_pattern[index]['waypoints'][0] = recover_temp
    #
    #             # start velocity
    #             self.trajectory_pattern[index]['vel_st'] = [0.0, 0.0, 0.0]
    #             # end velocity
    #             vec_move = np.array(self.trajectory_pattern[0]['waypoints'][-1]) - \
    #                        np.array(self.trajectory_pattern[0]['waypoints'][-2])
    #             s_move = np.linalg.norm(vec_move)
    #             if v_max ** 2 / (2 * a_max) <= s_move:
    #                 vel_ed_temp = vec_move / s_move * v_max
    #             else:
    #                 v_norm = math.sqrt(2 * a_max * s_move)
    #                 vel_ed_temp = vec_move / s_move * v_norm
    #             self.trajectory_pattern[index]['vel_ed'] = list(vel_ed_temp)
    #             # ========================================================================================= #
    #             index += 1
    #
    #         self.trajectory_pattern[index] = {}
    #         pos_c = tuple(self.waypoint_list[collision_point_ind[-1]])
    #         self.trajectory_pattern[index]['waypoints'] = self.waypoint_list[collision_point_ind[-1]:]
    #         recover_temp = [self.collision_info_dict[pos_c]['recovery_point'][0],
    #                         self.collision_info_dict[pos_c]['recovery_point'][1],
    #                         0.0]
    #         self.trajectory_pattern[index]['waypoints'][0] = recover_temp
    #
    #         # start velocity
    #         self.trajectory_pattern[index]['vel_st'] = [0.0, 0.0, 0.0]
    #         # end velocity
    #         self.trajectory_pattern[index]['vel_ed'] = self.vel_ed
    #     else:
    #         self.trajectory_pattern[0] = {}
    #         self.trajectory_pattern[0]['waypoints'] = deepcopy(self.waypoint_list)
    #         self.trajectory_pattern[0]['vel_st'] = self.vel_st
    #         self.trajectory_pattern[0]['vel_ed'] = self.vel_ed

    # def generate_vel_vec(self, index, pos_c, obs_key, v_reset):
    #     vec_prep = np.array(obs_key) - np.array(pos_c)
    #     vec_prep_norm = np.linalg.norm(vec_prep)
    #     unit_prep = vec_prep / vec_prep_norm
    #     vec_move = np.array(self.trajectory_pattern[index]['waypoints'][1]) - \
    #                np.array(self.trajectory_pattern[index]['waypoints'][0])
    #     vec_move_norm = np.linalg.norm(vec_move)
    #     unit_vec_move = vec_move / vec_move_norm
    #     move_prep = np.dot(unit_vec_move[0:2], unit_prep) * unit_prep
    #     vec_vel = unit_vec_move[0:2] - move_prep
    #     vel_st_temp = vec_vel * v_reset
    #     return vel_st_temp