import matplotlib.pyplot as plt
from Waypoint_generation_CI.waypoint_generation_horizon import path_generation
from Trajectory_generation.trajectory_profile_generation_CI_horizon import trajectory_pattern_generation
from Trajectory_generation.trajectory_generation_horizon import trajGenerator
from Trajectory_generation.trajectory_generation_uniform import trajGenerator as trajGenerator_uni
import numpy as np
import math
import time
import pandas as pd
from copy import deepcopy
from Mapping.Generate_map_with_horizon import map_gen_horizon

def main():
    min_x = -10.0
    min_y = -10.0
    max_x = 10.0
    max_y = 10.0
    resolution = 0.5
    # obs_radius = 0.58  # (0.028 + 0.03) m
    obs_radius = 0.43  # (0.028 + 0.015) m
    start = (-8.0, -8.0)
    goal = (8.0, 8.0)
    v_max = 3.0  # m/s
    v_horizon = 1.0
    a_max = 4.0  # m/s^2
    vel_recover_prop = 0.0
    pos_recover_prop = 0.366
    T_recover = 2.5
    h_vis = 5.0
    T_map = 5.0
    obstalce_circle_dict_vis = dict()

    # with 1 collision
    obstalce_circle_dict = {(5.25, 5.75): {'ind': 8, 'radius': 0.58}, (-6.75, 0.25): {'ind': 24, 'radius': 0.58},
(5.25, 1.75): {'ind': 20, 'radius': 0.58}, (-7.75, 6.75): {'ind': 9, 'radius': 0.58},
(-8.75, -4.25): {'ind': 23, 'radius': 0.58}, (4.75, 8.25): {'ind': 12, 'radius': 0.58},
(7.25, -8.75): {'ind': 16, 'radius': 0.58}, (5.75, -6.25): {'ind': 26, 'radius': 0.58},
(7.75, 3.75): {'ind': 5, 'radius': 0.58}, (-4.75, -1.75): {'ind': 19, 'radius': 0.58},
(-6.25, -5.75): {'ind': 13, 'radius': 0.58}, (-4.75, 2.75): {'ind': 21, 'radius': 0.58},
(0.75, -4.25): {'ind': 17, 'radius': 0.58}, (0.25, 8.75): {'ind': 28, 'radius': 0.58},
(-3.75, -6.75): {'ind': 11, 'radius': 0.58}, (2.25, -2.25): {'ind': 3, 'radius': 0.58},
(-0.25, 2.25): {'ind': 0, 'radius': 0.58}, (3.25, -6.25): {'ind': 7, 'radius': 0.58},
(8.25, 1.25): {'ind': 6, 'radius': 0.58}, (-7.25, 2.75): {'ind': 2, 'radius': 0.58},
(-2.75, -3.25): {'ind': 4, 'radius': 0.58}, (-2.75, 5.75): {'ind': 10, 'radius': 0.58},
(4.25, -8.75): {'ind': 14, 'radius': 0.58}, (0.75, -8.75): {'ind': 1, 'radius': 0.58},
(-0.75, -6.75): {'ind': 29, 'radius': 0.58}, (3.75, 3.75): {'ind': 25, 'radius': 0.58},
(7.75, -1.75): {'ind': 22, 'radius': 0.58}, (-1.75, -0.25): {'ind': 18, 'radius': 0.58},
(-2.75, 8.75): {'ind': 15, 'radius': 0.58}, (3.25, 0.25): {'ind': 27, 'radius': 0.58}}

    obstalce_circle_dict_1 = deepcopy(obstalce_circle_dict)

    traj_x_e = []
    traj_y_e = []
    vel_x_e = []
    vel_y_e = []
    vel_norm_e = []
    acc_x_e = []
    acc_y_e = []
    time_list_e = []
    ctrl_energy = 0.0
    path_length = 0.0
    sum_T = 0.0
    path_comp_time = 0.0
    traj_comp_time = 0.0

    vel_st = [0.0, 0.0, 0.0]
    vel_ed = [0.0, 0.0, 0.0]
    acc_st = [0.0, 0.0, 0.0]
    acc_ed = [0.0, 0.0, 0.0]
    jerk_st = [0.0, 0.0, 0.0]
    jerk_ed = [0.0, 0.0, 0.0]
    snap_st = [0.0, 0.0, 0.0]
    snap_ed = [0.0, 0.0, 0.0]

    ind_seg = 0
    while math.hypot(start[0] - goal[0], start[1] - goal[1]) >= h_vis:
        fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
        circle_vis = plt.Circle(start, h_vis, linewidth=2, linestyle=':', edgecolor='darkgray', fill=False)
        ax.add_patch(circle_vis)

        map_gen = map_gen_horizon(obstalce_circle_dict, start, obstalce_circle_dict_vis, h_vis, obs_radius - 0.28)
        map_gen.get_obs_in_horizon()
        obstalce_circle_dict_vis = map_gen.obstalce_circle_dict_vis
        # file_name = file_direction_name + "\obs_vis_" + str(ind_seg) + '.txt'
        # for item in obstalce_circle_dict_vis.keys():
        #     str1 = str(item)
        #     with open(file_name, 'a') as file:
        #         file.write(str1)
        #         file.write('\r')

        starttime = time.time()
        path_gen = path_generation(min_x, max_x, min_y, max_y, obstalce_circle_dict_vis, resolution, start, goal)
        waypoint_list, collision_state, collision_info_dict = path_gen.waypoint_generation()
        waypoint_list, collision_state, vel_ed_horizon = path_gen.adjust_end_state_by_horizon(waypoint_list,
                                                                                              collision_state,
                                                                                              0.1, h_vis, v_horizon)
        endtime = time.time()
        comp_time1 = (endtime - starttime) * 1000.0
        print 'generate path with ', comp_time1, 'ms. '
        path_comp_time += comp_time1
        print 'waypoints'
        print waypoint_list
        print 'collision states'
        print collision_state

        # file_name = file_direction_name + "\waypoints_" + str(ind_seg) + '.txt'
        # for item in waypoint_list:
        #     str1 = str(item)
        #     with open(file_name, 'a') as file:
        #         file.write(str1)
        #         file.write('\r')
        #
        # file_name = file_direction_name + "\collisions_" + str(ind_seg) + '.txt'
        # for item in collision_state:
        #     str1 = str(item)
        #     with open(file_name, 'a') as file:
        #         file.write(str1)
        #         file.write('\r')

        starttime = time.time()
        traj_pattern_gen = trajectory_pattern_generation(waypoint_list, collision_state, collision_info_dict,
                                                         vel_st=vel_st, acc_st=acc_st, jerk_st=jerk_st, snap_st=snap_st,
                                                         vel_ed=vel_ed_horizon)
        collision_info_dict_adjust = traj_pattern_gen.generate_traj_pattern(v_max, a_max, pos_recover_prop,
                                                                            vel_recover_prop)

        time_prior = 0.0
        TS_total = [0.0]
        index_list = sorted(traj_pattern_gen.trajectory_pattern.keys())
        print index_list
        traj_dict = {}
        for ind in index_list:
            waypoints_temp = np.array(traj_pattern_gen.trajectory_pattern[ind]['waypoints'])
            vel_st_temp = traj_pattern_gen.trajectory_pattern[ind]['vel_st']
            vel_ed_temp = traj_pattern_gen.trajectory_pattern[ind]['vel_ed']
            acc_st_temp = traj_pattern_gen.trajectory_pattern[ind]['acc_st']
            acc_ed_temp = traj_pattern_gen.trajectory_pattern[ind]['acc_ed']
            jerk_st_temp = traj_pattern_gen.trajectory_pattern[ind]['jerk_st']
            jerk_ed_temp = traj_pattern_gen.trajectory_pattern[ind]['jerk_ed']
            snap_st_temp = traj_pattern_gen.trajectory_pattern[ind]['jerk_st']
            snap_ed_temp = traj_pattern_gen.trajectory_pattern[ind]['jerk_ed']
            if collision_state[ind] == 1:
                traj_temp = trajGenerator(waypoints_temp, vel_st_temp, vel_ed_temp, acc_st_temp, acc_ed_temp,
                                          jerk_st_temp, jerk_ed_temp, snap_st_temp, snap_ed_temp,
                                          max_vel=v_max, gamma=1e3)
            else:
                # traj_temp = trajGenerator_uni(waypoints_temp, vel_st_temp, vel_ed_temp, max_vel=v_max, max_acc=a_max)
                traj_temp = trajGenerator(waypoints_temp, vel_st_temp, vel_ed_temp, acc_st_temp, acc_ed_temp,
                                          jerk_st_temp, jerk_ed_temp, snap_st_temp, snap_ed_temp,
                                          max_vel=v_max, gamma=1e3)
            traj_dict[ind] = {'traj_info': traj_temp, 'time_start': time_prior, 'time_interval': traj_temp.TS[-1]}
            TS_total.append(traj_temp.TS[-1] + time_prior)
            time_prior = traj_temp.TS[-1] + time_prior
            if ind < len(collision_state) - 1:
                if collision_state[ind + 1] == 1:
                    time_prior += T_recover
        endtime = time.time()
        comp_time2 = (endtime - starttime) * 1000.0
        print 'generate trajectory with ', comp_time2, 'ms. '
        traj_comp_time += comp_time2

        # TS_moving = [0.0]
        # time_prior = 0.0
        # for ind in index_list:
        #     TS_moving.append(traj_dict[ind]['time_interval'] + time_prior)
        #     time_prior = traj_dict[ind]['time_interval'] + time_prior

        ind = 0
        T_map_actual = T_map
        T_map_remain = T_map
        while ind < max(index_list):
            if T_map_remain > traj_dict[ind]['time_interval']:
                T_map_remain -= traj_dict[ind]['time_interval']
            if collision_state[ind + 1] == 1:
                T_map_actual += traj_dict[ind + 1]['time_interval'] + T_recover
                ind = ind + 2

        # print collision_info_dict_adjust
        # print 'traj pattern'
        # print traj_pattern_gen.trajectory_pattern
        # print TS_total
        # # print TS_moving
        # print traj_dict

        Tmax = TS_total[-1]
        # print 'time of trajectory:', Tmax

        traj_x_p = []
        traj_y_p = []
        vel_x_p = []
        vel_y_p = []
        vel_norm_p = []
        acc_x_p = []
        acc_y_p = []
        time_list_p = []
        # pos_pre = [start[0], start[1]]
        for ind in index_list:
            traj_seg = traj_dict[ind]['traj_info']
            Tmax_seg = traj_seg.TS[-1]
            t_start = traj_dict[ind]['time_start']
            if collision_state[ind] == 1:
                traj_x_p.append(waypoint_list[ind][0])
                traj_y_p.append(waypoint_list[ind][1])
                vel_x_p.append(0.0)
                vel_y_p.append(0.0)
                vel_norm_p.append(0.0)
                acc_x_p.append(0.0)
                acc_y_p.append(0.0)
                time_list_p.append(t_start - T_recover + sum_T)

            for t in np.arange(0.0, Tmax_seg, 0.01):
                des_state = traj_seg.get_des_state(t)
                traj_x_p.append(des_state.pos[0])
                traj_y_p.append(des_state.pos[1])
                vel_x_p.append(des_state.vel[0])
                vel_y_p.append(des_state.vel[1])
                vel_norm_p.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
                acc_x_p.append(des_state.acc[0])
                acc_y_p.append(des_state.acc[1])
                time_list_p.append(t + t_start + sum_T)
                # ctrl_energy += (des_state.acc[0] ** 2 + des_state.acc[1] ** 2) * 0.01
                # path_length += math.sqrt((des_state.pos[0] - pos_pre[0]) ** 2 + (des_state.pos[1] - pos_pre[1]) ** 2)
                # pos_pre = [des_state.pos[0], des_state.pos[1]]

        df_p = pd.DataFrame(columns=('time', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y'))
        df_p['time'] = time_list_p
        df_p['pos_x'] = traj_x_p
        df_p['pos_y'] = traj_y_p
        df_p['vel_x'] = vel_x_p
        df_p['vel_y'] = vel_y_p
        df_p['acc_x'] = acc_x_p
        df_p['acc_y'] = acc_y_p
        # file_name = file_direction_name + "\Traj_p" + str(ind_seg) + '.csv'
        # df_p.to_csv(file_name)

        traj_x_temp = []
        traj_y_temp = []
        vel_x_temp = []
        vel_y_temp = []
        vel_norm_temp = []
        acc_x_temp = []
        acc_y_temp = []
        time_list_temp = []
        pos_pre = [start[0], start[1]]
        break_flag = False
        sum_t_map = 0.0
        for ind in index_list:
            traj_seg = traj_dict[ind]['traj_info']
            Tmax_seg = traj_seg.TS[-1]
            t_start = traj_dict[ind]['time_start']
            if collision_state[ind] == 1:
                traj_x_temp.append(waypoint_list[ind][0])
                traj_y_temp.append(waypoint_list[ind][1])
                vel_x_temp.append(0.0)
                vel_y_temp.append(0.0)
                vel_norm_temp.append(0.0)
                acc_x_temp.append(0.0)
                acc_y_temp.append(0.0)
                time_list_temp.append(t_start - T_recover + sum_T)

            for t in np.arange(0.0, Tmax_seg, 0.01):
                if t + t_start >= T_map_actual - 0.01:
                    print t, t_start, T_map_actual
                    break_flag = True
                    break
                des_state = traj_seg.get_des_state(t)
                traj_x_temp.append(des_state.pos[0])
                traj_y_temp.append(des_state.pos[1])
                vel_x_temp.append(des_state.vel[0])
                vel_y_temp.append(des_state.vel[1])
                vel_norm_temp.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
                acc_x_temp.append(des_state.acc[0])
                acc_y_temp.append(des_state.acc[1])
                time_list_temp.append(t + t_start + sum_T)
                ctrl_energy += (des_state.acc[0] ** 2 + des_state.acc[1] ** 2) * 0.01
                path_length += math.sqrt((des_state.pos[0] - pos_pre[0]) ** 2 + (des_state.pos[1] - pos_pre[1]) ** 2)
                pos_pre = [des_state.pos[0], des_state.pos[1]]

            if break_flag:
                break

        # print 'time related'
        # print TS_total
        # print T_map_actual, ind, t

        traj_x_e.extend(traj_x_temp)
        traj_y_e.extend(traj_y_temp)
        vel_x_e.extend(vel_x_temp)
        vel_y_e.extend(vel_y_temp)
        vel_norm_e.extend(vel_norm_temp)
        acc_x_e.extend(acc_x_temp)
        acc_y_e.extend(acc_y_temp)
        time_list_e.extend(time_list_temp)

        df_e = pd.DataFrame(columns=('time', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y'))

        df_e['time'] = time_list_temp
        df_e['pos_x'] = traj_x_temp
        df_e['pos_y'] = traj_y_temp
        df_e['vel_x'] = vel_x_temp
        df_e['vel_y'] = vel_y_temp
        df_e['acc_x'] = acc_x_temp
        df_e['acc_y'] = acc_y_temp
        # file_name = file_direction_name + "\Traj_e" + str(ind_seg) + '.csv'
        # df_e.to_csv(file_name)

        path_x = []
        path_y = []
        for i in range(len(waypoint_list)):
            if collision_state[i] == 0:
                path_x.append(waypoint_list[i][0])
                path_y.append(waypoint_list[i][1])
            else:
                path_x.append(waypoint_list[i][0])
                path_y.append(waypoint_list[i][1])

                path_x.append(collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][0])
                path_y.append(collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][1])

        for obs_pos in obstalce_circle_dict.keys():
            if obs_pos in obstalce_circle_dict_vis:
                circle_temp_true = plt.Circle(obs_pos, obs_radius - 0.28, color='black')
                circle_temp_aug = plt.Circle(obs_pos, obstalce_circle_dict[obs_pos]['radius'], linewidth=2, linestyle=':',
                                             edgecolor='orange', fill=False)
            else:
                circle_temp_true = plt.Circle(obs_pos, obs_radius - 0.28, color='grey')
                circle_temp_aug = plt.Circle(obs_pos, obstalce_circle_dict[obs_pos]['radius'], linewidth=2, linestyle=':',
                                             edgecolor='wheat', fill=False)

            ax.add_patch(circle_temp_aug)
            ax.add_patch(circle_temp_true)

        plt.plot(start[0], start[1], "og")
        plt.plot(goal[0], goal[1], "sg")
        plt.plot([start[0], goal[0]], [start[1], goal[1]], 'k', linestyle='--')
        plt.plot(path_x, path_y, 'b', linestyle='--')
        plt.plot(traj_x_p, traj_y_p, 'b')
        plt.plot(traj_x_e, traj_y_e, 'r')

        for i in range(len(waypoint_list)):
            if collision_state[i] == 0:
                circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.28, color='c')
                ax.add_patch(circle_temp_robot)
            else:
                circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.28, color='r')
                ax.add_patch(circle_temp_robot)

                circle_temp_recover = plt.Circle([collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][0],
                                                  collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][1]],
                                                 0.3, color='pink')

                ax.add_patch(circle_temp_recover)

        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])
        plt.grid(True)
        plt.axis("equal")
        plt.show()

        # generate obstacle dictionary without collided obstacles
        for pos_c in collision_info_dict_adjust.keys():
            obs_c = collision_info_dict_adjust[pos_c]['obs_center']
            if obs_c in obstalce_circle_dict_1:
                del (obstalce_circle_dict_1[obs_c])

        ind_seg += 1

        # setup replanning start and start velocity
        T_now = T_map_actual
        if T_now > TS_total[-1]:
            T_now = TS_total[-1] - 0.001
        ind_map = np.where(T_now >= np.array(TS_total))[0][-1]
        t_map = T_now - TS_total[ind_map]
        print 'segment for start'
        print ind_map, t_map, T_map_actual, TS_total[-1]
        des_state = traj_dict[ind_map]['traj_info'].get_des_state(t_map)
        start = (des_state.pos[0], des_state.pos[1])
        vel_st = [des_state.vel[0], des_state.vel[1], des_state.vel[2]]
        acc_st = [des_state.acc[0], des_state.acc[1], des_state.acc[2]]
        jerk_st = [des_state.jerk[0], des_state.jerk[1], des_state.jerk[2]]
        snap_st = [des_state.snap[0], des_state.snap[1], des_state.snap[2]]
        sum_T += min(T_map_actual, TS_total[-1])

    # last segment
    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    circle_vis = plt.Circle(start, h_vis, linewidth=2, linestyle=':', edgecolor='darkgray', fill=False)
    ax.add_patch(circle_vis)

    map_gen = map_gen_horizon(obstalce_circle_dict, start, obstalce_circle_dict_vis, h_vis, obs_radius - 0.28)
    map_gen.get_obs_in_horizon()
    obstalce_circle_dict_vis = map_gen.obstalce_circle_dict_vis
    # file_name = file_direction_name + "\obs_vis_" + str(ind_seg) + '.txt'
    # for item in obstalce_circle_dict_vis.keys():
    #     str1 = str(item)
    #     with open(file_name, 'a') as file:
    #         file.write(str1)
    #         file.write('\r')

    starttime = time.time()
    path_gen = path_generation(min_x, max_x, min_y, max_y, obstalce_circle_dict_vis, resolution, start, goal)
    waypoint_list, collision_state, collision_info_dict = path_gen.waypoint_generation()
    waypoint_list, collision_state, vel_ed_horizon = path_gen.adjust_end_state_by_horizon(waypoint_list,
                                                                                          collision_state,
                                                                                          0.1, h_vis, v_horizon)
    endtime = time.time()
    comp_time1 = (endtime - starttime) * 1000.0
    print 'generate path with ', comp_time1, 'ms. '
    path_comp_time += comp_time1
    print 'waypoints'
    print waypoint_list
    print 'collision states'
    print collision_state

    # file_name = file_direction_name + "\waypoints_" + str(ind_seg) + '.txt'
    # for item in waypoint_list:
    #     str1 = str(item)
    #     with open(file_name, 'a') as file:
    #         file.write(str1)
    #         file.write('\r')
    #
    # file_name = file_direction_name + "\collisions_" + str(ind_seg) + '.txt'
    # for item in collision_state:
    #     str1 = str(item)
    #     with open(file_name, 'a') as file:
    #         file.write(str1)
    #         file.write('\r')

    starttime = time.time()
    traj_pattern_gen = trajectory_pattern_generation(waypoint_list, collision_state, collision_info_dict,
                                                     vel_st=vel_st, acc_st=acc_st, jerk_st=jerk_st, snap_st=snap_st,
                                                     vel_ed=vel_ed_horizon)
    collision_info_dict_adjust = traj_pattern_gen.generate_traj_pattern(v_max, a_max, pos_recover_prop,
                                                                        vel_recover_prop)

    time_prior = 0.0
    TS_total = [0.0]
    index_list = sorted(traj_pattern_gen.trajectory_pattern.keys())
    print index_list
    traj_dict = {}
    for ind in index_list:
        waypoints_temp = np.array(traj_pattern_gen.trajectory_pattern[ind]['waypoints'])
        vel_st_temp = traj_pattern_gen.trajectory_pattern[ind]['vel_st']
        vel_ed_temp = traj_pattern_gen.trajectory_pattern[ind]['vel_ed']
        acc_st_temp = traj_pattern_gen.trajectory_pattern[ind]['acc_st']
        acc_ed_temp = traj_pattern_gen.trajectory_pattern[ind]['acc_ed']
        jerk_st_temp = traj_pattern_gen.trajectory_pattern[ind]['jerk_st']
        jerk_ed_temp = traj_pattern_gen.trajectory_pattern[ind]['jerk_ed']
        snap_st_temp = traj_pattern_gen.trajectory_pattern[ind]['jerk_st']
        snap_ed_temp = traj_pattern_gen.trajectory_pattern[ind]['jerk_ed']
        if collision_state[ind] == 1:
            traj_temp = trajGenerator(waypoints_temp, vel_st_temp, vel_ed_temp, acc_st_temp, acc_ed_temp,
                                      jerk_st_temp, jerk_ed_temp, snap_st_temp, snap_ed_temp,
                                      max_vel=v_max, gamma=1e3)
        else:
            # traj_temp = trajGenerator_uni(waypoints_temp, vel_st_temp, vel_ed_temp, max_vel=v_max, max_acc=a_max)
            traj_temp = trajGenerator(waypoints_temp, vel_st_temp, vel_ed_temp, acc_st_temp, acc_ed_temp,
                                      jerk_st_temp, jerk_ed_temp, snap_st_temp, snap_ed_temp,
                                      max_vel=v_max, gamma=1e3)
        traj_dict[ind] = {'traj_info': traj_temp, 'time_start': time_prior, 'time_interval': traj_temp.TS[-1]}
        TS_total.append(traj_temp.TS[-1] + time_prior)
        time_prior = traj_temp.TS[-1] + time_prior
        if ind < len(collision_state) - 1:
            if collision_state[ind + 1] == 1:
                time_prior += T_recover
    endtime = time.time()
    comp_time2 = (endtime - starttime) * 1000.0
    print 'generate trajectory with ', comp_time2, 'ms. '
    traj_comp_time += comp_time2

    # TS_moving = [0.0]
    # time_prior = 0.0
    # for ind in index_list:
    #     TS_moving.append(traj_dict[ind]['time_interval'] + time_prior)
    #     time_prior = traj_dict[ind]['time_interval'] + time_prior

    ind = 0
    # T_map_actual = T_map
    # T_map_remain = T_map
    # while ind < max(index_list):
    #     if T_map_remain > traj_dict[ind]['time_interval']:
    #         T_map_remain -= traj_dict[ind]['time_interval']
    #     if collision_state[ind + 1] == 1:
    #         T_map_actual += traj_dict[ind + 1]['time_interval'] + T_recover
    #         ind = ind + 2

    # print collision_info_dict_adjust
    # print 'traj pattern'
    # print traj_pattern_gen.trajectory_pattern
    # print TS_total
    # # print TS_moving
    # print traj_dict
    #
    # Tmax = TS_total[-1]
    # print 'time of trajectory:', Tmax

    traj_x_p = []
    traj_y_p = []
    vel_x_p = []
    vel_y_p = []
    vel_norm_p = []
    acc_x_p = []
    acc_y_p = []
    time_list_p = []
    # pos_pre = [start[0], start[1]]
    for ind in index_list:
        traj_seg = traj_dict[ind]['traj_info']
        Tmax_seg = traj_seg.TS[-1]
        t_start = traj_dict[ind]['time_start']
        if collision_state[ind] == 1:
            traj_x_p.append(waypoint_list[ind][0])
            traj_y_p.append(waypoint_list[ind][1])
            vel_x_p.append(0.0)
            vel_y_p.append(0.0)
            vel_norm_p.append(0.0)
            acc_x_p.append(0.0)
            acc_y_p.append(0.0)
            time_list_p.append(t_start - T_recover + sum_T)

        for t in np.arange(0.0, Tmax_seg, 0.01):
            des_state = traj_seg.get_des_state(t)
            traj_x_p.append(des_state.pos[0])
            traj_y_p.append(des_state.pos[1])
            vel_x_p.append(des_state.vel[0])
            vel_y_p.append(des_state.vel[1])
            vel_norm_p.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
            acc_x_p.append(des_state.acc[0])
            acc_y_p.append(des_state.acc[1])
            time_list_p.append(t + t_start + sum_T)
            # ctrl_energy += (des_state.acc[0] ** 2 + des_state.acc[1] ** 2) * 0.01
            # path_length += math.sqrt((des_state.pos[0] - pos_pre[0]) ** 2 + (des_state.pos[1] - pos_pre[1]) ** 2)
            # pos_pre = [des_state.pos[0], des_state.pos[1]]

    df_p = pd.DataFrame(columns=('time', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y'))
    df_p['time'] = time_list_p
    df_p['pos_x'] = traj_x_p
    df_p['pos_y'] = traj_y_p
    df_p['vel_x'] = vel_x_p
    df_p['vel_y'] = vel_y_p
    df_p['acc_x'] = acc_x_p
    df_p['acc_y'] = acc_y_p
    # file_name = file_direction_name + "\Traj_p" + str(ind_seg) + '.csv'
    # df_p.to_csv(file_name)

    traj_x_temp = []
    traj_y_temp = []
    vel_x_temp = []
    vel_y_temp = []
    vel_norm_temp = []
    acc_x_temp = []
    acc_y_temp = []
    time_list_temp = []
    pos_pre = [start[0], start[1]]
    break_flag = False
    sum_t_map = 0.0
    for ind in index_list:
        traj_seg = traj_dict[ind]['traj_info']
        Tmax_seg = traj_seg.TS[-1]
        t_start = traj_dict[ind]['time_start']
        if collision_state[ind] == 1:
            traj_x_temp.append(waypoint_list[ind][0])
            traj_y_temp.append(waypoint_list[ind][1])
            vel_x_temp.append(0.0)
            vel_y_temp.append(0.0)
            vel_norm_temp.append(0.0)
            acc_x_temp.append(0.0)
            acc_y_temp.append(0.0)
            time_list_temp.append(t_start - T_recover + sum_T)

        for t in np.arange(0.0, Tmax_seg, 0.01):
            # if t + t_start >= T_map_actual - 0.01:
            #     break_flag = True
            #     break
            des_state = traj_seg.get_des_state(t)
            traj_x_temp.append(des_state.pos[0])
            traj_y_temp.append(des_state.pos[1])
            vel_x_temp.append(des_state.vel[0])
            vel_y_temp.append(des_state.vel[1])
            vel_norm_temp.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
            acc_x_temp.append(des_state.acc[0])
            acc_y_temp.append(des_state.acc[1])
            time_list_temp.append(t + t_start + sum_T)
            ctrl_energy += (des_state.acc[0] ** 2 + des_state.acc[1] ** 2) * 0.01
            path_length += math.sqrt((des_state.pos[0] - pos_pre[0]) ** 2 + (des_state.pos[1] - pos_pre[1]) ** 2)
            pos_pre = [des_state.pos[0], des_state.pos[1]]

        # if break_flag:
        #     break

    # print 'time related'
    # print TS_total
    # print T_map_actual, ind, t

    traj_x_e.extend(traj_x_temp)
    traj_y_e.extend(traj_y_temp)
    vel_x_e.extend(vel_x_temp)
    vel_y_e.extend(vel_y_temp)
    vel_norm_e.extend(vel_norm_temp)
    acc_x_e.extend(acc_x_temp)
    acc_y_e.extend(acc_y_temp)
    time_list_e.extend(time_list_temp)

    df_e = pd.DataFrame(columns=('time', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y'))

    df_e['time'] = time_list_temp
    df_e['pos_x'] = traj_x_temp
    df_e['pos_y'] = traj_y_temp
    df_e['vel_x'] = vel_x_temp
    df_e['vel_y'] = vel_y_temp
    df_e['acc_x'] = acc_x_temp
    df_e['acc_y'] = acc_y_temp

    # file_name = file_direction_name + "\Traj_e" + str(ind_seg) + '.csv'
    # df_e.to_csv(file_name)

    path_x = []
    path_y = []
    for i in range(len(waypoint_list)):
        if collision_state[i] == 0:
            path_x.append(waypoint_list[i][0])
            path_y.append(waypoint_list[i][1])
        else:
            path_x.append(waypoint_list[i][0])
            path_y.append(waypoint_list[i][1])

            path_x.append(collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][0])
            path_y.append(collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][1])

    for obs_pos in obstalce_circle_dict.keys():
        if obs_pos in obstalce_circle_dict_vis:
            circle_temp_true = plt.Circle(obs_pos, obs_radius - 0.28, color='black')
            circle_temp_aug = plt.Circle(obs_pos, obstalce_circle_dict[obs_pos]['radius'], linewidth=2, linestyle=':',
                                         edgecolor='orange', fill=False)
        else:
            circle_temp_true = plt.Circle(obs_pos, obs_radius - 0.28, color='grey')
            circle_temp_aug = plt.Circle(obs_pos, obstalce_circle_dict[obs_pos]['radius'], linewidth=2, linestyle=':',
                                         edgecolor='wheat', fill=False)

        ax.add_patch(circle_temp_aug)
        ax.add_patch(circle_temp_true)

    plt.plot(start[0], start[1], "og")
    plt.plot(goal[0], goal[1], "sg")
    plt.plot([start[0], goal[0]], [start[1], goal[1]], 'k', linestyle='--')
    plt.plot(path_x, path_y, 'b', linestyle='--')
    plt.plot(traj_x_p, traj_y_p, 'b')
    plt.plot(traj_x_e, traj_y_e, 'r')

    for i in range(len(waypoint_list)):
        if collision_state[i] == 0:
            circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.28, color='c')
            ax.add_patch(circle_temp_robot)
        else:
            circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.28, color='r')
            ax.add_patch(circle_temp_robot)

            circle_temp_recover = plt.Circle([collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][0],
                                              collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][1]],
                                             0.3, color='pink')

            ax.add_patch(circle_temp_recover)

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    # draw overall trajectory
    # plt.plot(time_list_e, vel_x_e, 'b', linestyle='--')
    # plt.plot(time_list_e, vel_y_e, 'r', linestyle='--')
    # plt.xlim([0.0, time_list_e[-1]])
    # plt.show()
    #
    # plt.plot(time_list_e, vel_norm_e, 'b', linestyle='--')
    # plt.xlim([0.0, time_list_e[-1]])
    # plt.title("velocity Norm")
    # plt.grid(True)
    # plt.show()
    #
    # plt.plot(time_list_e, acc_x_e, 'b', linestyle='--')
    # plt.plot(time_list_e, acc_y_e, 'r', linestyle='--')
    # plt.xlim([0.0, time_list_e[-1]])
    # plt.show()

    df = pd.DataFrame(columns=('time', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y'))

    df['time'] = time_list_e
    df['pos_x'] = traj_x_e
    df['pos_y'] = traj_y_e
    df['vel_x'] = vel_x_e
    df['vel_y'] = vel_y_e
    df['acc_x'] = acc_x_e
    df['acc_y'] = acc_y_e

    # df.to_csv('E:\Dropbox\ARQ_simulation_result\Trajectories\CP_horizon\env30\Map5\Traj_overall.csv')

    num_collision = 0
    for c_s in collision_state:
        if c_s == 1:
            num_collision += 1

    print 'path time:', path_comp_time
    print 'traj generation time', traj_comp_time
    print 'total time:', path_comp_time + traj_comp_time
    # print 'ctrl energy:', ctrl_energy
    print 'trajectory time:', time_list_e[-1]
    print 'path length:', path_length

    # calculate minimal obstacle distance
    dis_obs_list = []
    for pos_traj in zip(traj_x_e, traj_y_e):
        dist_temp = np.inf
        for pos_obs in obstalce_circle_dict_1.keys():
            # obs_radius = obstalce_circle_dict_1[pos_obs]['radius']
            dist_temp = min([dist_temp, math.hypot(pos_traj[0] - pos_obs[0], pos_traj[1] - pos_obs[1])])

        dis_obs_list.append(dist_temp)

    print 'minimal distance to obstacle', min(dis_obs_list)

if __name__ == '__main__':
    main()