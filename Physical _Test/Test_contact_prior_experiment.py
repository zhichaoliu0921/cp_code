import matplotlib.pyplot as plt
from Waypoint_generation_CI.waypoint_generation import path_generation
from Trajectory_generation.trajectory_profile_generation_CI import trajectory_pattern_generation
from Trajectory_generation.trajectory_generation import trajGenerator
import numpy as np
import math
import time
import pandas as pd
from copy import deepcopy

def main():
    min_x = -2.4
    min_y = -1.3
    max_x = 2.7
    max_y = 1.3
    resolution = 0.1
    obs_radius = 0.43  # (0.15 + 0.28) m
    start = (-2.0, 0.0)
    # goal = (2.0, 0.2)
    goal = (2.0, -0.2)
    v_max = 2.0  # m/s
    a_max = 4.0  # m/s^2
    vel_recover_prop = 0.0
    pos_recover_prop = 0.36
    T_recover = 1.9

    # with 1 collisions
    obstacle_circle_dict = {(0.0, 0.0): {'ind': 1, 'radius': 0.43}, (-1.0, 0.87): {'ind': 2, 'radius': 0.43},
                            (1.0, -0.93): {'ind': 3, 'radius': 0.43}, (-1.15, -0.85): {'ind': 0, 'radius': 0.43}}

    starttime = time.time()
    path_gen = path_generation(min_x, max_x, min_y, max_y, obstacle_circle_dict, resolution, start, goal)
    waypoint_list, collision_state, collision_info_dict = path_gen.waypoint_generation()
    endtime = time.time()
    comp_time1 = (endtime - starttime) * 1000.0
    # print 'generate path with ', comp_time1, 'ms. '

    starttime = time.time()
    traj_pattern_gen = trajectory_pattern_generation(waypoint_list, collision_state, collision_info_dict)
    collision_info_dict_adjust = traj_pattern_gen.generate_traj_pattern(v_max, a_max, pos_recover_prop,
                                                                        vel_recover_prop)

    time_prior = 0.0
    TS_total = [0.0]
    index_list = sorted(traj_pattern_gen.trajectory_pattern.keys())
    traj_dict = {}
    for ind in index_list:
        waypoints_temp = np.array(traj_pattern_gen.trajectory_pattern[ind]['waypoints'])
        vel_st = traj_pattern_gen.trajectory_pattern[ind]['vel_st']
        vel_ed = traj_pattern_gen.trajectory_pattern[ind]['vel_ed']
        traj_temp = trajGenerator(waypoints_temp, vel_st, vel_ed, max_vel=v_max, gamma=1.2e3)
        traj_dict[ind] = {'traj_info': traj_temp, 'time_start': time_prior, 'time_interval': traj_temp.TS[-1]}
        TS_total.append(traj_temp.TS[-1] + time_prior)
        time_prior = traj_temp.TS[-1] + time_prior
        if collision_state[ind + 1] == 1:
            time_prior += T_recover
    endtime = time.time()
    comp_time2 = (endtime - starttime) * 1000.0
    # print 'generate trajectory with ', comp_time2, 'ms. '

    # print waypoint_list
    # print collision_state
    # print collision_info_dict_adjust
    # print traj_pattern_gen.trajectory_pattern
    # print TS_total
    # print traj_dict

    Tmax = TS_total[-1]
    # print 'time of trajectory:', Tmax

    traj_x = []
    traj_y = []
    vel_x = []
    vel_y = []
    acc_x = []
    acc_y = []
    time_list = []
    ctrl_energy = 0.0
    path_length = 0.0
    pos_pre = [start[0], start[1]]
    for ind in index_list:
        traj_seg = traj_dict[ind]['traj_info']
        Tmax_seg = traj_seg.TS[-1]
        t_start = traj_dict[ind]['time_start']
        if collision_state[ind] == 1:
            traj_x.append(waypoint_list[ind][0])
            traj_y.append(waypoint_list[ind][1])
            vel_x.append(0.0)
            vel_y.append(0.0)
            acc_x.append(0.0)
            acc_y.append(0.0)
            time_list.append(t_start - T_recover)

        for t in np.arange(0.0, Tmax_seg, 0.05):
            des_state = traj_seg.get_des_state(t)
            traj_x.append(des_state.pos[0])
            traj_y.append(des_state.pos[1])
            vel_x.append(des_state.vel[0])
            vel_y.append(des_state.vel[1])
            acc_x.append(des_state.acc[0])
            acc_y.append(des_state.acc[1])
            time_list.append(t + t_start)
            ctrl_energy += (des_state.acc[0]**2 + des_state.acc[1]**2) * 0.05
            path_length += math.sqrt((des_state.pos[0] - pos_pre[0]) ** 2 + (des_state.pos[1] - pos_pre[1]) ** 2)
            pos_pre = [des_state.pos[0], des_state.pos[1]]

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

    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot

    for obs_pos in obstacle_circle_dict.keys():
        circle_temp_true = plt.Circle(obs_pos, obs_radius - 0.28, color='grey')
        circle_temp_aug = plt.Circle(obs_pos, obstacle_circle_dict[obs_pos]['radius'], linewidth=0.5, linestyle='-',
                                     edgecolor='black', fill=False)
        ax.add_patch(circle_temp_aug)
        ax.add_patch(circle_temp_true)



    for i in range(len(waypoint_list)):
        if collision_state[i] == 0:
            circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.3, color='c')
            ax.add_patch(circle_temp_robot)
        else:
            circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.3, color='r')
            ax.add_patch(circle_temp_robot)

            circle_temp_recover = plt.Circle([collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][0],
                                              collision_info_dict_adjust[tuple(waypoint_list[i])]['recovery_point'][1]],
                                              0.3, color='pink')

            ax.add_patch(circle_temp_recover)



    # plt.plot(time_list, traj_x, 'b', linestyle='--')
    # plt.plot(time_list, traj_y, 'r', linestyle='--')
    # plt.xlim([0.0, time_list[-1]])
    # plt.show()
    #
    # plt.plot(time_list, vel_x, 'b', linestyle='--')
    # plt.plot(time_list, vel_y, 'r', linestyle='--')
    # plt.xlim([0.0, time_list[-1]])
    # plt.show()
    #
    # plt.plot(time_list, acc_x, 'b', linestyle='--')
    # plt.plot(time_list, acc_y, 'r', linestyle='--')
    # plt.xlim([0.0, time_list[-1]])
    # plt.show()

    num_collision = 0
    for c_s in collision_state:
        if c_s == 1:
            num_collision += 1



    # calculate minimal obstacle distance

    # generate obstacle dictionary without collided obstacles

    obstalce_circle_dict_1 = deepcopy(obstacle_circle_dict)
    for pos_c in collision_info_dict_adjust.keys():
        obs_c = collision_info_dict_adjust[pos_c]['obs_center']
        if obs_c in obstalce_circle_dict_1:
            del(obstalce_circle_dict_1[obs_c])

    # print obstalce_circle_dict_1

    dis_obs_list = []
    for pos_traj in zip(traj_x, traj_y):
        dist_temp = np.inf
        for pos_obs in obstalce_circle_dict_1.keys():
            # obs_radius = obstalce_circle_dict_1[pos_obs]['radius']
            dist_temp = min([dist_temp, math.hypot(pos_traj[0] - pos_obs[0], pos_traj[1] - pos_obs[1])])

        dis_obs_list.append(dist_temp)
    print 'computational time:', comp_time1 #+ comp_time2
    print 'trajectory generation time:', comp_time2
    # print 'ctrl energy:', ctrl_energy
    print 'trajectory time:', Tmax
    print 'path length:', path_length
    print 'minimal distance to obstacle', min(dis_obs_list)

    plt.plot(start[0], start[1], "og")
    plt.plot(goal[0], goal[1], "sg")
    plt.plot([start[0], goal[0]], [start[1], goal[1]], 'k', linestyle='--')
    plt.plot(path_x, path_y, 'b', linestyle='--')
    plt.plot(traj_x, traj_y, 'b')

    plt.xlim([-3.0, 3.0])
    plt.ylim([-3.0, 3.0])
    ti = np.arange(-3.0, 3.0 + 1.0, 0.5)
    plt.xticks(ti)
    plt.yticks(ti)
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    # df = pd.DataFrame(columns=('time_line','pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y'))
    #
    # df['time_line'] = time_list
    # df['pos_x'] = traj_x
    # df['pos_y'] = traj_y
    # df['vel_x'] = vel_x
    # df['vel_y'] = vel_y
    # df['acc_x'] = acc_x
    # df['acc_y'] = acc_y
    #
    # df.to_csv('A:\MyDropbox\ARQ_simulation_result\Trajectories\contact_prior\experiment_env\Traj6.csv')


if __name__ == '__main__':
    main()