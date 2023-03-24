import matplotlib.pyplot as plt
from Waypoint_generation_CI.waypoint_generation import path_generation
from Trajectory_generation.trajectory_profile_generation_CI import trajectory_pattern_generation
from Trajectory_generation.trajectory_generation import trajGenerator
from Trajectory_generation.trajectory_generation_uniform import trajGenerator as trajGenerator_uni
import numpy as np
import math
import time
import pandas as pd
from copy import deepcopy

def main():
    min_x = -10.0
    min_y = -10.0
    max_x = 10.0
    max_y = 10.0
    resolution = 0.5
    obs_radius = 0.58  # (0.028 + 0.03) m
    start = (-8.0, -8.0)
    goal = (8.0, 8.0)
    v_max = 3.0  # m/s
    a_max = 4.0  # m/s^2
    vel_recover_prop = 0.0
    pos_recover_prop = 0.36
    T_recover = 2.5

    obstalce_circle_dict ={(-5.25, 5.25): {'ind': 12, 'radius': 0.58}, (-7.25, 1.75): {'ind': 4, 'radius': 0.58},
(5.75, -6.75): {'ind': 24, 'radius': 0.58}, (-8.75, 5.75): {'ind': 22, 'radius': 0.58},
(-4.75, -6.75): {'ind': 2, 'radius': 0.58}, (-8.75, -1.25): {'ind': 9, 'radius': 0.58},
(7.75, 1.75): {'ind': 27, 'radius': 0.58}, (-1.25, 6.75): {'ind': 25, 'radius': 0.58},
(4.25, 5.25): {'ind': 1, 'radius': 0.58}, (3.25, -8.75): {'ind': 18, 'radius': 0.58},
(8.25, -0.75): {'ind': 16, 'radius': 0.58}, (-2.25, -7.25): {'ind': 14, 'radius': 0.58},
(-1.25, -4.25): {'ind': 10, 'radius': 0.58}, (6.25, 3.75): {'ind': 13, 'radius': 0.58},
(3.75, 8.25): {'ind': 11, 'radius': 0.58}, (-3.25, 2.25): {'ind': 29, 'radius': 0.58},
(-6.75, 8.75): {'ind': 6, 'radius': 0.58}, (2.25, -5.75): {'ind': 21, 'radius': 0.58},
(-8.25, -4.25): {'ind': 20, 'radius': 0.58}, (1.25, 0.75): {'ind': 3, 'radius': 0.58},
(1.75, -3.25): {'ind': 0, 'radius': 0.58}, (3.25, -1.25): {'ind': 26, 'radius': 0.58},
(1.25, 8.25): {'ind': 8, 'radius': 0.58}, (-6.25, -2.25): {'ind': 23, 'radius': 0.58},
(8.25, -5.25): {'ind': 28, 'radius': 0.58}, (4.25, 2.25): {'ind': 5, 'radius': 0.58},
(0.25, 4.75): {'ind': 15, 'radius': 0.58}, (5.75, -1.75): {'ind': 7, 'radius': 0.58},
(-3.75, 7.75): {'ind': 17, 'radius': 0.58}, (-2.75, -0.25): {'ind': 19, 'radius': 0.58}}



    starttime = time.time()
    path_gen = path_generation(min_x, max_x, min_y, max_y, obstalce_circle_dict, resolution, start, goal)
    waypoint_list, collision_state, collision_info_dict = path_gen.waypoint_generation()
    endtime = time.time()
    comp_time1 = (endtime - starttime) * 1000.0
    print 'Found goal'

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
        if collision_state[ind] == 1:
            traj_temp = trajGenerator(waypoints_temp, vel_st, vel_ed, max_vel=v_max, gamma=2e4)
        else:
            # traj_temp = trajGenerator_uni(waypoints_temp, vel_st, vel_ed, max_vel=v_max, max_acc=a_max)
            traj_temp = trajGenerator(waypoints_temp, vel_st, vel_ed, max_vel=v_max, gamma=2e4)
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
    vel_norm = []
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

        for t in np.arange(0.0, Tmax_seg, 0.01):
            des_state = traj_seg.get_des_state(t)
            traj_x.append(des_state.pos[0])
            traj_y.append(des_state.pos[1])
            vel_x.append(des_state.vel[0])
            vel_y.append(des_state.vel[1])
            acc_x.append(des_state.acc[0])
            acc_y.append(des_state.acc[1])
            time_list.append(t + t_start)
            ctrl_energy += (des_state.acc[0] ** 2 + des_state.acc[1] ** 2) * 0.01
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

    # plot
    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    for obs_pos in obstalce_circle_dict.keys():
        circle_temp_true = plt.Circle(obs_pos, obs_radius - 0.28, color='grey')
        circle_temp_aug = plt.Circle(obs_pos, obstalce_circle_dict[obs_pos]['radius'], linewidth=0.5, linestyle='-',
                                     edgecolor='black', fill=False)
        ax.add_patch(circle_temp_aug)
        ax.add_patch(circle_temp_true)

    plt.plot(start[0], start[1], "og")
    plt.plot(goal[0], goal[1], "sg")
    plt.plot([start[0], goal[0]], [start[1], goal[1]], 'k', linestyle='--')
    plt.plot(path_x, path_y, 'b', linestyle='--')
    plt.plot(traj_x, traj_y, 'b')

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


    # plt.plot(time_list, vel_x, 'b', linestyle='--')
    # plt.plot(time_list, vel_y, 'r', linestyle='--')
    # plt.xlim([0.0, time_list[-1]])
    # plt.show()
    #
    # plt.plot(time_list, acc_x, 'b', linestyle='--')
    # plt.plot(time_list, acc_y, 'r', linestyle='--')
    # plt.xlim([0.0, time_list[-1]])
    # plt.show()

    df = pd.DataFrame(columns=('time_line','pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y'))

    df['time_line'] = time_list
    df['pos_x'] = traj_x
    df['pos_y'] = traj_y
    df['vel_x'] = vel_x
    df['vel_y'] = vel_y
    df['acc_x'] = acc_x
    df['acc_y'] = acc_y

    # df.to_csv('A:\MyDropbox\ARQ_simulation_result\Trajectories\contact_prior\env30\Traj5.csv')


    num_collision = 0
    for c_s in collision_state:
        if c_s == 1:
            num_collision += 1



# calculate minimal obstacle distance

    # generate obstacle dictionary without collided obstacles

    obstalce_circle_dict_1 = deepcopy(obstalce_circle_dict)
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

    print 'planning time:', comp_time1
    print 'trajectory generation time:', comp_time2
    print 'trajectory time:', Tmax
    print 'path length:', path_length
    print 'minimal distance to obstacle', min(dis_obs_list)

if __name__ == '__main__':
    main()