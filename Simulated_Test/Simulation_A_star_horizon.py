import matplotlib.pyplot as plt
import numpy as np
from Waypoint_generation_CA.A_star_horizon import AStarPlanner
from Trajectory_generation.trajectory_profile_generation_CA import trajectory_pattern_generation
from Trajectory_generation.trajectory_generation_horizon import trajGenerator
import math
import pandas as pd
from Mapping.Generate_map_with_horizon import map_gen_horizon

def main():
    import time
    print(__file__ + " start!!")

    # start and goal position
    min_x = -10.0
    min_y = -10.0
    max_x = 10.0
    max_y = 10.0
    resolution = 0.5
    obs_radius = 0.58  # (0.028 + 0.03) m
    # obs_radius = 0.43  # (0.028 + 0.015) m
    start = (-8.0, -8.0)
    goal = (8.0, 8.0)
    v_max = 1.5 * 0.707  # m/s
    a_max = 4.0  # m/s^2
    # pos_recover_prop = 0.366
    # T_recover = 0.19
    # d_ext = 0.2
    h_vis = 5.0
    T_map = 5.0

    # map trial 3
    obstalce_circle_dict ={(5.25, 5.75): {'ind': 8, 'radius': 0.58}, (-6.75, 0.25): {'ind': 24, 'radius': 0.58},
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


    dis_potential_bound = 1.0 * 1.5 ** 2 / (2 * a_max)
    dis_vis = 2.2
    # dis_potential_bound = 0.0
    obstalce_circle_dict_vis = dict()

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
    while math.hypot(start[0] - goal[0], start[1] - goal[1]) > h_vis:
        # generate the visible environment
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
        a_star = AStarPlanner(min_x, min_y, max_x, max_y, resolution, obstalce_circle_dict_vis, dis_potential_bound)
        rx, ry = a_star.planning(start[0], start[1], goal[0], goal[1], h_vis)
        waypoint_list = a_star.waypoint_generation(rx, ry, dis_vis)
        waypoint_list[0] = [start[0], start[1], 0.0]
        endtime = time.time()
        comp_time1 = (endtime - starttime) * 1000.0
        print 'generate path with ', comp_time1, 'ms. '
        path_comp_time += comp_time1

        print waypoint_list
        path_x = []
        path_y = []
        for i in range(len(waypoint_list)):
            path_x.append(waypoint_list[i][0])
            path_y.append(waypoint_list[i][1])

        # file_name = file_direction_name + "\waypoints_" + str(ind_seg) + '.txt'
        # for item in waypoint_list:
        #     str1 = str(item)
        #     with open(file_name, 'a') as file:
        #         file.write(str1)
        #         file.write('\r')

        waypoints = np.array(waypoint_list)

        starttime = time.time()
        traj = trajGenerator(waypoints, vel_st, vel_ed, acc_st, acc_ed, jerk_st, jerk_ed, snap_st, snap_ed,
                             max_vel=v_max, gamma=4e1)
        endtime = time.time()
        comp_time2 = (endtime - starttime) * 1000.0
        print 'generate trajectory with ', comp_time2, 'ms. '
        traj_comp_time += comp_time2

        Tmax = traj.TS[-1]
        print 'time of trajectory:', Tmax

        traj_x_p = []
        traj_y_p = []
        vel_x_p = []
        vel_y_p = []
        vel_norm_p = []
        acc_x_p = []
        acc_y_p = []
        time_list_p = []
        for t in np.arange(0.0, Tmax, 0.01):
            des_state = traj.get_des_state(t)
            traj_x_p.append(des_state.pos[0])
            traj_y_p.append(des_state.pos[1])
            vel_x_p.append(des_state.vel[0])
            vel_y_p.append(des_state.vel[1])
            vel_norm_p.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
            acc_x_p.append(des_state.acc[0])
            acc_y_p.append(des_state.acc[1])
            time_list_p.append(t + sum_T)

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
        for t in np.arange(0.0, T_map, 0.01):
            des_state = traj.get_des_state(t)
            traj_x_temp.append(des_state.pos[0])
            traj_y_temp.append(des_state.pos[1])
            vel_x_temp.append(des_state.vel[0])
            vel_y_temp.append(des_state.vel[1])
            vel_norm_temp.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
            acc_x_temp.append(des_state.acc[0])
            acc_y_temp.append(des_state.acc[1])
            time_list_temp.append(t + sum_T)
            ctrl_energy += (des_state.acc[0]**2 + des_state.acc[1]**2) * 0.01
            path_length += math.sqrt((des_state.pos[0] - pos_pre[0]) ** 2 + (des_state.pos[1] - pos_pre[1]) ** 2)
            pos_pre = [des_state.pos[0], des_state.pos[1]]

        # print 'time related'
        # print traj.TS
        # print T_map, t

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

        ind_seg += 1

        sum_T += T_map

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
        plt.axis("equal")

        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])

        plt.grid(True)
        plt.plot(path_x, path_y, 'b', linestyle='--')
        plt.plot(traj_x_p, traj_y_p, 'b')
        plt.plot(traj_x_e, traj_y_e, 'r')
        #
        for i in range(len(waypoint_list)):
            circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.28, color='c')
            ax.add_patch(circle_temp_robot)

        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])
        plt.grid(True)
        plt.axis("equal")
        plt.show()

        des_state = traj.get_des_state(t)
        start = (des_state.pos[0], des_state.pos[1])
        vel_st = [des_state.vel[0], des_state.vel[1], des_state.vel[2]]
        acc_st = [des_state.acc[0], des_state.acc[1], des_state.acc[2]]
        jerk_st = [des_state.jerk[0], des_state.jerk[1], des_state.jerk[2]]
        snap_st = [des_state.snap[0], des_state.snap[1], des_state.snap[2]]

    # generate the visible environment
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
    a_star = AStarPlanner(min_x, min_y, max_x, max_y, resolution, obstalce_circle_dict_vis, dis_potential_bound)
    rx, ry = a_star.planning(start[0], start[1], goal[0], goal[1], h_vis)
    waypoint_list = a_star.waypoint_generation(rx, ry, dis_vis)
    waypoint_list[0] = [start[0], start[1], 0.0]
    endtime = time.time()
    comp_time1 = (endtime - starttime) * 1000.0
    print 'generate path with ', comp_time1, 'ms. '
    path_comp_time += comp_time1

    print waypoint_list
    path_x = []
    path_y = []
    for i in range(len(waypoint_list)):
        path_x.append(waypoint_list[i][0])
        path_y.append(waypoint_list[i][1])

    # file_name = file_direction_name + "\waypoints_" + str(ind_seg) + '.txt'
    # for item in waypoint_list:
    #     str1 = str(item)
    #     with open(file_name, 'a') as file:
    #         file.write(str1)
    #         file.write('\r')

    waypoints = np.array(waypoint_list)

    starttime = time.time()
    traj = trajGenerator(waypoints, vel_st, vel_ed, acc_st, acc_ed, jerk_st, jerk_ed, snap_st, snap_ed,
                         max_vel=v_max, gamma=4e1)
    endtime = time.time()
    comp_time2 = (endtime - starttime) * 1000.0
    print 'generate trajectory with ', comp_time2, 'ms. '
    traj_comp_time += comp_time2

    Tmax = traj.TS[-1]
    print 'time of trajectory:', Tmax

    traj_x_p = []
    traj_y_p = []
    vel_x_p = []
    vel_y_p = []
    vel_norm_p = []
    acc_x_p = []
    acc_y_p = []
    time_list_p = []
    for t in np.arange(0.0, Tmax, 0.01):
        des_state = traj.get_des_state(t)
        traj_x_p.append(des_state.pos[0])
        traj_y_p.append(des_state.pos[1])
        vel_x_p.append(des_state.vel[0])
        vel_y_p.append(des_state.vel[1])
        vel_norm_p.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
        acc_x_p.append(des_state.acc[0])
        acc_y_p.append(des_state.acc[1])
        time_list_p.append(t + sum_T)

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
    for t in np.arange(0.0, Tmax, 0.01):
        des_state = traj.get_des_state(t)
        traj_x_temp.append(des_state.pos[0])
        traj_y_temp.append(des_state.pos[1])
        vel_x_temp.append(des_state.vel[0])
        vel_y_temp.append(des_state.vel[1])
        vel_norm_temp.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
        acc_x_temp.append(des_state.acc[0])
        acc_y_temp.append(des_state.acc[1])
        time_list_temp.append(t + sum_T)
        ctrl_energy += (des_state.acc[0] ** 2 + des_state.acc[1] ** 2) * 0.01
        path_length += math.sqrt((des_state.pos[0] - pos_pre[0]) ** 2 + (des_state.pos[1] - pos_pre[1]) ** 2)
        pos_pre = [des_state.pos[0], des_state.pos[1]]

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

    for obs_pos in obstalce_circle_dict.keys():
        if obs_pos in obstalce_circle_dict_vis:
            circle_temp_true = plt.Circle(obs_pos, obs_radius - 0.28, color='black')
            circle_temp_aug = plt.Circle(obs_pos, obstalce_circle_dict[obs_pos]['radius'], linewidth=2,
                                         linestyle=':',
                                         edgecolor='orange', fill=False)
        else:
            circle_temp_true = plt.Circle(obs_pos, obs_radius - 0.28, color='grey')
            circle_temp_aug = plt.Circle(obs_pos, obstalce_circle_dict[obs_pos]['radius'], linewidth=2,
                                         linestyle=':',
                                         edgecolor='wheat', fill=False)

        ax.add_patch(circle_temp_aug)
        ax.add_patch(circle_temp_true)

    plt.plot(start[0], start[1], "og")
    plt.plot(goal[0], goal[1], "sg")
    plt.axis("equal")

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    plt.grid(True)
    plt.plot(path_x, path_y, 'b', linestyle='--')
    # plt.plot(traj_x_p, traj_y_p, 'b')
    plt.plot(traj_x_e, traj_y_e, 'r')
    #
    for i in range(len(waypoint_list)):
        circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.28, color='c')
        ax.add_patch(circle_temp_robot)

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.grid(True)
    plt.axis("equal")
    plt.show()

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


    print 'plan time:', path_comp_time
    print 'trajectory generation time:', traj_comp_time
    # print 'ctrl energy:', ctrl_energy
    print 'trajectory time:', sum_T + Tmax
    print 'path length:', path_length


if __name__ == '__main__':
    main()