import matplotlib.pyplot as plt
import numpy as np
from Waypoint_generation_CA.A_star_experiment import AStarPlanner
from Trajectory_generation.trajectory_profile_generation_CA import trajectory_pattern_generation
from Trajectory_generation.trajectory_generation import trajGenerator
import math
import pandas as pd
from copy import deepcopy

def main():
    import time
    print(__file__ + " start!!")

    # start and goal position
    min_x = -2.4
    min_y = -1.3
    max_x = 2.7
    max_y = 1.3
    resolution = 0.1
    obs_radius = 0.43  # (0.15 + 0.28) m
    start = (-2.0, 0.0)
    goal = (2.0, -0.2)
    v_max = 1.5  # m/s
    a_max = 4.0  # m/s^2

    # with 1 collisions
    # typo to fix
    obstacle_circle_dict = {(0.0, 0.0): {'ind': 1, 'radius': 0.43}, (-1.0, 0.87): {'ind': 2, 'radius': 0.43},
                            (1.0, -0.93): {'ind': 3, 'radius': 0.43}, (-1.15, -0.85): {'ind': 0, 'radius': 0.43}}

    dis_potential_bound = 0.5*1.5 * 1.5 ** 2 / (2 * a_max)
    dis_vis = 0.2

    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    for obs_pos in obstacle_circle_dict.keys():
        circle_temp_true = plt.Circle(obs_pos, obs_radius - 0.28, color='grey')
        circle_temp_aug = plt.Circle(obs_pos, obstacle_circle_dict[obs_pos]['radius'], linewidth=0.5, linestyle='-',
                                     edgecolor='black', fill=False)
        ax.add_patch(circle_temp_aug)
        ax.add_patch(circle_temp_true)

    plt.plot(start[0], start[1], "og")
    plt.plot(goal[0], goal[1], "sg")

    starttime = time.time()
    a_star = AStarPlanner(min_x, min_y, max_x, max_y, resolution, obstacle_circle_dict, dis_potential_bound)
    rx, ry = a_star.planning(start[0], start[1], goal[0], goal[1])
    waypoint_list = a_star.waypoint_generation(rx, ry, dis_vis)
    endtime = time.time()
    comp_time1 = (endtime - starttime) * 1000.0
    # print 'generate path with ', comp_time1, 'ms. '

    # print waypoint_list
    path_x = []
    path_y = []
    for i in range(len(waypoint_list)):
        path_x.append(waypoint_list[i][0])
        path_y.append(waypoint_list[i][1])

    waypoints = np.array(waypoint_list)
    vel_st = [0.0, 0.0, 0.0]
    vel_ed = [0.0, 0.0, 0.0]

    starttime = time.time()
    traj = trajGenerator(waypoints, vel_st, vel_ed, max_vel=v_max, gamma=3e1)
    endtime = time.time()
    comp_time2 = (endtime - starttime) * 1000.0
    # print 'generate trajectory with ', comp_time2, 'ms. '

    Tmax = traj.TS[-1]
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
    for t in np.arange(0.0, Tmax, 0.05):
        des_state = traj.get_des_state(t)
        traj_x.append(des_state.pos[0])
        traj_y.append(des_state.pos[1])
        vel_x.append(des_state.vel[0])
        vel_y.append(des_state.vel[1])
        vel_norm.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
        acc_x.append(des_state.acc[0])
        acc_y.append(des_state.acc[1])
        time_list.append(t)
        ctrl_energy += (des_state.acc[0]**2 + des_state.acc[1]**2) * 0.05
        path_length += math.sqrt((des_state.pos[0] - pos_pre[0]) ** 2 + (des_state.pos[1] - pos_pre[1]) ** 2)
        pos_pre = [des_state.pos[0], des_state.pos[1]]

    path_x = []
    path_y = []
    for i in range(len(waypoint_list)):
        path_x.append(waypoint_list[i][0])
        path_y.append(waypoint_list[i][1])



    for i in range(len(waypoint_list)):
        circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.3, color='c')
        ax.add_patch(circle_temp_robot)

    # print 'waypoint list:', waypoint_list

    # plt.plot(time_list, traj_x, 'b', linestyle='--', label='x')
    # plt.plot(time_list, traj_y, 'r', linestyle='--', label='y')
    # plt.xlim([0.0, time_list[-1]])
    # plt.grid(True)
    # plt.legend()
    # plt.title("position")
    # plt.show()


    # plt.plot(time_list, vel_x, 'b', linestyle='--', label='x')
    # plt.plot(time_list, vel_y, 'r', linestyle='--', label='y')
    # plt.xlim([0.0, time_list[-1]])
    # plt.grid(True)
    # plt.legend()
    # plt.title("velocity")
    # plt.show()
    #
    # plt.plot(time_list, vel_norm, 'b', linestyle='--')
    # plt.xlim([0.0, time_list[-1]])
    # plt.title("velocity Norm")
    # plt.grid(True)
    # plt.show()
    #
    # plt.plot(time_list, acc_x, 'b', linestyle='--', label='x')
    # plt.plot(time_list, acc_y, 'r', linestyle='--', label='y')
    # plt.xlim([0.0, time_list[-1]])
    # plt.grid(True)
    # plt.legend()
    # plt.title("acceleration")
    # plt.show()

    obstacle_circle_dict_1 = deepcopy(obstacle_circle_dict)

    # print obstacle_circle_dict_1

    dis_obs_list = []
    for pos_traj in zip(traj_x, traj_y):
        dist_temp = np.inf
        for pos_obs in obstacle_circle_dict_1.keys():
            # obs_radius = obstacle_circle_dict_1[pos_obs]['radius']
            dist_temp = min([dist_temp, math.hypot(pos_traj[0] - pos_obs[0], pos_traj[1] - pos_obs[1])])

        dis_obs_list.append(dist_temp)


    print 'planning time:', comp_time1  # + comp_time2
    print 'trajectory generation time:', comp_time2
    # print 'ctrl energy:', ctrl_energy
    print 'trajectory time:', Tmax
    print 'path length:', path_length
    print 'minimal distance to obstacle', min(dis_obs_list)

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


    df = pd.DataFrame(columns=('pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y'))

    df['pos_x'] = traj_x
    df['pos_y'] = traj_y
    df['vel_x'] = vel_x
    df['vel_y'] = vel_y
    df['acc_x'] = acc_x
    df['acc_y'] = acc_y

    # df.to_csv('A:\MyDropbox\ARQ_simulation_result\Trajectories\A_star\experiment_env\Traj31.csv')

if __name__ == '__main__':
    main()