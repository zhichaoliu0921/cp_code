import matplotlib.pyplot as plt
import numpy as np
from Waypoint_generation_CA.rrt_star import RRTStar
from Trajectory_generation.trajectory_generation import trajGenerator
import math
import pandas as pd
from copy import deepcopy

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
    start = (-8.0, -8.0)
    goal = (8.0, 8.0)
    v_max = 1.5  # m/s
    a_max = 2.0  # m/s^2

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


    dis_potential_bound = 1.5 ** 2 / (2 * a_max)
    dis_vis = 2.2



    starttime = time.time()
    rrt_planner = RRTStar(start, goal, min_x, min_y, max_x, max_y, resolution, obstalce_circle_dict,
                          dis_potential_bound)
    path = rrt_planner.planning(animation=False)
    endtime = time.time()

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        # Draw final path
        # plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

        waypoint_list = rrt_planner.waypoint_generation(path, dis_vis)
        comp_time1 = (endtime - starttime) * 1000.0
        # print 'generate path with ', comp_time1, 'ms. '

        path_x = []
        path_y = []
        for i in range(len(waypoint_list)):
            path_x.append(waypoint_list[i][0])
            path_y.append(waypoint_list[i][1])

        waypoints = np.array(waypoint_list)
        vel_st = [0.0, 0.0, 0.0]
        vel_ed = [0.0, 0.0, 0.0]

        starttime = time.time()
        traj = trajGenerator(waypoints, vel_st, vel_ed, max_vel=v_max, gamma=1e1)
        endtime = time.time()
        comp_time2 = (endtime - starttime) * 1000.0
        # print 'generate trajectory with ', comp_time2, 'ms. '

        # initialise simulation with given controller and trajectory
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
        for t in np.arange(0.0, Tmax, 0.01):
            des_state = traj.get_des_state(t)
            traj_x.append(des_state.pos[0])
            traj_y.append(des_state.pos[1])
            vel_x.append(des_state.vel[0])
            vel_y.append(des_state.vel[1])
            vel_norm.append(math.sqrt(des_state.vel[0] ** 2 + des_state.vel[1] ** 2))
            acc_x.append(des_state.acc[0])
            acc_y.append(des_state.acc[1])
            time_list.append(t)
            ctrl_energy += (des_state.acc[0] ** 2 + des_state.acc[1] ** 2) * 0.01
            path_length += math.sqrt((des_state.pos[0] - pos_pre[0]) ** 2 + (des_state.pos[1] - pos_pre[1]) ** 2)
            pos_pre = [des_state.pos[0], des_state.pos[1]]

        obstalce_circle_dict_1 = deepcopy(obstalce_circle_dict)

        # print obstalce_circle_dict_1

        dis_obs_list = []
        for pos_traj in zip(traj_x, traj_y):
            dist_temp = np.inf
            for pos_obs in obstalce_circle_dict_1.keys():
                # obs_radius = obstalce_circle_dict_1[pos_obs]['radius']
                dist_temp = min([dist_temp, math.hypot(pos_traj[0] - pos_obs[0], pos_traj[1] - pos_obs[1])])

            dis_obs_list.append(dist_temp)

        print 'planning time:', comp_time1 #+ comp_time2
        print 'trajectory generation time:', comp_time2
        # print 'ctrl energy:', ctrl_energy
        print 'trajectory time:', Tmax
        print 'path length:', path_length
        print 'minimal distance to obstacle', min(dis_obs_list)


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

        plt.plot(path_x, path_y, 'b', linestyle='--')
        plt.plot(traj_x, traj_y, 'b')

        for i in range(len(waypoint_list)):
            circle_temp_robot = plt.Circle([waypoint_list[i][0], waypoint_list[i][1]], 0.28, color='c')
            ax.add_patch(circle_temp_robot)

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


        df = pd.DataFrame(columns=('pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y'))

        df['pos_x'] = traj_x
        df['pos_y'] = traj_y
        df['vel_x'] = vel_x
        df['vel_y'] = vel_y
        df['acc_x'] = acc_x
        df['acc_y'] = acc_y


if __name__ == '__main__':
    main()