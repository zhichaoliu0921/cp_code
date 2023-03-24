"""
Trajectory planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)
author: Zhouyu Lu
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import LineString
from shapely.geometry import Point

from Trajectory_generation.trajectory_generation import trajGenerator

show_animation = False


class RRT(object):
    """
    Class for RRT planning
    """

    class Node(object):
        """
        RRT Node
        """
        def __init__(self, x, y, vx, vy, collision_state, sample_num):
            self.x_pre = x
            self.y_pre = y
            self.x_post = x
            self.y_post = y
            self.vx_pre = vx
            self.vy_pre = vy
            self.vx_post = vx
            self.vy_post = vy
            self.collision_state = collision_state
            self.sample_num = sample_num
            self.goal_flag = False
            self.obstacle = {}
            self.traj_info = None
            self.traj_x = []
            self.traj_y = []
            self.parent = None

    def __init__(self,
                 start_pos,
                 goal_pos,
                 start_vel,
                 goal_vel,
                 min_x,
                 min_y,
                 max_x,
                 max_y,
                 min_vx,
                 min_vy,
                 max_vx,
                 max_vy,
                 min_ax,
                 min_ay,
                 max_ax,
                 max_ay,
                 collision_state_start,
                 collision_state_goal,
                 obstacle_circle_dict,
                 resolution_pos,
                 resolution_vel,
                 goal_area_block,
                 min_expand_t=0.5,
                 max_expand_t=5.0,
                 min_expand_d=0.1,
                 max_expand_d=3.0,
                 recover_time=2.0,
                 rho_t=1.0,
                 rho_c=1.0,
                 goal_sample_rate=5,
                 max_iter=400):
        """
        Setting Parameter
        start:Start Position [x,y, vx, vy, collision_state]
        goal:Goal Position [x,y, vx, vy, collision_state]
        obstacleList(circle):obstacle Positions [[x,y,size], ...]
        obstacleList(line_segment):obstacle Positions [[[x1,y1], [x2, y2]], ...]
        randArea:Random Sampling Area [min,max]
        randvelocity:Random Sampling velocity [v_min,v_max]
        """
        self.start = self.Node(start_pos[0], start_pos[1], start_vel[0], start_vel[1], collision_state_start, 0.0)
        self.end = self.Node(goal_pos[0], goal_pos[1], goal_vel[0], goal_vel[1], collision_state_goal, 1.0)
        self.goal_sample_rate = goal_sample_rate
        self.resolution_pos = resolution_pos
        self.resolution_vel = resolution_vel
        self.rho_t = rho_t
        self.rho_c = rho_c
        self.max_iter = max_iter
        self.node_list = []
        self.rnd_state_dict = dict()
        self.node_state_dict = dict()

        self.min_x, self.min_y = min_x, min_y
        self.max_x, self.max_y = max_x, max_y
        self.min_vx, self.min_vy = min_vx, min_vy
        self.max_vx, self.max_vy = max_vx, max_vy
        self.min_ax, self.min_ay = min_ax, min_ay
        self.max_ax, self.max_ay = max_ax, max_ay

        self.recover_time = recover_time
        self.goal_area_block = goal_area_block

        self.x_width = round((self.max_x - self.min_x) / self.resolution_pos)
        self.y_width = round((self.max_y - self.min_y) / self.resolution_pos)
        self.vx_width = round((self.max_vx - self.min_vx) / self.resolution_vel)
        self.vy_width = round((self.max_vy - self.min_vy) / self.resolution_vel)

        self.min_expand_t = min_expand_t
        self.min_expand_d = min_expand_d
        self.max_expand_t = max_expand_t
        self.max_expand_d = max_expand_d
        self.obstacle_circle_dict = obstacle_circle_dict

    def planning(self, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """
        self.node_list = [self.start]
        self.node_state_dict[(self.start.x_post, self.start.y_post, self.start.vx_post, self.start.vy_post,
                              self.start.collision_state)] = set()

        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd_node = self.get_random_node()
            nearest_ind, nearest_traj_result = self.get_nearest_node_index(self.node_list, rnd_node)
            if nearest_ind is None:
                continue

            # if the robot collide with the obstacle
            nearest_node = self.node_list[nearest_ind]

            rnd_node_state = (rnd_node.x_post, rnd_node.y_post, rnd_node.vx_post, rnd_node.vy_post,
                              rnd_node.collision_state)
            nearest_node_state = (nearest_node.x_post, nearest_node.y_post, nearest_node.vx_post, nearest_node.vy_post,
                                  nearest_node.collision_state)
            if rnd_node_state not in self.rnd_state_dict:
                self.rnd_state_dict[rnd_node_state] = set(nearest_node_state)
            elif nearest_node_state not in self.rnd_state_dict[rnd_node_state]:
                self.rnd_state_dict[rnd_node_state].add(nearest_node_state)
            else:
                continue

            new_node = self.steer(nearest_node, rnd_node, nearest_traj_result, run_collision_modification=True,
                                  max_extend_length=self.max_expand_d,
                                  min_extend_T=self.min_expand_t, max_extend_T=self.max_expand_t)

            # if self.check_visibility_circle([self.node_list[-1].x, self.node_list[-1].y],
            #                                 [self.end.x, self.end.y]):

            if new_node is None:
                continue

            self.node_list.append(new_node)

            goal_node = self.Node(self.end.x_pre, self.end.x_pre, self.end.vx_pre, self.end.vy_pre,
                                  self.end.collision_state, 1.0)
            traj_final = self.connect_node(self.node_list[-1], goal_node)
            cost_final = traj_final.cost + self.rho_t * traj_final.TS[-1]
            traj_goal_info = {'cost': cost_final, 'traj_info': traj_final}
            goal_node = self.steer(self.node_list[-1], goal_node, traj_goal_info, run_collision_modification=False,
                                   collision_dist_ext=0.3)

            if animation:
                if new_node.collision_state == 1:
                    plt.plot(new_node.x_post, new_node.y_post, "xr")
                else:
                    plt.plot(new_node.x_post, new_node.y_post, "xc")
                plt.plot(new_node.traj_x, new_node.traj_y, color="lightgrey")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if i % 5 == 0:
                    plt.pause(0.001)

            if goal_node:
                print goal_node.x_pre, goal_node.y_pre
                print 'find path! '
                print 'goal state:', self.node_list[-1].x_post, self.node_list[-1].y_post, self.node_list[-1].vx_post, \
                    self.node_list[-1].vy_post, self.node_list[-1].collision_state
                self.end.traj_info = goal_node.traj_info
                self.end.goal_flag = True
                self.end.parent = self.node_list[-1]
                print("Iter:", i, ", number of nodes:", len(self.node_list))
                return self.generate_final_course(self.node_list[-1])

        return None, None, None, None  # cannot find path

    def connect_node(self, from_node, to_node):
        path_length = math.hypot(to_node.x_pre - from_node.x_post, to_node.y_pre - from_node.y_post)
        if path_length < self.min_expand_d:
            return None
        waypoints = np.array([[from_node.x_post, from_node.y_post], [to_node.x_pre, to_node.y_pre]])
        vel_st = np.array([from_node.vx_post, from_node.vy_post])
        vel_ed = np.array([to_node.vx_pre, to_node.vy_pre])
        traj = trajGenerator(waypoints, vel_st, vel_ed, max_vel=self.max_vx, gamma=1e1)
        return traj

    def generate_final_course(self, goal_node):
        path = [[self.end.x_post, self.end.y_post]]
        collision_state_list = [self.end.collision_state]
        traj_list = [self.end.traj_info]
        node = goal_node
        collision_info_dict = dict()
        while node.parent is not None:
            path.append([node.x_post, node.y_post])
            collision_state_list.append(node.collision_state)
            if node.collision_state == 1:
                collision_info_dict[(node.x_pre, node.y_pre)] = {'obs_center': list(node.obstacle.keys())[0],
                                                                 'obs_radius': list(node.obstacle.values())[0]['radius'],
                                                                 'collision_point': [node.x_pre, node.y_pre, 0.0],
                                                                 'recovery_point': [node.x_pre, node.y_pre, 0.0]}
            if node.traj_info is not None:
                traj_list.append(node.traj_info)

            node = node.parent

        path.append([node.x_post, node.y_post])
        collision_state_list.append(node.collision_state)
        if node.collision_state == 1:
            collision_info_dict[(node.x_pre, node.y_pre)] = {'obs_center': list(node.obstacle.keys())[0],
                                                             'obs_radius': list(node.obstacle.values())[0]['radius'],
                                                             'collision_point': [node.x_pre, node.y_pre, 0.0],
                                                             'recovery_point': [node.x_pre, node.y_pre, 0.0]}
        if node.traj_info is not None:
            traj_list.append(node.traj_info)

        return path, collision_state_list, traj_list, collision_info_dict

    # def calc_dist_to_goal(self, x, y):
    #     dx = abs(x - self.end.x)
    #     dy = abs(y - self.end.y)
    #     return max(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_ix = random.randint(0, int(self.x_width))
            rand_iy = random.randint(0, int(self.y_width))
            rand_ivx = random.randint(0, int(self.vx_width))
            rand_ivy = random.randint(0, int(self.vy_width))
            rand_x = self.calc_grid_position(rand_ix, self.min_x)
            rand_y = self.calc_grid_position(rand_iy, self.min_y)
            rand_vx = self.calc_grid_velocity(rand_ivx, self.min_vx)
            rand_vy = self.calc_grid_velocity(rand_ivy, self.min_vy)

            if rand_x < self.min_x:
                rand_x = self.min_x
            elif rand_x > self.max_x:
                rand_x = self.max_x

            if rand_y < self.min_y:
                rand_y = self.min_y
            elif rand_y > self.max_y:
                rand_y = self.max_y

            if rand_vx < self.min_vx:
                rand_vx = self.min_vx
            elif rand_vx > self.max_vx:
                rand_vx = self.max_vx

            if rand_vy < self.min_vy:
                rand_vy = self.min_vy
            elif rand_vy > self.max_vy:
                rand_vy = self.max_vy

            rnd = self.Node(rand_x, rand_y, rand_vx, rand_vy, 0, random.uniform(0, 1.0))

        else:  # goal point sampling
            rnd = self.Node(self.end.x_post, self.end.y_post, self.end.vx_post, self.end.vy_post,
                            self.end.collision_state, 1.0)
            rnd.goal_flag = True
        return rnd

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution_pos + min_position
        return pos

    def calc_grid_velocity(self, index, min_velocity):
        """
        calc grid velocity
        :param index:
        :param min_velocity
        :return:
        """
        vel = index * self.resolution_vel + min_velocity
        return vel

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution_pos)

    def calc_vxy_index(self, velocity, min_vel):
        return round((velocity - min_vel) / self.resolution_vel)

    def steer(self, from_node, to_node, traj_result, run_collision_modification, max_extend_length=float("inf"),
              min_extend_T=0.0, max_extend_T=float("inf"), collision_dist_ext=0.0):
        traj_info = traj_result['traj_info']
        Tmax = traj_info.TS[-1]

        if Tmax < min_extend_T:
            Tmax = min_extend_T

        if Tmax > max_extend_T:
            Tmax = max_extend_T

        traj_x = []
        traj_y = []
        vel_x = []
        vel_y = []
        acc_x = []
        acc_y = []
        time = []
        for t in np.arange(0.0, Tmax, 0.05):
            des_state = traj_info.get_des_state(t)
            path_extend_x = des_state.pos[0] - from_node.x_post
            path_extend_y = des_state.pos[1] - from_node.y_post
            path_extend_length = math.hypot(path_extend_x, path_extend_y)
            if path_extend_length > max_extend_length:
                break
            traj_x.append(des_state.pos[0])
            traj_y.append(des_state.pos[1])
            vel_x.append(des_state.vel[0])
            vel_y.append(des_state.vel[1])
            acc_x.append(des_state.acc[0])
            acc_y.append(des_state.acc[1])
            time.append(t)

        if not traj_x:
            return None

        if time[-1] < min_extend_T:
            return None

        # within the space
        if min(traj_x) < self.min_x or max(traj_x) > self.max_x:
            return None

        if min(traj_y) < self.min_y or max(traj_y) > self.max_y:
            return None

        # kino-dynamic constraint
        if min(acc_x) < self.min_ax or max(acc_x) > self.max_ax:
            return None

        if min(acc_y) < self.min_ay or max(acc_y) > self.max_ay:
            return None

        if min(vel_x) < self.min_vx or max(vel_x) > self.max_vx:
            return None

        if min(vel_y) < self.min_vy or max(vel_y) > self.max_vy:
            return None

        # check collision
        safe_flag = True
        for i in range(len(time)):
            pos_x_temp = traj_x[i]
            pos_y_temp = traj_y[i]
            dist_obs_dict = self.generate_dictionary_distance_to_obstacles([pos_x_temp, pos_y_temp])

            d_min = min(dist_obs_dict.values())
            key_min = min(dist_obs_dict, key=dist_obs_dict.get)

            if d_min <= collision_dist_ext:
                safe_flag = False
                break

        if not safe_flag:
            if run_collision_modification:
                obs_pos = key_min
                if obs_pos in from_node.obstacle:
                    return None
                else:
                    # adjust collision primitives
                    p1 = [from_node.x_post, from_node.y_post]
                    vel_1 = [from_node.vx_post, from_node.vy_post]
                    obs_radius = self.obstacle_circle_dict[obs_pos]['radius']
                    vec_p1_center = np.array([obs_pos[0] - p1[0], obs_pos[1] - p1[1]])
                    vec_p1_center_norm = np.linalg.norm(vec_p1_center)
                    unit_p1_center = vec_p1_center / vec_p1_center_norm
                    vec_p1_collision = vec_p1_center - unit_p1_center * obs_radius
                    vec_p1_recovery = vec_p1_center - unit_p1_center * math.sqrt(2) * obs_radius
                    pos_temp_c = [vec_p1_collision[0] + p1[0], vec_p1_collision[1] + p1[1]]
                    pos_temp_r = [vec_p1_recovery[0] + p1[0], vec_p1_recovery[1] + p1[1]]
                    move_vec = np.array([pos_temp_c[0] - p1[0], pos_temp_c[1] - p1[1]])
                    move_vec_norm = np.linalg.norm(move_vec)
                    unit_move_vec = move_vec / move_vec_norm
                    vel_st_proj = np.dot(unit_move_vec, vel_1)
                    vel_set = self.max_vx

                    if move_vec_norm < (vel_set**2 - vel_st_proj**2) / (2 * self.max_ax):
                        vel_2_norm = vel_st_proj**2 + 2 * self.max_ax * move_vec_norm
                    else:
                        vel_2_norm = vel_set

                    vel_2 = vel_2_norm * unit_move_vec
                    waypoints = np.array([p1, pos_temp_c])
                    vel_1 = np.array(vel_1)
                    vel_2 = np.array(vel_2)
                    traj_collision = trajGenerator(waypoints, vel_1, vel_2, max_vel=self.max_vx, gamma=1e1)
                    Tmax = traj_collision.TS[-1]
                    traj_x = []
                    traj_y = []
                    vel_x = []
                    vel_y = []
                    acc_x = []
                    acc_y = []
                    time = []
                    for t in np.arange(0.0, Tmax, 0.05):
                        des_state = traj_collision.get_des_state(t)
                        traj_x.append(des_state.pos[0])
                        traj_y.append(des_state.pos[1])
                        vel_x.append(des_state.vel[0])
                        vel_y.append(des_state.vel[1])
                        acc_x.append(des_state.acc[0])
                        acc_y.append(des_state.acc[1])
                        time.append(t)

                    if not traj_x:
                        return None

                    if time[-1] < min_extend_T:
                        return None

                    # within the space
                    if min(traj_x) < self.min_x or max(traj_x) > self.max_x:
                        return None

                    if min(traj_y) < self.min_y or max(traj_y) > self.max_y:
                        return None

                    # kino-dynamic constraint
                    if min(acc_x) < self.min_ax or max(acc_x) > self.max_ax:
                        return None

                    if min(acc_y) < self.min_ay or max(acc_y) > self.max_ay:
                        return None

                    if min(vel_x) < self.min_vx or max(vel_x) > self.max_vx:
                        return None

                    if min(vel_y) < self.min_vy or max(vel_y) > self.max_vy:
                        return None

                    # check collision
                    for i in range(len(time)):
                        pos_x_temp = traj_x[i]
                        pos_y_temp = traj_y[i]
                        dist_obs_dict = self.generate_dictionary_distance_to_obstacles([pos_x_temp, pos_y_temp])

                        d_min_1 = min(dist_obs_dict.values())
                        key_min_1 = min(dist_obs_dict, key=dist_obs_dict.get)

                        if d_min_1 <= 0.0:
                            return None

                    energy_loss = 0.5 * vel_2_norm ** 2
                    new_node = self.Node(traj_x[-1], traj_y[-1], vel_x[-1], vel_y[-1], 1, to_node.sample_num)
                    new_node.x_post = pos_temp_r[0]
                    new_node.y_post = pos_temp_r[1]
                    new_node.vx_post = 0.0
                    new_node.vx_post = 0.0
                    new_node.sample_num = to_node.sample_num
                    new_node.obstacle[obs_pos] = self.obstacle_circle_dict[obs_pos]
                    traj_collision.TS[-1] = time[-1]
                    traj_collision.cost = traj_collision.get_cost_from_coeffs(traj_collision.TS[1:], traj_collision.coeffs)
                    new_node.sample_num = to_node.sample_num
                    new_node.traj_info = {'cost': traj_collision.cost +
                                                  self.rho_t * (traj_collision.TS[-1] + self.recover_time) +
                                                  self.rho_c * energy_loss,
                                          'traj_info': traj_collision}
                    new_node.traj_x = traj_x
                    new_node.traj_y = traj_y
                    new_node.parent = from_node
            else:
                return None
        else:
            new_node = self.Node(traj_x[-1], traj_y[-1], vel_x[-1], vel_y[-1], 0, to_node.sample_num)
            traj_info.TS[-1] = time[-1]
            traj_info.cost = traj_info.get_cost_from_coeffs(traj_info.TS[1:], traj_info.coeffs)
            new_node.sample_num = to_node.sample_num
            new_node.traj_info = {'cost': traj_info.cost + self.rho_t * traj_info.TS[-1],
                                  'traj_info': traj_info}
            new_node.traj_x = traj_x
            new_node.traj_y = traj_y
            new_node.parent = from_node

        new_node_state = (new_node.x_post, new_node.x_post, new_node.vx_post, new_node.vy_post,
                          new_node.collision_state)
        from_node_state = (from_node.x_post, from_node.x_post, from_node.vx_post, from_node.vy_post,
                           from_node.collision_state)
        if new_node_state not in self.node_state_dict:
            self.node_state_dict[new_node_state] = set(from_node_state)
        elif from_node_state not in self.node_state_dict[new_node_state]:
            self.node_state_dict[new_node_state].add(from_node_state)
        else:
            return None

        return new_node

    def get_nearest_node_index(self, node_list, rnd_node):
        traj_dict = {}
        for node in node_list:
            if node.sample_num < rnd_node.sample_num:
                traj_temp = self.connect_node(node, rnd_node)
                if traj_temp is not None:
                    node_index = node_list.index(node)
                    traj_dict[node_index] = {'cost': traj_temp.cost + self.rho_t * traj_temp.TS[-1],
                                             'traj_info': traj_temp}
        if traj_dict:
            min_item = min(traj_dict.items(), key=lambda kv: kv[1]['cost'])
            min_ind = min_item[0]
            min_val = min_item[1]
            return min_ind, min_val
        else:
            return None, None

    def generate_dictionary_distance_to_obstacles(self, pos_c):
        distance_dictionary = dict()
        for pos_obs in self.obstacle_circle_dict.keys():
            radius = self.obstacle_circle_dict[pos_obs]['radius']
            distance_temp = self.getDisPointToCircle(pos_c, pos_obs, radius)
            distance_dictionary[pos_obs] = distance_temp
        return distance_dictionary

    @staticmethod
    def getDisPointToLine(point, line_p1, line_p2):
        """
        @point, line_p1, line_p2 : [x, y, z]
        """
        x0 = point[0]
        y0 = point[1]
        z0 = 0  # point[2]

        x1 = line_p1[0]
        y1 = line_p1[1]
        z1 = 0  # line_p1[2]

        x2 = line_p2[0]
        y2 = line_p2[1]
        z2 = 0  # line_p2[2]

        cross = (x2 - x1) * (x0 - x1) + (y2 - y1) * (y0 - y1)
        if (cross <= 0):
            return math.sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1))
        d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
        if (cross >= d2):
            return math.sqrt((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2))
        r = cross / d2
        px = x1 + (x2 - x1) * r
        py = y1 + (y2 - y1) * r
        return math.sqrt((x0 - px) * (x0 - px) + (py - y0) * (py - y0))

    @staticmethod
    def getDisPointToCircle(point, center, radius):
        """
        @point, center : [x, y, z]
        @ radius, r(float)
        """
        x0 = point[0]
        y0 = point[1]
        z0 = 0  # point[2]

        xc = center[0]
        yc = center[1]
        zc = 0  # line_p1[2]

        d_c = math.hypot(xc - x0, yc - y0)
        return d_c - radius

    @staticmethod
    def check_visibility_circle(self, point1, point2, dis_vis):  # return false-collision, true-safe
        start_point = (point1[0], point1[1])
        end_point = (point2[0], point2[1])
        line_seg_path = LineString([start_point, end_point])
        for pos_obs in self.obstacle_circle_dict.keys():
            radius = self.obstacle_circle_dict[pos_obs]['radius'] + dis_vis
            circle_obs = Point(pos_obs[0], pos_obs[1]).buffer(radius).boundary
            bool_temp = circle_obs.intersects(line_seg_path)
            if bool_temp:
                return False
        return True