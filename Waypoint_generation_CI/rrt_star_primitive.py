"""
Trajectory planning Sample Code with RRT* collision accepted Jiaming Zha
author: Zhouyu Lu
"""
import os
import sys

import matplotlib.pyplot as plt
import time
import matplotlib
import math
import numpy as np
import heapq as hq
from copy import deepcopy
from Trajectory_generation.trajectory_generation import trajGenerator
# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\RRT")

try:
    from RRT_primitive import RRT
except ImportError:
    raise

show_animation = True


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y, vx, vy, collision_state, sample_num):
            self.cost = 0.0
            super(RRTStar.Node, self).__init__(x, y, vx, vy, collision_state, sample_num)

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
                 max_iter=50,
                 search_until_max_iter=True):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        super(RRTStar, self).__init__(start_pos, goal_pos, start_vel, goal_vel, min_x, min_y, max_x, max_y, min_vx,
                                      min_vy, max_vx, max_vy, min_ax, min_ay, max_ax, max_ay, collision_state_start,
                                      collision_state_goal, obstacle_circle_dict,  resolution_pos, resolution_vel,
                                      goal_area_block, min_expand_t, max_expand_t, min_expand_d, max_expand_d,
                                      recover_time, rho_t, rho_c, goal_sample_rate, max_iter)
        self.search_until_max_iter = search_until_max_iter
        self.cost_path_time = 0.0
        self.rewire_path_time = 0.0

    def planning(self, animation=True):
        """
        rrt star path planning
        animation: flag for animation on or off .
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

            if new_node is None:
                continue

            starttime1 = time.time()
            prior_inds = self.find_prior_nodes(new_node)
            new_node = self.choose_parent(new_node, prior_inds)
            endtime1 = time.time()

            self.cost_path_time = self.cost_path_time + (endtime1 - starttime1)

            if new_node is None:
                continue

            self.node_list.append(new_node)

            starttime2 = time.time()
            post_inds = self.find_post_nodes(new_node)
            self.rewire(new_node, post_inds)
            endtime2 = time.time()
            self.rewire_path_time = self.rewire_path_time + (endtime2 - starttime2)

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

            if not self.search_until_max_iter:  # if reaches goal
                last_node = self.search_feasible_goal_node()
                print("Iter:", i, ", number of nodes:", len(self.node_list))
                if last_node is not None:
                    return self.generate_final_course(last_node)

        print("reached max iteration")
        print("Iter:", i, ", number of nodes:", len(self.node_list))

        last_node = self.search_feasible_goal_node()
        if last_node is not None:
            return self.generate_final_course(last_node)
        return None, None, None, None  # cannot find path

    def choose_parent(self, new_node, prior_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node
            Returns.
            ------
                Node, a copy of new_node
        """
        if not prior_inds:
            return new_node

        # create an empty heap
        heap_cost = []  # stores -cost
        hq.heapify(heap_cost)
        k = int(2 * np.log10(len(self.node_list)))

        # search nearest cost in near_inds
        for i in prior_inds:
            prior_node = self.node_list[i]
            traj_temp = self.connect_node(prior_node, new_node)

            if traj_temp is None:
                continue

            traj_temp_info = {'cost': traj_temp.cost, 'traj_info': traj_temp}
            temp_node = self.steer_reconnect(prior_node, new_node, traj_temp_info, collision_dist_ext=0.0)

            if temp_node is None:
                continue

            cost_temp = temp_node.traj_info['cost']

            if heap_cost:
                if cost_temp > -min(heap_cost):
                    continue

            if prior_node.cost + cost_temp < new_node.cost:
                new_node = temp_node
                new_node.cost = prior_node.cost + cost_temp
                new_node.parent = prior_node

            hq.heappush(heap_cost, -cost_temp)
            if len(heap_cost) > k:
                hq.heappop(heap_cost)

        return new_node

    def search_feasible_goal_node(self):
        self.end.cost = np.inf
        goal_node = self.Node(self.end.x_pre, self.end.x_pre, self.end.vx_pre, self.end.vy_pre,
                              self.end.collision_state, 1.0)
        for i in range(len(self.node_list)):
            traj_final = self.connect_node(self.node_list[i], goal_node)
            if traj_final is None:
                continue
            cost_final = traj_final.cost + self.rho_t * traj_final.TS[-1]
            traj_goal_info = {'cost': cost_final, 'traj_info': traj_final}
            temp_goal_node = self.steer(self.node_list[i], goal_node, traj_goal_info,
                                        run_collision_modification=False, collision_dist_ext=0.3)

            print temp_goal_node

            if temp_goal_node is None:
                continue

            cost_primitive = temp_goal_node.traj_info.get('cost')

            if self.node_list[i].cost + cost_primitive < self.end.cost:
                self.end.cost = self.node_list[i].cost + cost_primitive
                self.end.traj_info = goal_node.traj_info
                self.end.parent = self.node_list[i]

        return self.end.parent

    def find_prior_nodes(self, new_node):
        """
        1) Returns all nodes are prior to this node
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes prior to this node
        """
        # Search vertices in a range no more than expand_dist
        prior_inds = []
        for i in range(len(self.node_list)):
            if self.node_list[i].sample_num < new_node.sample_num:
                prior_inds.append(i)
        return prior_inds

    def find_post_nodes(self, new_node):
        """
        1) Returns all nodes are post to this node
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes prior to this node
        """
        # Search vertices in a range no more than expand_dist
        post_inds = []
        for i in range(len(self.node_list)):
            if self.node_list[i].sample_num > new_node.sample_num:
                post_inds.append(i)
        return post_inds

    def rewire(self, new_node, post_inds):
        """
            For each node post to near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree
                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.
        """
        if not post_inds:
            return

        # create an empty heap
        heap_cost = []  # stores -cost
        k = int(2 * np.log10(len(self.node_list)))

        # search nearest cost in near_inds
        for i in post_inds:
            post_node = self.node_list[i]
            traj_temp = self.connect_node(new_node, post_node)

            if traj_temp is None:
                continue

            traj_temp_info = {'cost': traj_temp.cost, 'traj_info': traj_temp}
            temp_node = self.steer_reconnect(new_node, post_node, traj_temp_info, collision_dist_ext=0.0)
            if temp_node is None:
                continue

            cost_temp = temp_node.traj_info['cost']

            if heap_cost:
                if cost_temp > -min(heap_cost):
                    continue

            if new_node.cost + cost_temp < post_node.cost:
                post_node = temp_node
                post_node.cost = new_node.cost + cost_temp
                post_node.parent = new_node
                self.propagate_cost_to_leaves(post_node)

            hq.heappush(heap_cost, -cost_temp)
            if len(heap_cost) > k:
                hq.heappop(heap_cost)

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = parent_node.cost + node.traj_info.get('cost')
                self.propagate_cost_to_leaves(node)

    def steer_reconnect(self, from_node, to_node, traj_result, min_extend_T=0.0, collision_dist_ext=0.0):
        traj_info = traj_result['traj_info']
        Tmax = traj_info.TS[-1]
        if Tmax < min_extend_T:
            return None
        traj_x = []
        traj_y = []
        vel_x = []
        vel_y = []
        acc_x = []
        acc_y = []
        time = []
        for t in np.arange(0.0, Tmax, 0.05):
            des_state = traj_info.get_des_state(t)
            traj_x.append(des_state.pos[0])
            traj_y.append(des_state.pos[1])
            vel_x.append(des_state.vel[0])
            vel_y.append(des_state.vel[1])
            acc_x.append(des_state.acc[0])
            acc_y.append(des_state.acc[1])
            time.append(t)

        if not traj_x:
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

            if d_min_1 <= collision_dist_ext:
                return None

        new_node = to_node
        if new_node.collision_state == 1:
            energy_loss = 0.5 * (new_node.vx_pre**2 + new_node.vy_pre**2)
            new_node.traj_info = {'cost': traj_info.cost +
                                          self.rho_t * (traj_info.TS[-1] + self.recover_time) +
                                          self.rho_c * energy_loss,
                                  'traj_info': traj_info}
        else:
            new_node.traj_info = {'cost': traj_info.cost +
                                          self.rho_t * traj_info.TS[-1],
                                  'traj_info': traj_info}
        new_node.traj_x = traj_x
        new_node.traj_y = traj_y
        new_node.parent = from_node
        return new_node

    # def waypoint_generation(self, path, velocity, collision_state, time_interval):
    #     path.reverse()
    #     velocity.reverse()
    #     time_interval.reverse()
    #     collision_state.reverse()
    #
    #     print ('original path without refinement:', path)
    #     print ('original time interval without refinement:', time_interval)
    #     print ('original velocity without refinement:', velocity)
    #     print ('original collision state without refinement:', collision_state)
    #
    #     path_array = np.array(path)
    #     path_point_repeat = []
    #     index_collision_next = []
    #     for path_point in path:
    #         location = []
    #         path_point = np.array(path_point)
    #         for i in np.arange(len(path_array)):
    #             if (path_array[i] == path_point).all():
    #                 location.append(i)
    #         if len(location) > 1:
    #             path_point_repeat.append(path_point)
    #
    #     if len(path_point_repeat) > 0:
    #         path_point_repeat = np.array(path_point_repeat)
    #         path_point_repeat = np.unique(path_point_repeat, axis=0)
    #
    #         for point_repeat in path_point_repeat:
    #             location_set = np.where((path_array == point_repeat).all(axis=1))[0]
    #             start_location = location_set[0]
    #             end_location = location_set[-1]
    #             if end_location - start_location <= 1:
    #                 collision_state_temp = max(collision_state[start_location], collision_state[end_location])
    #             else:
    #                 collision_state_temp = collision_state[start_location]
    #             del path[start_location + 1:end_location + 1]
    #             del velocity[start_location + 1:end_location + 1]
    #             path_array = np.array(path)
    #             del time_interval[start_location + 1:end_location + 1]
    #             del collision_state[start_location + 1:end_location + 1]
    #             collision_state[start_location] = collision_state_temp
    #             index_collision_next.append(start_location)
    #
    #     waypoint_list = [np.hstack([np.mat(path[0]), np.mat([0])])]
    #     velocity_list = [np.hstack([np.mat(velocity[0]), np.mat([0])])]
    #     t_refinement = []
    #     time_seg = 0.0
    #     for i in range(1, len(path) - 1):
    #         waypoint_c = path[i]
    #         velocity_c = velocity[i]
    #         # without simplification
    #         waypoint_list.append(np.hstack([np.mat([waypoint_c]), np.mat([0])]))
    #         velocity_list.append(np.hstack([np.mat([velocity_c]), np.mat([0])]))
    #         if collision_state[i] == 0:
    #             t_refinement.append(time_interval[i])
    #         else:
    #             t_refinement.append(time_interval[i] - self.recover_deform_T)
    #     waypoint_list.append(np.hstack([np.mat(path[- 1]), np.mat([0])]))
    #     velocity_list.append(np.hstack([np.mat(velocity[- 1]), np.mat([0])]))
    #     t_refinement.append(time_interval[-1] + time_seg)
    #
    #     delete_index = []
    #     for i in range(2, len(waypoint_list)):
    #         if collision_state[i] > 0:
    #             p_p = waypoint_list[i - 1]
    #             distance_dictionary_p = self.generate_dictionary_distance_to_obstacles(p_p,
    #                                                                                    self.obstacle_list_line_segment,
    #                                                                                    self.obstacle_list_circle)
    #             d_min_p = (min(distance_dictionary_p.values()))
    #             key_min_p = (min(distance_dictionary_p, key=distance_dictionary_p.get))
    #
    #             p_i = waypoint_list[i]
    #             distance_dictionary_i = self.generate_dictionary_distance_to_obstacles(p_i,
    #                                                                                    self.obstacle_list_line_segment,
    #                                                                                    self.obstacle_list_circle)
    #             d_min_i = (min(distance_dictionary_i.values()))
    #             key_min_i = (min(distance_dictionary_i, key=distance_dictionary_i.get))
    #
    #             if cmp(key_min_i, key_min_p) == 0 and d_min_p == d_min_i:
    #                 delete_index.append(i - 1)
    #
    #     i = 0
    #     for del_ind in delete_index:
    #         ti = t_refinement[del_ind - 1 - i]
    #         t_refinement[del_ind - i] = t_refinement[del_ind - i] + ti
    #         del t_refinement[del_ind - 1 - i]
    #         del waypoint_list[del_ind - i]
    #         del velocity_list[del_ind - i]
    #         del collision_state[del_ind - i]
    #         i = i + 1
    #     return waypoint_list, velocity_list, t_refinement
    #
    # def waypoint_refinement_collision(self, waypoint_list, velocity_list, collision_state, time_refinement, R_collision):
    #     for i in range(0, len(waypoint_list)):
    #         if collision_state[i] > 0:
    #             p_i = deepcopy(waypoint_list[i])
    #             v_i = deepcopy(velocity_list[i])
    #             distance_dictionary_i = self.generate_dictionary_distance_to_obstacles(p_i,
    #                                                                                    self.obstacle_list_line_segment,
    #                                                                                    self.obstacle_list_circle)
    #             key_i = min(distance_dictionary_i, key=distance_dictionary_i.get)
    #             if key_i[0] == 0:
    #                 obstacle = self.obstacle_list_line_segment[key_i[1]]
    #                 center_obs = np.mat(
    #                     [0.5 * (obstacle[0][0] + obstacle[1][0]), 0.5 * (obstacle[0][1] + obstacle[1][1]), 0.0])
    #                 radius_obs = 0.0
    #             else:
    #                 obstacle = self.obstacle_list_circle[key_i[1]]
    #                 center_obs = np.mat([obstacle[0][0], obstacle[0][1], 0.0])
    #                 radius_obs = obstacle[0][2]
    #             R_w_wall = self.generate_frame_3d(np.array(p_i)[0], obstacle)
    #             p_i_wall = (R_w_wall.T * (p_i - center_obs).T).T
    #             p_i_wall_pre = deepcopy(p_i_wall)
    #             p_i_wall[0, 0] = R_collision + radius_obs
    #             p_i = (R_w_wall * p_i_wall.T).T + center_obs
    #             v_i_wall = (R_w_wall.T * v_i.T).T
    #             if v_i_wall[0, 0] == 0:
    #                 dt_i = 0.0
    #             else:
    #                 dt_i = abs((p_i_wall_pre[0, 0] - p_i_wall[0, 0]) / v_i_wall[0, 0])
    #             waypoint_list[i] = p_i
    #             time_refinement[i-1] = time_refinement[i-1] + dt_i
    #
    #     return waypoint_list, time_refinement