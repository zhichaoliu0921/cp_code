"""
Trajectory planning Sample Code with RRT* collision accepted Jiaming Zha
author: Zhouyu Lu
"""

import math
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True
from shapely.geometry import LineString
from shapely.geometry import Point

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../RRT/")

try:
    from RRT import RRT
except ImportError:
    raise

class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super(RRTStar.Node, self).__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 min_x,
                 min_y,
                 max_x,
                 max_y,
                 resolution,
                 obstacle_circle_dict,
                 dis_potential_bound,
                 max_expand_dis=1.0,
                 min_expand_dis=0.1,
                 goal_sample_rate=20,
                 max_iter=300,
                 connect_circle_dist=2.0,
                 weight_potential=0.1,
                 search_until_max_iter=True):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        super(RRTStar, self).__init__(start, goal, min_x, min_y, max_x, max_y, resolution, obstacle_circle_dict,
                                      max_expand_dis, min_expand_dis, goal_sample_rate, max_iter)
        self.connect_circle_dist = connect_circle_dist
        self.dis_potential_bound = dis_potential_bound
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.weight_potential = weight_potential

    def planning(self, animation=True):
        """
        rrt star path planning
        animation: flag for animation on or off .
        """

        self.node_list = [self.start]
        self.node_set.add((self.start.x, self.start.y))
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd,
                                  self.max_expand_dis, self.min_expand_dis)

            if (new_node.x, new_node.y) in self.node_set:
                continue

            near_node = self.node_list[nearest_ind]
            path_dist_cost = math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
            path_potential_cost = self.calc_potential_cost(near_node, new_node)
            new_node.cost = near_node.cost + path_dist_cost + path_potential_cost

            if self.check_collision_circle(near_node, new_node):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                    self.node_set.add((new_node.x, new_node.y))
                else:
                    self.node_list.append(new_node)
                    self.node_set.add((new_node.x, new_node.y))

                if animation:
                    plt.plot(new_node.x, new_node.y, "xc")
                    # for stopping simulation with the esc key.
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event: [exit(
                                                     0) if event.key == 'escape' else None])
                    if i % 5 == 0:
                        plt.pause(0.001)

            if ((not self.search_until_max_iter)
                    and new_node):  # if reaches goal
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
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
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision_circle(near_node, t_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.max_expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision_circle(self.node_list[goal_ind], t_node):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        if hasattr(self, 'expand_dis'):
            r = min(r, self.max_expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
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
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)
            no_collision = self.check_collision_circle(new_node, edge_node)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        potential_cost = self.calc_potential_cost(from_node, to_node)
        return from_node.cost + d + potential_cost

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def calc_potential_cost(self, from_node, to_node):
        pos_st = [from_node.x, from_node.y]
        pos_ed = [to_node.x, to_node.y]

        dist_obs_dict = self.generate_line_distance_to_obstacles(pos_st, pos_ed)

        f_potential = 0.0
        for pos_obs in dist_obs_dict.keys():
            dis_obs_temp = dist_obs_dict[pos_obs]
            if dis_obs_temp <= self.dis_potential_bound:
                f_potential += (dis_obs_temp - self.dis_potential_bound) ** 2
        print f_potential
        return f_potential * self.weight_potential

    def waypoint_generation(self, path, dis_vis):
        path.reverse()

        waypoint_list = []
        wp1 = [path[0][0], path[0][1], 0.0]
        wp1_2d = [path[0][0], path[0][1]]
        waypoint_list.append(wp1)
        waypoint_list_set = set(tuple(wp1))

        for i in range(1, len(path)):
            wp_candidate_2d = [path[i][0], path[i][1]]
            visibility_flag = self.check_visibility_circle(wp1_2d, wp_candidate_2d, dis_vis)
            if not visibility_flag and i > 1:
                wp_candidate = [path[i - 1][0], path[i - 1][1], 0]
                if tuple(wp_candidate) not in waypoint_list_set:
                    waypoint_list.append(wp_candidate)
                    waypoint_list_set.add(tuple(wp_candidate))
                wp1_2d = [wp_candidate[0], wp_candidate[1]]

        waypoint_list.append([path[-1][0], path[-1][1], 0.0])
        return waypoint_list

    def check_visibility_circle(self, point1, point2, dis_vis):  # return false-collision, true-safe
        start_point = (point1[0], point1[1])
        end_point = (point2[0], point2[1])
        # line_seg_path = LineString([start_point, end_point])
        for pos_obs in self.obstacle_circle_dict.keys():
            radius = self.obstacle_circle_dict[pos_obs]['radius'] + dis_vis
            # circle_obs = Point(pos_obs[0], pos_obs[1]).buffer(radius).boundary
            # bool_temp = circle_obs.intersects(line_seg_path)
            dist_check = self.getDisPointToLine(pos_obs, start_point, end_point)
            if dist_check <= radius:
                return False
        return True
