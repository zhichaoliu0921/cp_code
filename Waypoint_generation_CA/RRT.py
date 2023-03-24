"""
Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)
author: AtsushiSakai(@Atsushi_twi)
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point

show_animation = False


class RRT(object):
    """
    Class for RRT planning
    """

    class Node(object):
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None

    def __init__(self,
                 start,
                 goal,
                 min_x,
                 min_y,
                 max_x,
                 max_y,
                 resolution,
                 obstacle_circle_dict,
                 max_expand_dis=1.0,
                 min_expand_dis=0.1,
                 goal_sample_rate=5,
                 max_iter=500):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.max_expand_dis = max_expand_dis
        self.min_expand_dis = min_expand_dis
        self.resolution = resolution
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_circle_dict = obstacle_circle_dict
        self.node_list = []
        self.node_set = set()

    def planning(self, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        self.node_set.add((self.start.x, self.start.y))

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.max_expand_dis, self.min_expand_dis)

            if (new_node.x, new_node.y) in self.node_set:
                continue

            if self.check_collision_circle(nearest_node, new_node):
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

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.max_expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.max_expand_dis)
                if self.check_collision_circle(self.node_list[-1], final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    def steer(self, from_node, to_node, max_extend_length=float("inf"), min_extend_length=float("-inf")):

        new_node = self.Node(from_node.x, from_node.y)
        new_node_idx = self.calc_xy_index(new_node.x, self.min_x)
        new_node_idy = self.calc_xy_index(new_node.y, self.min_y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        if d > max_extend_length:
            extend_length = max_extend_length
        elif d < max_extend_length:
            extend_length = min_extend_length
        else:
            extend_length = d

        extend_x = new_node.x + extend_length * math.cos(theta)
        extend_y = new_node.y + extend_length * math.sin(theta)
        extend_idx = self.calc_xy_index(extend_x, self.min_x)
        extend_idy = self.calc_xy_index(extend_y, self.min_y)

        if extend_idx == new_node_idx and extend_idy == new_node_idy:
            motion = [[1, 0],
                      [0, 1],
                      [-1, 0],
                      [0, -1],
                      [-1, -1],
                      [-1, 1],
                      [1, -1],
                      [1, 1]]
            rand_move = random.randint(0, len(motion) - 1)
            extend_idx += motion[rand_move][0]
            extend_idy += motion[rand_move][1]

        extend_x = self.calc_grid_position(extend_idx, self.min_x)
        extend_y = self.calc_grid_position(extend_idy, self.min_y)
        new_node.x = extend_x
        new_node.y = extend_y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_ix = random.randint(0, int(self.x_width))
            rand_iy = random.randint(0, int(self.y_width))
            rand_x = self.calc_grid_position(rand_ix, self.min_x)
            rand_y = self.calc_grid_position(rand_iy, self.min_y)
            if rand_x < self.min_x:
                rand_x = self.min_x
            elif rand_x > self.max_x:
                rand_x = self.max_x

            if rand_y < self.min_y:
                rand_y = self.min_y
            elif rand_y > self.max_y:
                rand_y = self.max_y

            rnd = self.Node(rand_x, rand_y)

        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def get_nearest_node_index(self, node_list, rnd_node):
        dist_list = []
        for node in node_list:
            dist_cost = (node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
            dist_list.append(dist_cost)

        minind = dist_list.index(min(dist_list))

        return minind

    def check_collision_circle(self, from_node, to_node):
        if from_node is None or to_node is None:
            return False

        pos_st = [from_node.x, from_node.y]
        pos_ed = [to_node.x, to_node.y]

        dist_obs_dict = self.generate_line_distance_to_obstacles(pos_st, pos_ed)
        d_min = min(dist_obs_dict.values())
        key_min = min(dist_obs_dict, key=dist_obs_dict.get)

        if d_min <= 0.0:
            return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def generate_line_distance_to_obstacles(self, pos_st, pos_ed):
        distance_dictionary = dict()
        for pos_obs in self.obstacle_circle_dict.keys():
            radius = self.obstacle_circle_dict[pos_obs]['radius']
            distance_temp = self.getDisPointToLine(pos_obs, pos_st, pos_ed) - radius
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