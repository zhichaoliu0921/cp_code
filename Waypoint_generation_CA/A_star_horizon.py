import math
import numpy as np

import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import Point

show_animation = False


class AStarPlanner:

    def __init__(self, min_x, min_y, max_x, max_y, resolution, obstacle_circle_dict, dis_potential_bound,
                 weight_potential=0.05):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.min_x, self.min_y = min_x, min_y
        self.max_x, self.max_y = max_x, max_y
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.motion = self.get_motion_model()
        self.obstacle_circle_dict = obstacle_circle_dict
        self.dis_potential_bound = dis_potential_bound
        self.weight_potential = weight_potential
        # self.calc_obstacle_map()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost  # consist of path cost and potential cost on the node
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy, h_vis):
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        six = self.calc_xy_index(sx, self.min_x)
        siy = self.calc_xy_index(sy, self.min_y)
        cost_potential_st = self.calc_potential_cost(six, siy) / self.resolution * self.weight_potential
        # cost_potential_st = 0.0
        start_node = self.Node(six, siy, cost_potential_st,  -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))

            # c_id = min(
            #     open_set,
            #     key=lambda o: open_set[o].cost)

            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            cur_x = self.calc_grid_position(current.x, self.min_x)
            cur_y = self.calc_grid_position(current.x, self.min_x)
            dis_cur = math.hypot(cur_x - sx, cur_y - sy)
            if dis_cur >= h_vis:
                print("Out of horizon")
                goal_node = closed_set[current.parent_index]
                break

            if (current.x == goal_node.x and current.y == goal_node.y):
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                ix_temp = current.x + self.motion[i][0]
                iy_temp = current.y + self.motion[i][1]
                cost_potential = self.calc_potential_cost(ix_temp, iy_temp) / self.resolution * self.weight_potential
                # cost_potential = 0.0
                node = self.Node(ix_temp, iy_temp, current.cost + self.motion[i][2]
                                 + cost_potential, c_id)

                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def calc_potential_cost(self, ix, iy):
        px = self.calc_grid_position(ix, self.min_x)
        py = self.calc_grid_position(iy, self.min_y)

        dist_obs_dict = self.generate_dictionary_distance_to_obstacles([px, py])

        f_potential = 0.0
        for pos_obs in dist_obs_dict.keys():
            dis_obs_temp = dist_obs_dict[pos_obs]
            if dis_obs_temp <= self.dis_potential_bound:
                f_potential += (dis_obs_temp - self.dis_potential_bound) ** 2
        return f_potential

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

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

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px > self.max_x:
            return False
        elif py > self.max_y:
            return False

        # collision check by calculating distance

        dist_obs_dict = self.generate_dictionary_distance_to_obstacles([px, py])

        d_min = min(dist_obs_dict.values())
        key_min = min(dist_obs_dict, key=dist_obs_dict.get)

        if d_min <= 0.0:
            return False

        return True

    def waypoint_generation(self, rx, ry, dis_vis):
        rx.reverse()
        ry.reverse()

        waypoint_list = []
        wp1 = [rx[0], ry[0], 0.0]
        wp1_2d = [rx[0], ry[0]]
        waypoint_list.append(wp1)
        waypoint_list_set = set(tuple(wp1))

        for i in range(1, len(rx)):
            wp_candidate_2d = [rx[i], ry[i]]
            visibility_flag = self.check_visibility_circle(wp1_2d, wp_candidate_2d, dis_vis)
            if not visibility_flag and i > 1:
                wp_candidate = [rx[i-1], ry[i-1], 0]
                if tuple(wp_candidate) not in waypoint_list_set:
                    waypoint_list.append(wp_candidate)
                    waypoint_list_set.add(tuple(wp_candidate))
                wp1_2d = [wp_candidate[0], wp_candidate[1]]

        waypoint_list.append([rx[-1], ry[-1], 0.0])
        return waypoint_list

    # def calc_obstacle_map(self):
    #     ox, oy = self.get_map_list_from_obstalce_list()
    #
    #     self.min_x = round(min(ox) / self.resolution_pos) * self.resolution_pos
    #     self.min_y = round(min(oy) / self.resolution_pos) * self.resolution_pos
    #     self.max_x = round(max(ox) / self.resolution_pos) * self.resolution_pos
    #     self.max_y = round(max(oy) / self.resolution_pos) * self.resolution_pos
    #     print("min_x:", self.min_x)
    #     print("min_y:", self.min_y)
    #     print("max_x:", self.max_x)
    #     print("max_y:", self.max_y)
    #
    #     self.x_width = round((self.max_x - self.min_x) / self.resolution_pos)
    #     self.y_width = round((self.max_y - self.min_y) / self.resolution_pos)
    #
    #     print("x_width:", self.x_width)
    #     print("y_width:", self.y_width)
    #
    #     # obstacle map generation
    #     self.obstacle_map = [[False for _ in np.arange(self.y_width + 1)]
    #                          for _ in np.arange(self.x_width + 1)]
    #
    #     for ix in np.arange(self.x_width + 1):
    #         ix = int(ix)
    #         x = self.calc_grid_position(ix, self.min_x)
    #         for iy in np.arange(self.y_width + 1):
    #             iy = int(iy)
    #             y = self.calc_grid_position(iy, self.min_y)
    #             p_current = np.mat([x, y])
    #             d_dict_cur = self.generate_dictionary_distance_to_obstacles(p_current, self.obstacle_list_line_segment,
    #                                                                         self.obstacle_list_circle)
    #             d = min(d_dict_cur.values())
    #             # min_obs_ind = min(d_dict_cur, key=d_dict_cur.get)
    #             if d <= self.rr:
    #                 self.obstacle_map[ix][iy] = True

    def generate_dictionary_distance_to_obstacles(self, pos_c):
        distance_dictionary = dict()
        for pos_obs in self.obstacle_circle_dict.keys():
            radius = self.obstacle_circle_dict[pos_obs]['radius']
            distance_temp = self.getDisPointToCircle(pos_c, pos_obs, radius)
            distance_dictionary[pos_obs] = distance_temp
        return distance_dictionary

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

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

    def check_visibility_circle(self, point1, point2, dis_vis):  # return false-collision, true-safe
        start_point = (point1[0], point1[1])
        end_point = (point2[0], point2[1])
        # line_seg_path = LineString([start_point, end_point])
        for pos_obs in self.obstacle_circle_dict.keys():
            radius = self.obstacle_circle_dict[pos_obs]['radius'] + dis_vis
            # circle_obs = Point(pos_obs[0], pos_obs[1]).buffer(radius).boundary
            # bool_temp = circle_obs.intersects(line_seg_path)
            dist_check = self.getDisPointToLine(pos_obs, start_point, end_point)
            if dist_check < radius:
                return False
        return True

    @staticmethod
    def generate_waypoints_within_horizon(waypoint_list, h_vis, d_sample):
        start = waypoint_list[0]
        for i in range(1, len(waypoint_list)):
            dis_temp = math.hypot(waypoint_list[i][0] - waypoint_list[i][0])


