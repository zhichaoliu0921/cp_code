import random
import math
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
from copy import deepcopy
class path_generation:
    def __init__(self, min_x, max_x, min_y, max_y, obstalce_circle_dict, resolution, start, goal):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.resolution = resolution  # resolution of the position
        self.obstalce_circle_dict = obstalce_circle_dict  # key: position, radius:radius, ind: index
        self.start = start
        self.goal = goal
        self.start_c = 0
        self.goal_c = 0
        # self.ext_len = 0.1

    def find_collision_point_path(self, p1, p2):
        dis_intersection_dict = dict()
        line_seg_path = LineString([p1, p2])
        for obs in self.obstalce_circle_dict:
            obs_radius = self.obstalce_circle_dict[obs]['radius']
            circle_obs = Point(obs[0], obs[1]).buffer(obs_radius).boundary
            collision_state = circle_obs.intersects(line_seg_path)
            if collision_state:
                # collision_points = circle_obs.intersection(line_seg_path)
                # for collision_point in collision_points:
                #     pos_temp = [collision_point.x, collision_point.y]
                #     dis_temp = math.sqrt((p1[0] - pos_temp[0])**2 + (p1[1] - pos_temp[1])**2)
                #     dis_intersection_dict[dis_temp] = {'obs_center': obs, 'obs_radius': obs_radius,
                #                                        'collision_point': pos_temp}

                dis_temp = math.sqrt((p1[0] - obs[0]) ** 2 + (p1[1] - obs[1]) ** 2)
                vec_p1_center = np.array([obs[0] - p1[0], obs[1] - p1[1]])
                vec_p1_center_norm = np.linalg.norm(vec_p1_center)
                unit_p1_center = vec_p1_center / vec_p1_center_norm
                vec_p1_collision = vec_p1_center - unit_p1_center * obs_radius
                vec_p1_recovery = vec_p1_center - unit_p1_center * math.sqrt(2) * obs_radius
                pos_temp_c = [vec_p1_collision[0] + p1[0], vec_p1_collision[1] + p1[1]]
                pos_temp_r = [vec_p1_recovery[0] + p1[0], vec_p1_recovery[1] + p1[1]]
                dis_intersection_dict[dis_temp] = {'obs_center': obs, 'obs_radius': obs_radius,
                                                   'collision_point': pos_temp_c, 'recovery_point': pos_temp_r}

        if dis_intersection_dict.keys():
            min_intersection_point = dis_intersection_dict[min(dis_intersection_dict)]
            return min_intersection_point
        else:
            return None

    def find_add_waypoint(self, p1, p2, intersection_point):
        center_obs = intersection_point['obs_center']
        obs_radius = intersection_point['obs_radius']
        vec_center = np.array([center_obs[0] - p1[0], center_obs[1] - p1[1]])
        vec_center_unit = vec_center / np.linalg.norm(vec_center)
        vec_center_unit_3d = np.array([vec_center_unit[0], vec_center_unit[1], 0])
        z_axis = np.array([0.0, 0.0, 1.0])
        vec_prep_ext_3d_1 = 1.0 * np.cross(z_axis, vec_center_unit_3d) * math.sqrt(2) * obs_radius
        vec_prep_ext_3d_2 = -1.0 * np.cross(z_axis, vec_center_unit_3d) * math.sqrt(2) * obs_radius
        vec_prep_ext_1 = np.array([vec_prep_ext_3d_1[0], vec_prep_ext_3d_1[1]])
        waypoint_add_c1 = list(vec_prep_ext_1 + np.array(center_obs))
        vec_move_1 = np.array([waypoint_add_c1[0] - p1[0], waypoint_add_c1[1] - p1[1]])
        vec_move_2 = np.array([p2[0] - waypoint_add_c1[0], p2[1] - waypoint_add_c1[1]])
        path_length_c1 = np.linalg.norm(vec_move_1) + np.linalg.norm(vec_move_2)

        vec_prep_ext_2 = np.array([vec_prep_ext_3d_2[0], vec_prep_ext_3d_2[1]])
        waypoint_add_c2 = list(vec_prep_ext_2 + np.array(center_obs))
        vec_move_1 = np.array([waypoint_add_c2[0] - p1[0], waypoint_add_c2[1] - p1[1]])
        vec_move_2 = np.array([p2[0] - waypoint_add_c2[0], p2[1] - waypoint_add_c2[1]])
        path_length_c2 = np.linalg.norm(vec_move_1) + np.linalg.norm(vec_move_2)

        if path_length_c1 < path_length_c2:
            return waypoint_add_c1
        elif path_length_c1 > path_length_c2:
            return waypoint_add_c2
        else:
            return random.choice([waypoint_add_c1, waypoint_add_c2])


        # vec_line = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        # vec_line_norm = np.linalg.norm(vec_line)
        # w_proj = np.dot(vec_center, vec_line) / vec_line_norm
        # vec_proj = vec_line / vec_line_norm * w_proj
        # vec_prep = -vec_center + vec_proj
        # if np.linalg.norm(vec_prep) != 0:
        #     vec_prep_ext = vec_prep / np.linalg.norm(vec_prep) * (obs_radius + self.ext_len)
        #     waypoint_add = list(vec_prep_ext + np.array(center_obs))
        #     return waypoint_add
        #
        # else:
        #     vec_center_unit = vec_center / np.linalg.norm(vec_center)
        #     vec_center_unit_3d = np.array([vec_center_unit[0], vec_center_unit[1], 0])
        #     z_axis = np.array([0.0, 0.0, 1.0])
        #     vec_prep_ext_3d = random.choice([-1.0, 1.0]) * np.cross(z_axis, vec_center_unit_3d) * \
        #                       (obs_radius + self.ext_len)
        #     vec_prep_ext = np.array([vec_prep_ext_3d[0], vec_prep_ext_3d[1]])
        #     waypoint_add = list(vec_prep_ext + np.array(center_obs))
        #     return waypoint_add

    def waypoint_generation(self):
        waypoint_list = [[self.start[0], self.start[1], 0.0]]
        collision_list = [self.start_c]
        collision_info_dict = dict()
        p1 = self.start
        p2 = self.goal
        collision_point = self.find_collision_point_path(p1, p2)
        while collision_point is not None:
            pos_c = collision_point['collision_point']
            while collision_point is not None:
                pos_c = collision_point['collision_point']
                p2 = pos_c
                collision_point_pre = deepcopy(collision_point)
                collision_point = self.find_collision_point_path(p1, p2)
                if collision_point is not None:
                    if collision_point_pre['obs_center'][0] == collision_point['obs_center'][0] and \
                            collision_point_pre['obs_center'][1] == collision_point['obs_center'][1]:
                        break
            waypoint_list.append([pos_c[0], pos_c[1], 0.0])
            collision_info_dict[(pos_c[0], pos_c[1], 0.0)] = collision_point_pre
            collision_list.append(1)
            p2 = self.goal
            waypoint_add = self.find_add_waypoint(p1, p2, collision_point_pre)
            waypoint_list.append([waypoint_add[0], waypoint_add[1], 0.0])
            collision_list.append(0)
            p1 = waypoint_add
            collision_point = self.find_collision_point_path(p1, p2)

        waypoint_list.append([self.goal[0], self.goal[1], 0.0])
        collision_list.append(self.goal_c)
        return waypoint_list, collision_list, collision_info_dict

    def adjust_end_state_by_horizon(self, waypoint_list, collision_list, resolution, h_horizon, v_horizon):
        p_c = [waypoint_list[0][0], waypoint_list[0][1]]
        p1 = [waypoint_list[-2][0], waypoint_list[-2][1]]
        p2 = [waypoint_list[-1][0], waypoint_list[-1][1]]
        vec_path = np.array(p2) - np.array(p1)
        length_path = np.linalg.norm(vec_path)
        unit_vec_path = vec_path / length_path
        for s in np.arange(0, 1, resolution):
            p_sample = s * length_path * unit_vec_path + p1
            dis_sample = np.linalg.norm(np.array(p_sample - np.array(p_c)))
            if dis_sample > h_horizon:
                if s == 0 or s == resolution:
                    del(waypoint_list[-1])
                    del(collision_list[-1])
                    return waypoint_list, collision_list, [0.0, 0.0, 0.0]
                else:
                    p_waypoint = (s-resolution) * length_path * unit_vec_path + p1
                    waypoint_list[-1] = [p_waypoint[0], p_waypoint[1], 0.0]
                    vel_ed_2d = unit_vec_path * v_horizon
                    return waypoint_list, collision_list, [vel_ed_2d[0], vel_ed_2d[1], 0.0]

        return waypoint_list, collision_list, [0.0, 0.0, 0.0]

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