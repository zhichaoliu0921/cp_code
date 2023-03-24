import random
import math
import numpy as np
class dynamic_env_generator:
    def __init__(self, min_x, max_x, min_y, max_y, obstalce_circle_dict, obs_dis_limit, resolution,
                 start, goal, num_change):
        self.obs_dis_limit = obs_dis_limit  # minimum distance between obstalces > 2 * obs_radius
        self.min_x = min_x + self.obs_dis_limit / 2
        self.max_x = max_x - self.obs_dis_limit / 2
        self.min_y = min_y + self.obs_dis_limit / 2
        self.max_y = max_y - self.obs_dis_limit / 2
        self.resolution = resolution  # resolution of the position
        self.obstalce_circle_dict = obstalce_circle_dict  # key: position, radius:radius, ind: index
        self.start = start
        self.goal = goal
        self.num_change = num_change
        self.obs_change_keys = None

    def adjust_one_obs(self, obstacle_pos_pre):
        range_x = (self.max_x - self.min_x) // self.resolution
        range_y = (self.max_y - self.min_x) // self.resolution

        grid_ind_x = random.randint(0, range_x)
        grid_ind_y = random.randint(0, range_y)

        obs_x = self.calc_grid_position(grid_ind_x, self.min_x)
        obs_y = self.calc_grid_position(grid_ind_y, self.min_y)

        if obs_x < self.min_x or obs_y < self.min_y:
            return False

        if obs_x > self.max_x or obs_y > self.max_y:
            return False

        for pos_obs in self.obstalce_circle_dict.keys():
            temp_dis_sq = (obs_x - pos_obs[0]) ** 2 + (obs_y - pos_obs[1]) ** 2
            temp_dis_st = (obs_x - self.start[0]) ** 2 + (obs_y - self.start[1]) ** 2
            temp_dis_ed = (obs_x - self.goal[0]) ** 2 + (obs_y - self.goal[1]) ** 2
            if temp_dis_st < (self.obs_dis_limit / 2) ** 2:
                return False

            if temp_dis_ed < (self.obs_dis_limit / 2) ** 2:
                return False

            if (obs_x, obs_y) in self.obstalce_circle_dict:
                return False

            if temp_dis_sq < self.obs_dis_limit ** 2:
                return False

            if temp_dis_ed < self.obs_dis_limit ** 2:
                return False

        self.obstalce_circle_dict[(obs_x, obs_y)] = {'radius': self.obstalce_circle_dict[obstacle_pos_pre]['radius'],
                                                     'ind': self.obstalce_circle_dict[obstacle_pos_pre]['ind']}

        del(self.obstalce_circle_dict[obstacle_pos_pre])
        return True

    def adjust_dynamic_env(self):

        index_change = 0

        if self.obs_change_keys is None:
            return

        for i in range(10000):
            if index_change >= self.num_change:
                return
            if self.adjust_one_obs(self.obs_change_keys[index_change]):
                index_change += 1

    def select_change_obs(self):
        self.obs_change_keys = random.sample(self.obstalce_circle_dict.keys(), self.num_change)
        print 'changed obstacles are:', self.obs_change_keys


    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos