import random
import math
import numpy as np
class forest_env_generator:
    def __init__(self, min_x, max_x, min_y, max_y, obs_radius, obs_num, obs_dis_limit, resolution, start, goal):
        self.obs_radius = obs_radius  # augmented obstacle radius
        self.obs_dis_limit = obs_dis_limit  # minimum distance between obstalces > 2 * obs_radius
        self.obs_num = obs_num
        self.min_x = min_x + self.obs_dis_limit / 2
        self.max_x = max_x - self.obs_dis_limit / 2
        self.min_y = min_y + self.obs_dis_limit / 2
        self.max_y = max_y - self.obs_dis_limit / 2
        self.resolution = resolution  # resolution of the position
        self.obstalce_circle_dict = dict()  # key: position, radius:radius, ind: index
        self.start = start
        self.goal = goal

    def generate_one_obs(self, index):
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

        self.obstalce_circle_dict[(obs_x, obs_y)] = {'radius': self.obs_radius, 'ind': index}
        return True

    def generate_forest(self):
        index_obs = 0
        area_obs = (self.max_x - self.min_x + self.obs_dis_limit) * \
                   (self.max_y - self.min_y + self.obs_dis_limit)
        num_obs = self.obs_num

        for i in range(10000):
            if index_obs >= num_obs:
                self.obs_num = np.round(len(self.obstalce_circle_dict) * math.pi * self.obs_radius ** 2 / area_obs, decimals=2)
                return

            if self.generate_one_obs(index_obs):
                index_obs += 1

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