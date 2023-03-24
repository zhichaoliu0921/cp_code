import math
import numpy as np
class map_gen_horizon:
    def __init__(self, obstalce_circle_dict, current_position, obstalce_circle_dict_vis, dis_horizon, obs_radius):
        self.cur_pos = current_position
        self.obstalce_circle_dict = obstalce_circle_dict  # key: position, radius: radius, ind: index
        self.obstalce_circle_dict_vis = obstalce_circle_dict_vis
        self.dis_horizon = dis_horizon
        self.obs_radius = obs_radius


    def get_obs_in_horizon(self):
        for pos_obs in self.obstalce_circle_dict.keys():
            temp_dis = self.getDisPointToCircle(self.cur_pos, pos_obs, self.obs_radius)
            if temp_dis <= self.dis_horizon:
                if pos_obs not in self.obstalce_circle_dict_vis:
                    self.obstalce_circle_dict_vis[pos_obs] = {'radius': self.obstalce_circle_dict[pos_obs]['radius'],
                                                              'ind': self.obstalce_circle_dict[pos_obs]['ind']}

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