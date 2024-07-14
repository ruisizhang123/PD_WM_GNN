import torch
import math

import logging
logger = logging.getLogger(__name__)

class WM_extract(object):
    '''
    @brief Watermark extraction
    '''
    def __init__(self, ori_pos, wm_pos, keys, params, placedb, data_collections, device, wm_dist, wm_cell_idx=None, phase="before_attack", attack_ratio=0):
        '''
        @brief initialization
        '''
        self.ori_pos = ori_pos
        self.wm_pos = wm_pos
        self.keys = keys
        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.device = device
        self.wm_cell_idx = wm_cell_idx
        self.wm_dist = wm_dist
        self.phase = phase
        self.attack_ratio = attack_ratio

    def random_extract_before(self):
        x_diff_pos = self.wm_pos[:self.placedb.num_nodes] - self.ori_pos[:self.placedb.num_nodes]
        y_diff_pos = self.wm_pos[self.placedb.num_nodes:] - self.ori_pos[self.placedb.num_nodes:]

        total_match = 0  
        for i in range(len(self.wm_cell_idx)):
            cur_axis = self.keys[i]
            if cur_axis == 0:
                if x_diff_pos[self.wm_cell_idx[i]] != 0 or y_diff_pos[self.wm_cell_idx[i]] != 0:
                    total_match = total_match + 1
            if cur_axis == 1:
                if y_diff_pos[self.wm_cell_idx[i]] != 0 or x_diff_pos[self.wm_cell_idx[i]] != 0:
                    total_match = total_match + 1
            self.wm_dist[0][i] = x_diff_pos[self.wm_cell_idx[i]]
            self.wm_dist[1][i] = y_diff_pos[self.wm_cell_idx[i]]
        extract_ratio = total_match / len(self.wm_cell_idx)
        logger.info("Watermark extracted ratio: %f" % (extract_ratio))

        '''
        total_match = 0
        for i in range(len(self.wm_cell_idx)):
            cur_axis = self.keys[i]
            if cur_axis == 0:
                if x_diff_pos[self.wm_cell_idx[i]] != 0:
                    total_match = total_match + 1
            if cur_axis == 1:
                if y_diff_pos[self.wm_cell_idx[i]] != 0:
                    total_match = total_match + 1
        extract_ratio = total_match / len(self.wm_cell_idx)
        logger.info("Watermark extracted without y axis: %f" % (extract_ratio))
        '''

        return extract_ratio

    def random_extract_after(self):
        x_diff_pos = self.wm_pos[:self.placedb.num_nodes] - self.ori_pos[:self.placedb.num_nodes]
        y_diff_pos = self.wm_pos[self.placedb.num_nodes:] - self.ori_pos[self.placedb.num_nodes:]

        total_match = 0  
        for i in range(len(self.wm_cell_idx)):
            if x_diff_pos[self.wm_cell_idx[i]] == self.wm_dist[0][i] and y_diff_pos[self.wm_cell_idx[i]] == self.wm_dist[1][i]:
                total_match = total_match + 1
      
        extract_ratio = total_match / len(self.wm_cell_idx)
        logger.info("Watermark extracted with y axis: %f" % (extract_ratio))

        '''
        total_match = 0  
        for i in range(len(self.wm_cell_idx)):
            cur_axis = self.keys[i]
            if cur_axis == 0:
                if x_diff_pos[self.wm_cell_idx[i]] == self.wm_dist[0][i]:
                    total_match = total_match + 1
            if cur_axis == 1:
                if y_diff_pos[self.wm_cell_idx[i]] == self.wm_dist[1][i]:
                    total_match = total_match + 1
        extract_ratio = total_match / len(self.wm_cell_idx)
        logger.info("Watermark extracted without y axis: %f" % (extract_ratio))
        '''
        return extract_ratio

    def extract(self):
        assert self.wm_cell_idx is not None
        if self.phase == "before_attack":
            extract_ratio = self.random_extract_before()
        elif self.phase == "after_attack":
            extract_ratio = self.random_extract_after()
            
        return extract_ratio, self.wm_dist
   