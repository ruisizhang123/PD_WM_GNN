##
# @file   legality_check.py
# @author Yibo Lin
# @date   Jan 2020
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.legality_check.legality_check_cpp as legality_check_cpp


class LegalityCheck(object):
    """ Check legality including, 
    1. out of boundary 
    2. row and site alignment 
    3. overlap 
    4. fence region 
    """
    def __init__(self, node_size_x, node_size_y, flat_region_boxes,
                 flat_region_boxes_start, node2fence_region_map, xl, yl, xh,
                 yh, site_width, row_height, scale_factor, num_terminals,
                 num_movable_nodes, wm_parity_cells=None, wm_parity_keys=None):
        super(LegalityCheck, self).__init__()
        self.node_size_x = node_size_x.cpu()
        self.node_size_y = node_size_y.cpu()
        self.flat_region_boxes = flat_region_boxes.cpu()
        self.flat_region_boxes_start = flat_region_boxes_start.cpu()
        self.node2fence_region_map = node2fence_region_map.cpu()
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.site_width = site_width
        self.row_height = row_height
        # due to limited numerical precision, we must know
        # the scale factor to control the precision for comparison;
        # we assume the scale factor is computed from row_height and site width;
        # the assumption about scale factor is that
        # everything is integer before being scaled.
        self.scale_factor = scale_factor
        self.num_terminals = num_terminals
        self.num_movable_nodes = num_movable_nodes

        # row parity check
        self.wm_parity_cells = wm_parity_cells
        self.wm_parity_keys = wm_parity_keys

    def __call__(self, pos):
        return self.forward(pos)

    def forward(self, pos):
        """ 
        @param pos current roughly legal position
        """
        if pos.is_cuda:
            pos_cpu = pos.cpu()
        else:
            pos_cpu = pos
        
        if self.wm_parity_cells is not None:  
            flag1 = legality_check_cpp.forward(
                pos_cpu, self.node_size_x, self.node_size_y,
                self.flat_region_boxes, self.flat_region_boxes_start,
                self.node2fence_region_map, self.xl, self.yl, self.xh, self.yh,
                self.site_width, self.row_height, self.scale_factor,
                self.num_terminals, self.num_movable_nodes)
            total_matched = 0
            for i in range(len(self.wm_parity_cells)):
                cur_cell = self.wm_parity_cells[i]
                cur_key = self.wm_parity_keys[i]
                node_yl = pos_cpu[len(self.node_size_x)+ cur_cell]
                row_id = (node_yl - self.yl) // self.row_height
                if row_id % 2 == cur_key:
                    total_matched += 1
            extract_ratio = total_matched/len(self.wm_parity_cells)
            
            print("before legalize extract ratio: ", extract_ratio)
            if extract_ratio == 1:
                flag2 = True
            else:
                flag2 = False
            return flag1 and flag2
        else:
            return legality_check_cpp.forward(
                pos_cpu, self.node_size_x, self.node_size_y,
                self.flat_region_boxes, self.flat_region_boxes_start,
                self.node2fence_region_map, self.xl, self.yl, self.xh, self.yh,
                self.site_width, self.row_height, self.scale_factor,
                self.num_terminals, self.num_movable_nodes)
    

