##
# @file   greedy_legalize.py
# @author Yibo Lin
# @date   Jun 2018
#

import math 
import torch
from torch import nn
from torch.autograd import Function
import pdb

import dreamplace.ops.greedy_legalize.greedy_legalize_cpp as greedy_legalize_cpp

class GreedyLegalizeFunction(Function):
    """ Legalize cells with greedy approach 
    """
    @staticmethod
    def forward(
          init_pos,
          pos,
          node_size_x,
          node_size_y,
          node_weights, 
          flat_region_boxes, 
          flat_region_boxes_start, 
          node2fence_region_map, 
          xl, 
          yl, 
          xh, 
          yh, 
          site_width, 
          row_height, 
          num_bins_x, 
          num_bins_y, 
          num_movable_nodes, 
          num_terminal_NIs, 
          num_filler_nodes,
          num_parity_cells,
          wm_parity_cells,
          wm_parity_keys
          ):
        #import pdb; pdb.set_trace()
        print("print info")
        print("init_pos", init_pos, len(init_pos))
        print("pos", pos, len(pos))
        print("node_size_x", node_size_x, len(node_size_x))
        print("node_size_y", node_size_y, len(node_size_y))
        print("node_weights", node_weights, len(node_weights))
        print("flat_region_boxes", flat_region_boxes, len(flat_region_boxes))
        print("flat_region_boxes_start", flat_region_boxes_start, len(flat_region_boxes_start))
        print("node2fence_region_map", node2fence_region_map, len(node2fence_region_map))
        print("xl", xl, yl, xh, yh, site_width, row_height, num_bins_x, num_bins_y)
        print("num", num_movable_nodes, num_terminal_NIs, num_filler_nodes, num_parity_cells)
        if pos.is_cuda:
            output = greedy_legalize_cpp.forward(
                    init_pos.view(init_pos.numel()).cpu(), 
                    pos.view(pos.numel()).cpu(), 
                    node_size_x.cpu(),
                    node_size_y.cpu(),
                    node_weights.cpu(), 
                    flat_region_boxes.cpu(), 
                    flat_region_boxes_start.cpu(), 
                    node2fence_region_map.cpu(), 
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    site_width, 
                    row_height, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_nodes, 
                    num_terminal_NIs, 
                    num_filler_nodes,
                    num_parity_cells,
                    wm_parity_cells.cpu(),
                    wm_parity_keys.cpu()
                    ).cuda()
        else:
            output = greedy_legalize_cpp.forward(
                    init_pos.view(init_pos.numel()), 
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    node_weights, 
                    flat_region_boxes, 
                    flat_region_boxes_start, 
                    node2fence_region_map, 
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    site_width, 
                    row_height, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_nodes, 
                    num_terminal_NIs, 
                    num_filler_nodes,
                    num_parity_cells,
                    wm_parity_cells,
                    wm_parity_keys
                    )
        return output

class GreedyLegalize(object):
    """ Legalize cells with greedy approach 
    """
    def __init__(self, node_size_x, node_size_y, node_weights, 
            flat_region_boxes, flat_region_boxes_start, node2fence_region_map, 
            xl, yl, xh, yh, site_width, row_height, num_bins_x, num_bins_y, num_movable_nodes, num_terminal_NIs, num_filler_nodes,
            wmParityCells=None, wmParityKeys=None):
        super(GreedyLegalize, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.node_weights = node_weights
        self.flat_region_boxes = flat_region_boxes 
        self.flat_region_boxes_start = flat_region_boxes_start 
        self.node2fence_region_map = node2fence_region_map
        self.xl = xl 
        self.yl = yl
        self.xh = xh 
        self.yh = yh 
        self.site_width = site_width 
        self.row_height = row_height 
        self.num_bins_x = num_bins_x 
        self.num_bins_y = num_bins_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminal_NIs = num_terminal_NIs
        self.num_filler_nodes = num_filler_nodes

        self.wmParityCells = wmParityCells
        if wmParityCells is not None:
            self.wm_parity_cells = torch.zeros(num_movable_nodes, dtype=torch.int, device=self.node_size_x.device)
            self.wm_parity_cells[wmParityCells.long()] = 1
            self.wm_parity_keys = torch.ones_like(self.wm_parity_cells)
            self.wm_parity_keys *= -1
            self.wm_parity_keys[wmParityCells.long()] = wmParityKeys
            self.num_parity_cells = self.wm_parity_cells.sum().item()
        else:
            self.wm_parity_cells = torch.zeros(num_movable_nodes, dtype=torch.int, device=self.node_size_x.device)
            self.wm_parity_keys = torch.ones_like(self.wm_parity_cells)
            self.wm_parity_keys *= -1
            self.num_parity_cells = self.wm_parity_cells.sum().item()

    def __call__(self, init_pos, pos): 
        """ 
        @param init_pos the reference position for displacement minization
        @param pos current roughly legal position
        """
        return GreedyLegalizeFunction.forward(
                init_pos, 
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                node_weights=self.node_weights, 
                flat_region_boxes=self.flat_region_boxes, 
                flat_region_boxes_start=self.flat_region_boxes_start, 
                node2fence_region_map=self.node2fence_region_map, 
                xl=self.xl, 
                yl=self.yl, 
                xh=self.xh, 
                yh=self.yh, 
                site_width=self.site_width, 
                row_height=self.row_height, 
                num_bins_x=self.num_bins_x, 
                num_bins_y=self.num_bins_y,
                num_movable_nodes=self.num_movable_nodes, 
                num_terminal_NIs=self.num_terminal_NIs, 
                num_filler_nodes=self.num_filler_nodes,
                num_parity_cells=self.num_parity_cells,
                wm_parity_cells=self.wm_parity_cells,
                wm_parity_keys=self.wm_parity_keys
                )
    