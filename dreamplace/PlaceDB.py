##
# @file   PlaceDB.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  placement database
#

import sys
import os
import re
import math
import time
import random
import numpy as np
import torch
import logging
import Params
import dreamplace
import dreamplace.ops.place_io.place_io as place_io
import dreamplace.ops.fence_region.fence_region as fence_region
import pdb
import dgl
import itertools
import pickle
from GNN import *
import torch.nn.functional as F
# plot the score map
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

import numpy as np
datatypes = {
        'float32' : np.float32,
        'float64' : np.float64
}

class PlaceDB (object):
    """
    @brief placement database
    """
    def __init__(self):
        """
        initialization
        To avoid the usage of list, I flatten everything.
        """
        self.rawdb = None # raw placement database, a C++ object
        self.pydb = None # python placement database interface

        self.num_physical_nodes = 0 # number of real nodes, including movable nodes, terminals, and terminal_NIs
        self.num_terminals = 0 # number of terminals, essentially fixed macros
        self.num_terminal_NIs = 0 # number of terminal_NIs that can be overlapped, essentially IO pins
        self.node_name2id_map = {} # node name to id map, cell name
        self.node_names = None # 1D array, cell name
        self.node_x = None # 1D array, cell position x
        self.node_y = None # 1D array, cell position y
        self.node_orient = None # 1D array, cell orientation
        self.node_size_x = None # 1D array, cell width
        self.node_size_y = None # 1D array, cell height

        self.node2orig_node_map = None # some fixed cells may have non-rectangular shapes; we flatten them and create new nodes
                                        # this map maps the current multiple node ids into the original one

        self.pin_direct = None # 1D array, pin direction IO
        self.pin_offset_x = None # 1D array, pin offset x to its node
        self.pin_offset_y = None # 1D array, pin offset y to its node

        self.net_name2id_map = {} # net name to id map
        self.net_names = None # net name
        self.net_weights = None # weights for each net

        self.net2pin_map = None # array of 1D array, each row stores pin id
        self.flat_net2pin_map = None # flatten version of net2pin_map
        self.flat_net2pin_start_map = None # starting index of each net in flat_net2pin_map

        self.node2pin_map = None # array of 1D array, contains pin id of each node
        self.flat_node2pin_map = None # flatten version of node2pin_map
        self.flat_node2pin_start_map = None # starting index of each node in flat_node2pin_map

        self.pin2node_map = None # 1D array, contain parent node id of each pin
        self.pin2net_map = None # 1D array, contain parent net id of each pin

        self.rows = None # NumRows x 4 array, stores xl, yl, xh, yh of each row

        self.regions = None # array of 1D array, placement regions like FENCE and GUIDE
        self.flat_region_boxes = None # flat version of regions
        self.flat_region_boxes_start = None # start indices of regions, length of num regions + 1
        self.node2fence_region_map = None # map cell to a region, maximum integer if no fence region

        self.xl = None
        self.yl = None
        self.xh = None
        self.yh = None

        self.row_height = None
        self.site_width = None

        self.bin_size_x = None
        self.bin_size_y = None
        self.num_bins_x = None
        self.num_bins_y = None

        self.num_movable_pins = None

        self.total_movable_node_area = None # total movable cell area
        self.total_fixed_node_area = None # total fixed cell area
        self.total_space_area = None # total placeable space area excluding fixed cells

        # enable filler cells
        # the Idea from e-place and RePlace
        self.total_filler_node_area = None
        self.num_filler_nodes = None

        self.routing_grid_xl = None
        self.routing_grid_yl = None
        self.routing_grid_xh = None
        self.routing_grid_yh = None
        self.num_routing_grids_x = None
        self.num_routing_grids_y = None
        self.num_routing_layers = None
        self.unit_horizontal_capacity = None # per unit distance, projected to one layer
        self.unit_vertical_capacity = None # per unit distance, projected to one layer
        self.unit_horizontal_capacities = None # per unit distance, layer by layer
        self.unit_vertical_capacities = None # per unit distance, layer by layer
        self.initial_horizontal_demand_map = None # routing demand map from fixed cells, indexed by (grid x, grid y), projected to one layer
        self.initial_vertical_demand_map = None # routing demand map from fixed cells, indexed by (grid x, grid y), projected to one layer

        self.max_net_weight = None # maximum net weight in timing opt
        self.dtype = None

    def scale_pl(self, shift_factor, scale_factor):
        """
        @brief scale placement solution only
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        self.node_x -= shift_factor[0]
        self.node_x *= scale_factor
        self.node_y -= shift_factor[1]
        self.node_y *= scale_factor

    def unscale_pl(self, shift_factor, scale_factor): 
        """
        @brief unscale placement solution only
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        unscale_factor = 1.0 / scale_factor
        if shift_factor[0] == 0 and shift_factor[1] == 0 and unscale_factor == 1.0:
            node_x = self.node_x
            node_y = self.node_y
        else:
            node_x = self.node_x * unscale_factor + shift_factor[0]
            node_y = self.node_y * unscale_factor + shift_factor[1]

        return node_x, node_y

    def scale(self, shift_factor, scale_factor):
        """
        @brief shift and scale coordinates
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        logging.info("shift coordinate system by (%g, %g), scale coordinate system by %g" 
                % (shift_factor[0], shift_factor[1], scale_factor))
        self.scale_pl(shift_factor, scale_factor)
        self.node_size_x *= scale_factor
        self.node_size_y *= scale_factor
        self.pin_offset_x *= scale_factor
        self.pin_offset_y *= scale_factor
        self.xl -= shift_factor[0]
        self.xl *= scale_factor
        self.yl -= shift_factor[1]
        self.yl *= scale_factor
        self.xh -= shift_factor[0]
        self.xh *= scale_factor
        self.yh -= shift_factor[1]
        self.yh *= scale_factor
        self.row_height *= scale_factor
        self.site_width *= scale_factor

        # shift factor for rectangle 
        box_shift_factor = np.array([shift_factor, shift_factor]).reshape(1, -1)
        self.rows -= box_shift_factor
        self.rows *= scale_factor
        self.total_space_area *= scale_factor * scale_factor # this is area

        if len(self.flat_region_boxes): 
            self.flat_region_boxes -= box_shift_factor
            self.flat_region_boxes *= scale_factor
        # may have performance issue
        # I assume there are not many boxes
        for i in range(len(self.regions)):
            self.regions[i] -= box_shift_factor
            self.regions[i] *= scale_factor

    def sort(self):
        """
        @brief Sort net by degree.
        Sort pin array such that pins belonging to the same net is abutting each other
        """
        logging.info("sort nets by degree and pins by net")

        # sort nets by degree
        net_degrees = np.array([len(pins) for pins in self.net2pin_map])
        net_order = net_degrees.argsort() # indexed by new net_id, content is old net_id
        self.net_names = self.net_names[net_order]
        self.net2pin_map = self.net2pin_map[net_order]
        for net_id, net_name in enumerate(self.net_names):
            self.net_name2id_map[net_name] = net_id
        for new_net_id in range(len(net_order)):
            for pin_id in self.net2pin_map[new_net_id]:
                self.pin2net_map[pin_id] = new_net_id
        ## check
        #for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id

        # sort pins such that pins belonging to the same net is abutting each other
        pin_order = self.pin2net_map.argsort() # indexed new pin_id, content is old pin_id
        self.pin2net_map = self.pin2net_map[pin_order]
        self.pin2node_map = self.pin2node_map[pin_order]
        self.pin_direct = self.pin_direct[pin_order]
        self.pin_offset_x = self.pin_offset_x[pin_order]
        self.pin_offset_y = self.pin_offset_y[pin_order]
        old2new_pin_id_map = np.zeros(len(pin_order), dtype=np.int32)
        for new_pin_id in range(len(pin_order)):
            old2new_pin_id_map[pin_order[new_pin_id]] = new_pin_id
        for i in range(len(self.net2pin_map)):
            for j in range(len(self.net2pin_map[i])):
                self.net2pin_map[i][j] = old2new_pin_id_map[self.net2pin_map[i][j]]
        for i in range(len(self.node2pin_map)):
            for j in range(len(self.node2pin_map[i])):
                self.node2pin_map[i][j] = old2new_pin_id_map[self.node2pin_map[i][j]]
        ## check
        #for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id
        #for node_id in range(len(self.node2pin_map)):
        #    for j in range(len(self.node2pin_map[node_id])):
        #        assert self.pin2node_map[self.node2pin_map[node_id][j]] == node_id

    @property
    def num_movable_nodes(self):
        """
        @return number of movable nodes
        """
        return self.num_physical_nodes - self.num_terminals - self.num_terminal_NIs

    @property
    def num_nodes(self):
        """
        @return number of movable nodes, terminals, terminal_NIs, and fillers
        """
        return self.num_physical_nodes + self.num_filler_nodes

    @property
    def num_nets(self):
        """
        @return number of nets
        """
        return len(self.net2pin_map)

    @property
    def num_pins(self):
        """
        @return number of pins
        """
        return len(self.pin2net_map)

    @property
    def width(self):
        """
        @return width of layout
        """
        return self.xh-self.xl

    @property
    def height(self):
        """
        @return height of layout
        """
        return self.yh-self.yl

    @property
    def area(self):
        """
        @return area of layout
        """
        return self.width*self.height

    def bin_xl(self, id_x):
        """
        @param id_x horizontal index
        @return bin xl
        """
        return self.xl+id_x*self.bin_size_x

    def bin_xh(self, id_x):
        """
        @param id_x horizontal index
        @return bin xh
        """
        return min(self.bin_xl(id_x)+self.bin_size_x, self.xh)

    def bin_yl(self, id_y):
        """
        @param id_y vertical index
        @return bin yl
        """
        return self.yl+id_y*self.bin_size_y

    def bin_yh(self, id_y):
        """
        @param id_y vertical index
        @return bin yh
        """
        return min(self.bin_yl(id_y)+self.bin_size_y, self.yh)

    def num_bins(self, l, h, bin_size):
        """
        @brief compute number of bins
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return number of bins
        """
        return int(np.ceil((h-l)/bin_size))

    def bin_centers(self, l, h, bin_size):
        """
        @brief compute bin centers
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return array of bin centers
        """
        num_bins = self.num_bins(l, h, bin_size)
        centers = np.zeros(num_bins, dtype=self.dtype)
        for id_x in range(num_bins):
            bin_l = l+id_x*bin_size
            bin_h = min(bin_l+bin_size, h)
            centers[id_x] = (bin_l+bin_h)/2
        return centers

    @property
    def routing_grid_size_x(self):
        return (self.routing_grid_xh - self.routing_grid_xl) / self.num_routing_grids_x

    @property
    def routing_grid_size_y(self):
        return (self.routing_grid_yh - self.routing_grid_yl) / self.num_routing_grids_y

    def net_hpwl(self, x, y, net_id):
        """
        @brief compute HPWL of a net
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of a net
        """
        pins = self.net2pin_map[net_id]
        nodes = self.pin2node_map[pins]
        hpwl_x = np.amax(x[nodes]+self.pin_offset_x[pins]) - np.amin(x[nodes]+self.pin_offset_x[pins])
        hpwl_y = np.amax(y[nodes]+self.pin_offset_y[pins]) - np.amin(y[nodes]+self.pin_offset_y[pins])

        return (hpwl_x+hpwl_y)*self.net_weights[net_id]

    def hpwl(self, x, y):
        """
        @brief compute total HPWL
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of all nets
        """
        wl = 0
        for net_id in range(len(self.net2pin_map)):
            wl += self.net_hpwl(x, y, net_id)
        return wl

    def sum_pin_weights(self, weights=None):
        """
        @brief sum pin weights inside a physical node
        @param weights the torch tensor to store node weight data
        """
        if weights is None:
            weights = torch.zeros(
                self.num_nodes,
                dtype=self.net_weights.dtype, device="cpu")
        self.pydb.sum_pin_weights(
            torch.tensor(self.net_weights),
            weights)
        return weights

    def overlap(self, xl1, yl1, xh1, yh1, xl2, yl2, xh2, yh2):
        """
        @brief compute overlap between two boxes
        @return overlap area between two rectangles
        """
        return max(min(xh1, xh2)-max(xl1, xl2), 0.0) * max(min(yh1, yh2)-max(yl1, yl2), 0.0)

    def density_map(self, x, y):
        """
        @brief this density map evaluates the overlap between cell and bins
        @param x horizontal cell locations
        @param y vertical cell locations
        @return density map
        """
        bin_index_xl = np.maximum(np.floor(x/self.bin_size_x).astype(np.int32), 0)
        bin_index_xh = np.minimum(np.ceil((x+self.node_size_x)/self.bin_size_x).astype(np.int32), self.num_bins_x-1)
        bin_index_yl = np.maximum(np.floor(y/self.bin_size_y).astype(np.int32), 0)
        bin_index_yh = np.minimum(np.ceil((y+self.node_size_y)/self.bin_size_y).astype(np.int32), self.num_bins_y-1)

        density_map = np.zeros([self.num_bins_x, self.num_bins_y])

        for node_id in range(self.num_physical_nodes):
            for ix in range(bin_index_xl[node_id], bin_index_xh[node_id]+1):
                for iy in range(bin_index_yl[node_id], bin_index_yh[node_id]+1):
                    density_map[ix, iy] += self.overlap(
                            self.bin_xl(ix), self.bin_yl(iy), self.bin_xh(ix), self.bin_yh(iy),
                            x[node_id], y[node_id], x[node_id]+self.node_size_x[node_id], y[node_id]+self.node_size_y[node_id]
                            )

        for ix in range(self.num_bins_x):
            for iy in range(self.num_bins_y):
                density_map[ix, iy] /= (self.bin_xh(ix)-self.bin_xl(ix))*(self.bin_yh(iy)-self.bin_yl(iy))

        return density_map

    def density_overflow(self, x, y, target_density):
        """
        @brief if density of a bin is larger than target_density, consider as overflow bin
        @param x horizontal cell locations
        @param y vertical cell locations
        @param target_density target density
        @return density overflow cost
        """
        density_map = self.density_map(x, y)
        return np.sum(np.square(np.maximum(density_map-target_density, 0.0)))

    def print_node(self, node_id):
        """
        @brief print node information
        @param node_id cell index
        """
        logging.debug("node %s(%d), size (%g, %g), pos (%g, %g)" % (self.node_names[node_id], node_id, self.node_size_x[node_id], self.node_size_y[node_id], self.node_x[node_id], self.node_y[node_id]))
        pins = "pins "
        for pin_id in self.node2pin_map[node_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def print_net(self, net_id):
        """
        @brief print net information
        @param net_id net index
        """
        logging.debug("net %s(%d)" % (self.net_names[net_id], net_id))
        pins = "pins "
        for pin_id in self.net2pin_map[net_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def print_row(self, row_id):
        """
        @brief print row information
        @param row_id row index
        """
        logging.debug("row %d %s" % (row_id, self.rows[row_id]))
        
    def create_graph_from_netlist(self, params, net2pin_map, pin2node_map, pin_direct, num_nets, num_physical_nodes):
        src = []
        dst = []

        if len(self.flat_region_boxes) > 0:
            placeholder_node_num = len(self.flat_region_boxes)
        print(num_nets)
        
        for net_id in range(num_nets):
            pin_conect = net2pin_map[net_id]
            output_nodes = [pin2node_map[pin_id] for pin_id in pin_conect if pin_direct[pin_id] == b'OUTPUT']
            if len(output_nodes) == 0:
                continue
            input_nodes = [pin2node_map[pin_id] for pin_id in pin_conect if pin_direct[pin_id] == b'INPUT']
      
            if len(self.flat_region_boxes) > 0:
                # if number is larger than self.num_movable_nodes, add placeholder_node_num
                for i in range(len(input_nodes)):
                    if input_nodes[i] >= self.num_movable_nodes:
                        input_nodes[i] += placeholder_node_num 
                    if input_nodes[i] < self.num_movable_nodes and self.node2fence_region_map[input_nodes[i]] < 100:
                        #input_nodes[i] = self.num_movable_nodes + self.node2fence_region_map[input_nodes[i]]
                        input_nodes.append(self.num_movable_nodes + self.node2fence_region_map[input_nodes[i]])
                if output_nodes[0] >= self.num_movable_nodes:
                    output_nodes[0] += placeholder_node_num 
                #if output_nodes[0] < self.num_movable_nodes and self.node2fence_region_map[output_nodes[0]] < 100:
                #   output_nodes[0] = self.num_movable_nodes + self.node2fence_region_map[output_nodes[0]]
                src.extend(input_nodes)
                dst.extend([output_nodes[0]] * len(input_nodes))
                if output_nodes[0] < self.num_movable_nodes and self.node2fence_region_map[output_nodes[0]] < 100:
                    src.extend(input_nodes)
                    dst.extend([int(self.num_movable_nodes + self.node2fence_region_map[output_nodes[0]])] * len(input_nodes))
            else:

                src.extend(input_nodes)
                dst.extend([output_nodes[0]] * len(input_nodes))
          
        src = torch.tensor(src)
        dst = torch.tensor(dst)
        src = src.long()
        dst = dst.long()
        if len(self.flat_region_boxes) > 0:
            #removed_list = list(set(removed_list))
            dgl_graph = dgl.graph((src, dst), num_nodes=num_physical_nodes+placeholder_node_num)
        else:
            dgl_graph = dgl.graph((src, dst), num_nodes=num_physical_nodes)
        
        #import pdb; pdb.set_trace()
        path = "%s/%s" % (params.result_dir, params.design_name())
        if "ispd2015" in params.bench_name():
            additional_dir = params.bench_name().split("/")[2]
            path = "%s/%s" % (params.result_dir, additional_dir)
        posname = "%s/plot/pos.pkl" % (path)

        with open(posname, "rb") as f:
            self.save_pos, self.sizex, self.sizey, _ = pickle.load(f)
            self.save_pos_x = self.save_pos.cpu()[:self.num_physical_nodes]
            self.save_pos_y = self.save_pos.cpu()[len(self.sizex):len(self.sizex)+self.num_physical_nodes]
            self.sizex = self.sizex.cpu()[:self.num_physical_nodes]
            self.sizey = self.sizey.cpu()[:self.num_physical_nodes]
        
        node_num_prev = len(self.sizex)
        cell_nums = self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs

        self.x_position_left = self.save_pos[:cell_nums]
        self.y_position_down = self.save_pos[node_num_prev : node_num_prev+cell_nums]
        self.x_position_right = self.save_pos[:cell_nums] + self.sizex[:cell_nums]
        self.y_position_up = self.save_pos[node_num_prev : node_num_prev+cell_nums] + self.sizey[:cell_nums]

        self.x_position_left_stand = self.x_position_left[:self.num_movable_nodes]
        self.y_position_down_stand = self.y_position_down[:self.num_movable_nodes]
        self.x_position_right_stand = self.x_position_right[:self.num_movable_nodes]
        self.y_position_up_stand = self.y_position_up[:self.num_movable_nodes]

        self.x_position_left_macro = self.x_position_left[self.num_movable_nodes:cell_nums]
        self.y_position_down_macro = self.y_position_down[self.num_movable_nodes:cell_nums]
        self.x_position_right_macro = self.x_position_right[self.num_movable_nodes:cell_nums]
        self.y_position_up_macro = self.y_position_up[self.num_movable_nodes:cell_nums]
 
        self.macro_size_x = self.sizex[self.num_movable_nodes:self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs]
        self.macro_size_y = self.sizey[self.num_movable_nodes:self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs]


        if len(self.flat_region_boxes) > 0:
            self.x_fence_left = torch.tensor([self.flat_region_boxes[i][0]/self.constant  for i in range(len(self.flat_region_boxes))])
            self.y_fence_low = torch.tensor([self.flat_region_boxes[i][1]/self.constant  for i in range(len(self.flat_region_boxes))])
            self.x_fence_right = torch.tensor([self.flat_region_boxes[i][2]/self.constant  for i in range(len(self.flat_region_boxes))])
            self.y_fence_up = torch.tensor([self.flat_region_boxes[i][3]/self.constant  for i in range(len(self.flat_region_boxes))])
        
        feat_save_pos_x = self.save_pos_x.unsqueeze(1)
        feat_save_pos_y = self.save_pos_y.unsqueeze(1)
        feat_sizex_upt = self.sizex.unsqueeze(1)
        feat_sizey_upt = self.sizey.unsqueeze(1)
        if params.add_bert_embed:
            model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
            node_names = [self.node_names[i].decode("utf-8") for i in range(self.num_physical_nodes)]
            cur_map = torch.tensor(self.node2fence_region_map)
            #import pdb; pdb.set_trace()
            if len(self.flat_region_boxes) > 0:
                fence_pos_x = []
                fence_pos_y = []
                fence_size_x = []
                fence_size_y = []
                fence_node_name = []
                count = self.num_terminals  + 1
                for sub_region in self.flat_region_boxes:
                    cur_region = [(sub_region[0]-self.xl)/self.constant, (sub_region[1]-self.yl)/self.constant, (sub_region[2]-self.xl)/self.constant, (sub_region[3]-self.yl)/self.constant]
                    fence_pos_x.append(float(cur_region[0]))
                    fence_pos_y.append(float(cur_region[1]))
                    fence_size_x.append(float(cur_region[2]-cur_region[0]))
                    fence_size_y.append(float(cur_region[3]-cur_region[1]))
                    #import pdb; pdb.set_trace()
                    fence_node_name.append('h'+str(count)+'.DREAMPlace.Shape0')
                    count += 1
                node_names = node_names[:self.num_movable_nodes] + fence_node_name + node_names[self.num_movable_nodes:]
                feat_save_pos_x = torch.cat([feat_save_pos_x[:self.num_movable_nodes], torch.tensor(fence_pos_x).unsqueeze(1), feat_save_pos_x[self.num_movable_nodes:]], dim=0)
                feat_save_pos_y = torch.cat([feat_save_pos_y[:self.num_movable_nodes], torch.tensor(fence_pos_y).unsqueeze(1), feat_save_pos_y[self.num_movable_nodes:]], dim=0)
                feat_sizex_upt = torch.cat([feat_sizex_upt[:self.num_movable_nodes], torch.tensor(fence_size_x).unsqueeze(1), feat_sizex_upt[self.num_movable_nodes:]], dim=0)
                feat_sizey_upt = torch.cat([feat_sizey_upt[:self.num_movable_nodes], torch.tensor(fence_size_y).unsqueeze(1), feat_sizey_upt[self.num_movable_nodes:]], dim=0)
            unique_map = torch.unique(cur_map)
            path = "%s/%s" % (params.result_dir, params.design_name())
            if "ispd2015" in params.bench_name():
                additional_dir = params.bench_name().split("/")[2]
                path = "%s/%s" % (params.result_dir, additional_dir)
            
            if os.path.exists("%s/plot/node_embed.pkl" % (path)):
                node_embed = pickle.load(open("%s/plot/node_embed.pkl" % (path), "rb"))
            else:
                node_embed = model.encode(node_names, batch_size=12800)
                with open("%s/plot/node_embed.pkl" % (path), "wb") as f:
                    pickle.dump(node_embed, f)
           
            for i in range(len(unique_map)):
                if i == len(unique_map)-1:
                    break
                cur_idx = torch.where(cur_map == unique_map[i])[0]
                feat_save_pos_x[cur_idx] = 0
                feat_save_pos_y[cur_idx] = 0
                feat_sizex_upt[cur_idx] = 0
                feat_sizey_upt[cur_idx] = 0    
                for j in cur_idx:
                    node_names[j] = 0
            
            pca = PCA(n_components=4)
            node_embed = pca.fit_transform(node_embed) 
            node_embed = torch.tensor(node_embed)
 
         
        # create node feature
        feat = torch.cat([feat_save_pos_x, feat_save_pos_y, feat_sizex_upt, feat_sizey_upt, node_embed], dim=1)

        if params.feat_ablation == "no_loc":
            feat = torch.cat([feat_sizex_upt, feat_sizey_upt, node_embed], dim=1)
            params.in_dim = 6
        elif params.feat_ablation == "no_size":
            feat = torch.cat([feat_save_pos_x, feat_save_pos_y, node_embed], dim=1)
            params.in_dim = 6
        elif params.feat_ablation == "no_feat":
            feat = torch.cat([feat_save_pos_x, feat_save_pos_y, feat_sizex_upt, feat_sizey_upt], dim=1)
            params.in_dim = 4
        label = []
        region_size = int(params.num_layers-2)*2*torch.min(self.sizey[:self.num_movable_nodes])
        if params.mode == "train":
            if params.select_train_node == "pwlr":        
                with open("./position_dict_final.pkl", "rb") as f:
                    data = pickle.load(f)

                self.train_node_id = list(data.keys())
                pwlr = list(data.values())
                # normalize pwlr to [0, 1]
                for i in range(len(pwlr)):
                    cur_data = pwlr[i]
                    if cur_data > 1+params.label_clip:
                        cur_data = 1
                    elif cur_data < 1:
                        cur_data = 0
                    else:
                        cur_data = (cur_data-1)/params.label_clip

                data_new = {self.train_node_id[i]: pwlr[i] for i in range(len(self.train_node_id))}

                for i in range(self.num_physical_nodes):
                    if i in self.train_node_id:
                        sub_score1 = data_new[i]
                        #sub_score2 = self.get_region_score(self.save_pos, self.sizex, self.sizey, self.save_pos_x[i]-region_size/2, self.save_pos_y[i]-region_size/2, 
                        #                                region_size, params)
                        sub_score = float(sub_score1)#*0.5+ sub_score2*0.5  
                    else:
                        sub_score = 0
                    label.append(sub_score)
                label = torch.tensor(label)
        else:
            # tensor of zero with size self.num_physical_nodes
            label = torch.zeros(self.num_physical_nodes)
        
        if len(self.flat_region_boxes) > 0:
            label = torch.zeros(self.num_physical_nodes+placeholder_node_num)

        dgl_graph.ndata['feat'] = feat
        dgl_graph.ndata['label'] = label

        if params.visual:
            self.visualize(dgl_graph)
        
        #import pdb; pdb.set_trace()
        print("graph construction done! Mode:", params.mode)
        return dgl_graph
    

    def train_gnn(self, params, dgl_graph, model):
        list_of_node = [15, 20, 35, 50, 100, 200, 500, 1000, 2000, 5000]
        dgl_graph = dgl.add_self_loop(dgl_graph)
       
        sampler = dgl.dataloading.NeighborSampler(list_of_node[:params.num_layers])
        # sample full neighor 
        #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(params.num_layers)
        if params.mode == "train":
            label = dgl_graph.ndata['label']   
            #import pdb; pdb.set_trace()
            dataloader =dgl.dataloading.DataLoader(dgl_graph, torch.tensor(self.train_node_id), graph_sampler=sampler, batch_size=params.batch_size, drop_last=False, shuffle=True)
            loss_fcn = torch.nn.L1Loss()
            optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, weight_decay=params.weight_decay, momentum=0.9)
            #optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay, betas=(0.9, 0.999), eps=1e-08)
            if params.gpu:
                model = model.cuda()
                 
            list_loss = []
            model.train()
            for epoch in range(params.num_epochs):
                total_loss = 0
                for step, (input_nodes, output_nodes, mfgs) in enumerate(dataloader):
                    inputs = mfgs[0].srcdata["feat"]
                    labels = mfgs[-1].dstdata["label"]
                    if params.gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    predictions = model(mfgs, inputs, train_state=True)
                    predictions = F.sigmoid(predictions)
                    #import pdb; pdb.set_trace()
                    loss = loss_fcn(predictions, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() 
                    #print("epoch", epoch, "step", step, "loss", loss.item())
                list_loss.append(total_loss/len(dataloader))
                print("Epoch %d | Loss: %.4f" % (epoch, total_loss))

            # plot loss
            path = "%s/%s" % (params.result_dir, params.design_name())
            plt.clf()
            plt.figure()
            plt.plot(list_loss)
            # set y log scale

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.savefig("%s/plot/loss.png" % (path))
            plt.close()
            #import pdb; pdb.set_trace()
            os.makedirs("%s/model" % (path), exist_ok=True)
            if params.ablation:
                torch.save(model.state_dict(), "%s/model/gnn_model_test.pt" % (path))
                #torch.save(model.state_dict(), "%s/model/gnn_model_%s.pt" % (path, params.label_clip))
                #torch.save(model.state_dict(), "%s/model/gnn_model%s_%s.pt" % (path, params.GNN_type, params.lr))
            else:
                torch.save(model.state_dict(), "%s/model/gnn_model_test.pt" % (path))
        else:
            model.load_state_dict(torch.load("./gnn_model.pt"))

        model.eval()
        # set seed
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        dgl.seed(42)
        sampler = dgl.dataloading.NeighborSampler(list_of_node[:params.num_layers])
        #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(params.num_layers)
        valid_dataloader =dgl.dataloading.DataLoader(dgl_graph, torch.arange(self.num_movable_nodes), graph_sampler=sampler, batch_size=params.batch_size, drop_last=False, shuffle=False, num_workers=1)
        
        print("current batch size", params.batch_size, len(valid_dataloader))
        infer_predictions = torch.ones_like(dgl_graph.ndata['label'])
        final_step = 0

        with torch.no_grad():
            for step, (input_nodes, output_nodes, mfgs) in enumerate(valid_dataloader):
                pred = model(mfgs, mfgs[0].srcdata["feat"])
                infer_predictions[output_nodes] = pred.squeeze(1)
        
        print("GNN inference done! Mode:", params.mode)

        print(infer_predictions)
        infer_predictions = infer_predictions.cpu().detach().numpy()

        # avoid the fence region is outside layout
        region_size = int(params.num_layers-2)*2*torch.min(self.sizey[:self.num_movable_nodes])
        avaliable_x = [region_size/2, (self.xh-self.xl)/self.constant-region_size/2]
        avaliable_y = [region_size/2, (self.yh-self.yl)/self.constant-region_size/2]

        index_out_x1 = np.where((self.save_pos_x[:self.num_movable_nodes]-region_size/2)<avaliable_x[0])[0]
        index_out_y1 = np.where((self.save_pos_y[:self.num_movable_nodes]-region_size/2)<avaliable_y[0])[0]
        index_out_x2 = np.where((self.save_pos_x[:self.num_movable_nodes]+region_size/2)>avaliable_x[1])[0]
        index_out_y2 = np.where((self.save_pos_y[:self.num_movable_nodes]+region_size/2)>avaliable_y[1])[0]
        infer_predictions[index_out_x1] = max(infer_predictions)
        infer_predictions[index_out_y1] = max(infer_predictions)
        infer_predictions[index_out_x2] = max(infer_predictions)
        infer_predictions[index_out_y2] = max(infer_predictions)

        cur_map = torch.tensor(self.node2fence_region_map)
        # get unique value from cur_map
        unique_map = torch.unique(cur_map)
        if len(unique_map) > 1:
            for i in range(len(unique_map)):
                if i == len(unique_map)-1:
                    break
                cur_idx = torch.where(cur_map == unique_map[i])[0]
                infer_predictions[cur_idx] = max(infer_predictions)
            #import pdb; pdb.set_trace()
            for i in range(len(self.flat_region_boxes)):
                cur_region = self.flat_region_boxes[i]
                cur_region_norm = [(cur_region[0]-self.xl)/self.constant, (cur_region[1]-self.yl)/self.constant, 
                    (cur_region[2]-self.xl)/self.constant, (cur_region[3]-self.yl)/self.constant]
                # convert the region 
               
                region_x = [cur_region_norm[0]-region_size, cur_region_norm[2]]
                region_y = [cur_region_norm[1]-region_size, cur_region_norm[3]]
                index_x = np.where((self.save_pos_x>region_x[0]) & (self.save_pos_x<region_x[1]))[0]
                index_y = np.where((self.save_pos_y>region_y[0]) & (self.save_pos_y<region_y[1]))[0]
                index = np.intersect1d(index_x, index_y)
                infer_predictions[index] = max(infer_predictions)

        def find_min_score_node(dgl_graph, infer_predictions, num_hops=20):
            def message_func(edges):
                # remove the edges that are not in the movable nodes
                #import pdb; pdb.set_trace()
                return {'score': edges.src['score']}

            def reduce_func(nodes):
                #import pdb; pdb.set_trace()
                return {'neighbor_score': torch.mean(nodes.mailbox['score'], dim=1)}

            with dgl_graph.local_scope():
                dgl_graph.ndata['score'] = infer_predictions
                for _ in range(num_hops):
                    dgl_graph.update_all(message_func, reduce_func)
                    dgl_graph.ndata['score'] = dgl_graph.ndata['neighbor_score']
                neighbor_scores = dgl_graph.ndata['score']
            final_scores = infer_predictions + neighbor_scores * params.abla_gamma

            min_score, min_node_id = torch.min(final_scores[:self.num_movable_nodes], dim=0)
            neighbor_min_score,  neighbor_min_node_id = torch.min(neighbor_scores[:self.num_movable_nodes], dim=0)
             
            return min_node_id.item(), min_score.item(), neighbor_min_node_id.item(), neighbor_min_score.item()

        if self.num_physical_nodes > 200000 :
            infer_predictions = torch.tensor(infer_predictions)
            num_hops=20
            min_idx, _, _, _ = find_min_score_node(dgl_graph, infer_predictions, num_hops=num_hops)
        else:
            min_idx = np.argmin(infer_predictions[:self.num_movable_nodes])

        #infer_predictions[min_idx] = max(infer_predictions)
        #min_idx = np.argmin(infer_predictions[:self.num_movable_nodes])
        #infer_predictions[min_idx] = max(infer_predictions)
        #min_idx = np.argmin(infer_predictions[:self.num_movable_nodes])
        print("min_idx", min_idx, infer_predictions[min_idx])
        #import pdb; pdb.set_trace()
        return min_idx


    def get_cell_in_region(self, position, sizex, sizey, x, y, width, height, params):
        '''
        position: axis of cells
        sizex: size of cells in x direction
        '''
        cell_idx = []
        node_num_prev = len(sizex)

        print("position", position.size(), sizex.size(), sizey.size(), self.num_movable_nodes, node_num_prev)
        x_position_right = position[:self.num_movable_nodes]+ sizex[:self.num_movable_nodes]
        x_position_left = position[:self.num_movable_nodes] #+ sizex[:self.num_movable_nodes]/2

        y_position_up = position[node_num_prev:node_num_prev+self.num_movable_nodes] + sizey[:self.num_movable_nodes]
        y_position_down = position[node_num_prev:node_num_prev+self.num_movable_nodes] #+ sizey[:self.num_movable_nodes]/2

        #x_inter_idx = np.where((x_position_right <= x+width) & ( x_position_left>= (x)))[0]
        #y_inter_idx = np.where((y_position_up <= y+height) & (y_position_down > = (y)))[0]

        x_inter_idx = np.where((x_position_right < x+width) & ( x_position_left > (x)))[0]
        y_inter_idx = np.where((y_position_up < y+height) & (y_position_down > (y)))[0]

        cell_idx = np.intersect1d(x_inter_idx, y_inter_idx)

        #import pdb; pdb.set_trace()
        total_region = width * height
        cell_region = torch.sum(sizex[cell_idx] * sizey[cell_idx]).item()
        return cell_idx, cell_region/total_region, len(cell_idx)
    
    def initialize_from_rawdb(self, params):
        """
        @brief initialize data members from raw database
        @param params parameters
        """
        pydb = place_io.PlaceIOFunction.pydb(self.rawdb)

        self.num_physical_nodes = pydb.num_nodes
        self.num_terminals = pydb.num_terminals
        self.num_terminal_NIs = pydb.num_terminal_NIs
        self.node_name2id_map = pydb.node_name2id_map
        self.node_names = np.array(pydb.node_names, dtype=np.string_)
        self.movable_cell_start = self.num_physical_nodes - self.num_terminals - self.num_terminal_NIs
        #import pdb; pdb.set_trace()
        self.xl = float(pydb.xl)
        self.yl = float(pydb.yl)
        self.xh = float(pydb.xh)
        self.yh = float(pydb.yh)

        self.constant = 1

        if "adaptec" in params.design_name() or "bigblue" in params.design_name():
            self.constant = 1
        elif "mgc_superblue" in params.bench_name() or "floorplan" == params.design_name():
            self.constant = 100
        elif  "iccad2015.ot" in params.lef_input:
            self.constant = 380
        else:
            self.constant = 200
            
        print("self.xl, self.yl, self.xh, self.yh", self.xl, self.yl, self.xh, self.yh)
        self.row_height = float(pydb.row_height)
        self.site_width = float(pydb.site_width)
        print("self.row_height, self.site_width", self.row_height, self.site_width)

        use_read_pl_flag = False
        if (not params.global_place_flag) and os.path.exists(params.aux_input):
            filename = None
            with open(params.aux_input, "r") as f:
                for line in f:
                    line = line.strip()
                    if ".pl" in line:
                        tokens = line.split()
                        for token in tokens:
                            if token.endswith(".pl"):
                                filename = token
                                break
            filename = os.path.join(os.path.dirname(params.aux_input), filename)
            if filename is not None and os.path.exists(filename):
                self.node_x = np.zeros(self.num_physical_nodes, dtype=self.dtype)
                self.node_y = np.zeros(self.num_physical_nodes, dtype=self.dtype)
                self.node_orient = np.zeros(self.num_physical_nodes, dtype=np.string_)
                self.read_pl(params, filename)
                use_read_pl_flag = True
        
        if not use_read_pl_flag:
            self.node_x = np.array(pydb.node_x, dtype=self.dtype)
            self.node_y = np.array(pydb.node_y, dtype=self.dtype)
            self.node_orient = np.array(pydb.node_orient, dtype=np.string_)
        self.node_size_x = np.array(pydb.node_size_x, dtype=self.dtype)
        self.node_size_y = np.array(pydb.node_size_y, dtype=self.dtype)
        self.node2orig_node_map = np.array(pydb.node2orig_node_map, dtype=np.int32)
        

        self.net_name2id_map = pydb.net_name2id_map
        self.net_names = np.array(pydb.net_names, dtype=np.string_)
        self.net2pin_map = pydb.net2pin_map
        self.pin_direct = np.array(pydb.pin_direct, dtype=np.string_)
        self.pin_offset_x = np.array(pydb.pin_offset_x, dtype=self.dtype)
        self.pin_offset_y = np.array(pydb.pin_offset_y, dtype=self.dtype)
        
        self.flat_net2pin_map = np.array(pydb.flat_net2pin_map, dtype=np.int32)
        self.flat_net2pin_start_map = np.array(pydb.flat_net2pin_start_map, dtype=np.int32)
        self.pin_names = np.array(pydb.pin_names, dtype=np.string_)
        self.pin_name2id_map = pydb.pin_name2id_map
        self.node2pin_map = pydb.node2pin_map
        self.flat_node2pin_map = np.array(pydb.flat_node2pin_map, dtype=np.int32) # TO DO : update this two
        self.flat_node2pin_start_map = np.array(pydb.flat_node2pin_start_map, dtype=np.int32)
        self.pin2node_map = np.array(pydb.pin2node_map, dtype=np.int32)
        self.pin2net_map = np.array(pydb.pin2net_map, dtype=np.int32)
        
        self.rows = np.array(pydb.rows, dtype=self.dtype)
        self.regions = pydb.regions
        self.net_weights = np.array(pydb.net_weights, dtype=self.dtype)
        self.net_weight_deltas = np.array(pydb.net_weight_deltas, dtype=self.dtype)
        self.net_criticality = np.array(pydb.net_criticality, dtype=self.dtype)
        self.net_criticality_deltas = np.array(pydb.net_criticality_deltas, dtype=self.dtype)
        self.node2fence_region_map = np.array(pydb.node2fence_region_map, dtype=np.int32)
        for i in range(len(self.regions)):
            self.regions[i] = np.array(self.regions[i], dtype=self.dtype)

        #### For watermark:
        #### Add additional fence region for watermarking

        self.flat_region_boxes = np.array(pydb.flat_region_boxes, dtype=self.dtype)
        
        print("self.flat_region_boxes ", self.flat_region_boxes )
        print("self.regions", self.regions)
        wm_start = []
        wm_cells = []
        region_ratios = []
        wm_cell_nums = []
        start_num = 0
        if (params.watermark_flag and params.watermark_type == "gnn" and params.phase == 2) or  (params.watermark_flag and params.watermark_type == "benchmark") or params.watermark_type == "combine":
            path = "%s/%s" % (params.result_dir, params.design_name())
            if "ispd2015" in params.bench_name():
                additional_dir = params.bench_name().split("/")[2]
                path = "%s/%s" % (params.result_dir, additional_dir)
            figname = "%s/plot/pos.pkl" % (path)
            with open(figname, "rb") as f:
                save_pos, sizex, sizey, cell_names = pickle.load(f)
                if save_pos.is_cuda:
                    save_pos = save_pos.cpu()
                if sizex.is_cuda:
                    sizex = sizex.cpu()
                if sizey.is_cuda:
                    sizey = sizey.cpu()
            self.sizex, self.sizey, self.save_pos = sizex, sizey, save_pos

        if (params.watermark_flag and params.watermark_type == "global" and params.globa_wm_phase == 1) or params.watermark_type == "combine":   
            height = torch.min(sizey[:self.num_movable_nodes])
            params.fence_region_size[0] = params.fence_region_size[0]*height
            params.fence_region_stride = params.fence_region_stride*height
            import time 
            start_time = time.time()
            x,y = self.score_fence_region(save_pos, sizex, sizey, params.fence_region_size[0], params.fence_region_stride, params)
            x_tmp = x#-torch.min(self.x_position_left)
            y_tmp = y#-torch.min(self.y_position_down)
            #self.constant = (self.xh - self.xl) / (torch.max(self.x_position_right-torch.min(self.x_position_left)))

            #x = x - torch.min(self.x_position_left)
            #y = y - torch.min(self.y_position_bottom)
            print("initial x y", x, y)
            region_tuple = [(x_tmp*self.constant +self.xl, y_tmp*self.constant +self.yl, 
                    (x_tmp+params.fence_region_size[0])*self.constant +self.xl, 
                    (y_tmp+params.fence_region_size[0])*self.constant +self.yl)]
            
            self.regions.append(np.array(region_tuple, dtype=self.dtype))
            if params.timing_opt_flag:
                wm_cell, region_ratio, wm_cell_num= self.get_cell_in_region_timing(save_pos, sizex, sizey, x-int(params.fence_region_size[0]/2), y-int(params.fence_region_size[0]/2), params.fence_region_size[0], params.fence_region_size[0], params)
            else:
                wm_cell, region_ratio, wm_cell_num= self.get_cell_in_region(save_pos, sizex, sizey, x-int(params.fence_region_size[0]/2), y-int(params.fence_region_size[0]/2), params.fence_region_size[0], params.fence_region_size[0], params)
            end_time = time.time()
            with open(path+"/plot/time_combine.txt", "a") as f:
                f.write("search time: %f\n" % (end_time-start_time))

            if len(params.wm_cell_ratio_ablation) > 0:
                wm_cell = wm_cell[:int(len(wm_cell)*params.wm_cell_ratio_ablation[i])]
            wm_cells.append(wm_cell)
            region_ratios.append(region_ratio*100)
            wm_cell_nums.append(wm_cell_num)
             
            wm_start.append((x, y))
            print("wm_cell_nums: ", wm_cell_nums)
            print("wm_cell: ", wm_cell)
            print("after wm", self.regions)
            params.wm_start = wm_start
            params.save_pos = save_pos
            params.region_ratios = region_ratios
            params.wm_cell_nums = wm_cell_nums
        elif (params.watermark_flag and params.watermark_type == "gnn" and params.phase == 2) or (params.watermark_flag and params.watermark_type == "benchmark" and params.phase == 1):
            
            import time
            start_time = time.time()
            self.graph = self.create_graph_from_netlist(params, self.net2pin_map, self.pin2node_map, self.pin_direct, self.num_nets, self.num_physical_nodes)
            if params.GNN_type == "GCN":
                self.model = GCN(params.in_dim, params.hidden_dim, params.out_dim, params.num_layers, F.relu)
            elif params.GNN_type == "GAT":
                self.model = GAT(params.in_dim, params.hidden_dim, params.out_dim, params.num_layers, F.relu)
            elif params.GNN_type == "SAGE":
                self.model = SAGE(params.in_dim, params.hidden_dim, params.out_dim, params.num_layers, F.relu)
            
            if self.num_physical_nodes > 200000:
                region_size = int(params.num_layers-2)*2*torch.min(self.sizey[:self.num_movable_nodes])*5
                if "superblue12" in params.bench_name():
                    region_size = int(region_size/2)
          
                if "superblue16" in params.bench_name():
                    region_size = int(region_size/10)
                    params.abla_gamma = 0
                if "superblue11" in params.bench_name():
                    region_size = int(region_size/10)
                    params.abla_gamma = 0
        
                if "floorplan" == params.design_name():
                    region_size = int(region_size/10)
                    params.abla_gamma = 0
            else:
                region_size = int(params.num_layers-2)*2*torch.min(self.sizey[:self.num_movable_nodes])
                if "mgc_pci_bridge32_b" in params.bench_name():
                    region_size = int(region_size/2)
                if "mgc_fft_b" in params.bench_name():
                    region_size = int(region_size/2)
            min_idx = self.train_gnn(params, self.graph, self.model)

            params.fence_region_size[0] = region_size
            x, y = self.save_pos_x[min_idx], self.save_pos_y[min_idx]
            x_temp, y_temp = x, y#-int(params.fence_region_size[0]/2), y-int(params.fence_region_size[0]/2)
            region_tuple = [(x_temp*self.constant +self.xl, y_temp*self.constant +self.yl, 
                    (x_temp+params.fence_region_size[0])*self.constant +self.xl, 
                    (y_temp+params.fence_region_size[0])*self.constant +self.yl)]
            #import pdb; pdb.set_trace()
            self.regions.append(np.array(region_tuple, dtype=self.dtype))
            if params.timing_opt_flag:
                wm_cell, region_ratio, wm_cell_num= self.get_cell_in_region_timing(save_pos, sizex, sizey, x-int(params.fence_region_size[0]/2), y-int(params.fence_region_size[0]/2), params.fence_region_size[0], params.fence_region_size[0], params)
            else:
                wm_cell, region_ratio, wm_cell_num= self.get_cell_in_region(save_pos, sizex, sizey, x-int(params.fence_region_size[0]/2), y-int(params.fence_region_size[0]/2), params.fence_region_size[0], params.fence_region_size[0], params)
            
            end_time = time.time()
            path = "%s/%s" % (params.result_dir, params.design_name())
            if "ispd2015" in params.bench_name():
                additional_dir = params.bench_name().split("/")[2]
                path = "%s/%s" % (params.result_dir, additional_dir)
            with open(path+"/plot/time_combine.txt", "a") as f:
                f.write("search time: %f\n" % (end_time-start_time))

            wm_cells.append(wm_cell)
            #import pdb; pdb.set_trace()
            #self.node2fence_region_map[wm_cell]
            
            region_ratios.append(region_ratio*100)
            wm_cell_nums.append(wm_cell_num)
            
            wm_start.append((x, y))
            print("wm_cell_nums: ", wm_cell_nums)
            print("wm_cell: ", wm_cell)
            print("after wm", self.regions)
            params.wm_start = wm_start
            params.save_pos = save_pos
            params.region_ratios = region_ratios
            params.wm_cell_nums = wm_cell_nums
            self.wm_candidate = wm_cell
            #import pdb; pdb.set_trace()
        elif params.watermark_type == "benchmark" and params.phase == 2:
            region_size = int(params.num_layers-2)*2*torch.min(self.sizey[:self.num_movable_nodes])
            params.fence_region_size[0] = region_size

            x, y = params.position_x, params.position_y
            x_temp, y_temp = x, y
            region_tuple = [(x_temp*self.constant +self.xl, y_temp*self.constant +self.yl, 
                    (x_temp+params.fence_region_size[0])*self.constant +self.xl, 
                    (y_temp+params.fence_region_size[0])*self.constant +self.yl)]
            #import pdb; pdb.set_trace()
            self.regions.append(np.array(region_tuple, dtype=self.dtype))
            if params.timing_opt_flag:
                wm_cell, region_ratio, wm_cell_num= self.get_cell_in_region_timing(save_pos, sizex, sizey, x-int(params.fence_region_size[0]/2), y-int(params.fence_region_size[0]/2), params.fence_region_size[0], params.fence_region_size[0], params)
            else:
                wm_cell, region_ratio, wm_cell_num= self.get_cell_in_region(save_pos, sizex, sizey, x-int(params.fence_region_size[0]/2), y-int(params.fence_region_size[0]/2), params.fence_region_size[0], params.fence_region_size[0], params)
            
            wm_cells.append(wm_cell)
            
            region_ratios.append(region_ratio*100)
            wm_cell_nums.append(wm_cell_num)
            
            wm_start.append((x, y))
            print("wm_cell_nums: ", wm_cell_nums)
            print("wm_cell: ", wm_cell)
            print("after wm", self.regions)
            params.wm_start = wm_start
            params.save_pos = save_pos
            params.region_ratios = region_ratios
            params.wm_cell_nums = wm_cell_nums
            self.wm_candidate = wm_cell
        else:
            params.wm_start = None
 
        
        #### For watermark:
        #### Add flat region box
        self.flat_region_boxes_ori = self.flat_region_boxes 
        if  (params.watermark_flag and params.watermark_type == "global" and params.globa_wm_phase == 1) or params.watermark_type == "combine" or (params.watermark_flag and params.watermark_type == "gnn"and params.phase == 2) or params.watermark_type == "benchmark":
            assert params.fence_region_num == len(params.fence_region_size)
            for i in range(params.fence_region_num):
                if len(wm_cells[i]) > 0:
                    #region_tuple = [(wm_start[i][0]*200, wm_start[i][1]*200, (wm_start[i][0]+params.fence_region_size[i])*200, (wm_start[i][1]+params.fence_region_size[i])*200)]
                    #region_tuple = [(wm_start[i][0], wm_start[i][1], (wm_start[i][0]+params.fence_region_size[i]), (wm_start[i][1]+params.fence_region_size[i]))]
                    if "adaptec" in params.design_name() or "bigblue" in params.design_name():
                        region_tuple = region_tuple
                    else:
                        region_tuple = region_tuple
                    
                    
                    print("self.flat_region_boxes", self.flat_region_boxes, region_tuple)
                    if self.flat_region_boxes.shape[0] == 0:
                        
                        self.flat_region_boxes = np.array(region_tuple, dtype=self.dtype)
                    else:
                        self.flat_region_boxes = np.concatenate((self.flat_region_boxes, np.array(region_tuple, dtype=self.dtype)), axis=0)
            
        self.flat_region_boxes_start = np.array(pydb.flat_region_boxes_start, dtype=np.int32)
        self.flat_region_boxes_start_ori = self.flat_region_boxes_start
        start_num = len(self.flat_region_boxes_start)-1
        
        if  (params.watermark_flag and params.watermark_type == "global" and params.globa_wm_phase == 1) or params.watermark_type == "combine"or (params.watermark_flag and params.watermark_type == "gnn"and params.phase == 2) or params.watermark_type == "benchmark":
            assert params.fence_region_num == len(params.fence_region_size)
            for i in range(params.fence_region_num):
                if len(wm_cells[i]) > 0:
                    self.flat_region_boxes_start = np.concatenate((self.flat_region_boxes_start, np.array([len(self.flat_region_boxes)], dtype=np.int32)), axis=0)
     
        ####h For watermark:
        #### Set which cells are in the fence region
        self.node2fence_region_map_ori = self.node2fence_region_map
        if  (params.watermark_flag and params.watermark_type == "global" and params.globa_wm_phase == 1) or params.watermark_type == "combine" or (params.watermark_flag and params.watermark_type == "gnn"and params.phase == 2) or params.watermark_type == "benchmark":
            size_region_num = len(self.node2fence_region_map)
            for i in range(params.fence_region_num):
                if len(wm_cells[i]) > 0:
                    self.node2fence_region_map[wm_cells[i]]  = start_num+i
        params.wm_cells = wm_cells
        print("self.node2fence_region_map", self.node2fence_region_map, self.node2fence_region_map.shape)
 
        self.num_movable_pins = pydb.num_movable_pins
        self.total_space_area = float(pydb.total_space_area)

        self.routing_grid_xl = float(pydb.routing_grid_xl)
        self.routing_grid_yl = float(pydb.routing_grid_yl)
        self.routing_grid_xh = float(pydb.routing_grid_xh)
        self.routing_grid_yh = float(pydb.routing_grid_yh)
        print("pydb.num_routing_grids_x", pydb.num_routing_grids_x)
        if pydb.num_routing_grids_x:
            print("pydb.unit_horizontal_capacities", pydb.unit_horizontal_capacities)
            self.num_routing_grids_x = pydb.num_routing_grids_x
            self.num_routing_grids_y = pydb.num_routing_grids_y
            self.num_routing_layers = len(pydb.unit_horizontal_capacities)
            self.unit_horizontal_capacity = np.array(pydb.unit_horizontal_capacities, dtype=self.dtype).sum()
            self.unit_vertical_capacity = np.array(pydb.unit_vertical_capacities, dtype=self.dtype).sum()
            self.unit_horizontal_capacities = np.array(pydb.unit_horizontal_capacities, dtype=self.dtype)
            self.unit_vertical_capacities = np.array(pydb.unit_vertical_capacities, dtype=self.dtype)
            self.initial_horizontal_demand_map = np.array(pydb.initial_horizontal_demand_map, dtype=self.dtype).reshape((-1, self.num_routing_grids_x, self.num_routing_grids_y)).sum(axis=0)
            self.initial_vertical_demand_map = np.array(pydb.initial_vertical_demand_map, dtype=self.dtype).reshape((-1, self.num_routing_grids_x, self.num_routing_grids_y)).sum(axis=0)
        else:
            self.num_routing_grids_x = params.route_num_bins_x
            self.num_routing_grids_y = params.route_num_bins_y
            self.num_routing_layers = 1
            self.unit_horizontal_capacity = params.unit_horizontal_capacity
            self.unit_vertical_capacity = params.unit_vertical_capacity

            self.unit_horizontal_capacities = np.array([params.unit_horizontal_capacity], dtype=self.dtype)
            self.unit_vertical_capacities = np.array([params.unit_vertical_capacity], dtype=self.dtype)

        # convert node2pin_map to array of array
        for i in range(len(self.node2pin_map)):
            self.node2pin_map[i] = np.array(self.node2pin_map[i], dtype=np.int32)
        self.node2pin_map = np.array(self.node2pin_map)

        # convert net2pin_map to array of array
        for i in range(len(self.net2pin_map)):
            self.net2pin_map[i] = np.array(self.net2pin_map[i], dtype=np.int32)
        self.net2pin_map = np.array(self.net2pin_map)
        self.max_net_weight = np.float64(params.max_net_weight)
 
    def read(self, params):
        """
        @brief read using c++
        @param params parameters
        """
        self.dtype = datatypes[params.dtype]
        self.rawdb = place_io.PlaceIOFunction.read(params)
        self.initialize_from_rawdb(params)

    def __call__(self, params):
        """
        @brief top API to read placement files
        @param params parameters
        """
        tt = time.time()

        self.read(params)
        self.initialize(params)

        logging.info("reading benchmark takes %g seconds" % (time.time()-tt))

    def initialize_num_bins(self, params):
        """
        @brief initialize number of bins with a heuristic method, which many not be optimal.
        The heuristic is adapted form RePlAce, 2x2 to 4096x4096. 
        """
        # derive bin dimensions by keeping the aspect ratio 
        # this bin setting is not for global placement, only for other steps 
        # global placement has its bin settings defined in global_place_stages
        if params.num_bins_x <= 1 or params.num_bins_y <= 1: 
            total_bin_area = self.area
            avg_movable_area = self.total_movable_node_area / self.num_movable_nodes
            ideal_bin_area = avg_movable_area / params.target_density
            ideal_num_bins = total_bin_area / ideal_bin_area
            if (ideal_num_bins < 4): # smallest number of bins 
                ideal_num_bins = 4
            aspect_ratio = (self.yh - self.yl) / (self.xh - self.xl)
            y_over_x = True 
            if aspect_ratio < 1: 
                aspect_ratio = 1.0 / aspect_ratio
                y_over_x = False 
            aspect_ratio = int(math.pow(2, round(math.log2(aspect_ratio))))
            num_bins_1d = 2 # min(num_bins_x, num_bins_y)
            while num_bins_1d <= 4096:
                found_num_bins = num_bins_1d * num_bins_1d * aspect_ratio
                if (found_num_bins > ideal_num_bins / 4 and found_num_bins <= ideal_num_bins): 
                    break 
                num_bins_1d *= 2
            if y_over_x:
                self.num_bins_x = num_bins_1d
                self.num_bins_y = num_bins_1d * aspect_ratio
            else:
                self.num_bins_x = num_bins_1d * aspect_ratio
                self.num_bins_y = num_bins_1d
            params.num_bins_x = self.num_bins_x
            params.num_bins_y = self.num_bins_y
        else:
            self.num_bins_x = params.num_bins_x
            self.num_bins_y = params.num_bins_y


    def calc_num_filler_for_fence_region(self, region_id, node2fence_region_map, target_density):
        """
        @description: calculate number of fillers for each fence region
        @param fence_regions{type}
        @return:
        """
        print("region_id", region_id, node2fence_region_map, self.num_movable_nodes)
        num_regions = len(self.regions)
        node2fence_region_map = node2fence_region_map[: self.num_movable_nodes]
        if region_id < len(self.regions):
            fence_region_mask = node2fence_region_map == region_id
        else:
            fence_region_mask = node2fence_region_map >= len(self.regions)

        num_movable_nodes = self.num_movable_nodes

       
        movable_node_size_x = self.node_size_x[:num_movable_nodes][fence_region_mask]
        # movable_node_size_y = self.node_size_y[:num_movable_nodes][fence_region_mask]
        print("movable_node_size_x", movable_node_size_x)

        lower_bound = np.percentile(movable_node_size_x, 5)
        upper_bound = np.percentile(movable_node_size_x, 95)
        filler_size_x = np.mean(
            movable_node_size_x[(movable_node_size_x >= lower_bound) & (movable_node_size_x <= upper_bound)]
        )
        filler_size_y = self.row_height

        area = (self.xh - self.xl) * (self.yh - self.yl)

        total_movable_node_area = np.sum(
            self.node_size_x[:num_movable_nodes][fence_region_mask]
            * self.node_size_y[:num_movable_nodes][fence_region_mask]
        )

        if region_id < num_regions:
            ## placeable area is not just fention region area. Macros can have overlap with fence region. But we approximate by this method temporarily
            region = self.regions[region_id]
            placeable_area = np.sum((region[:, 2] - region[:, 0]) * (region[:, 3] - region[:, 1]))
        else:
            ### invalid area outside the region, excluding macros? ignore overlap between fence region and macro
            fence_regions = np.concatenate(self.regions, 0).astype(np.float32)
            fence_regions_size_x = fence_regions[:, 2] - fence_regions[:, 0]
            fence_regions_size_y = fence_regions[:, 3] - fence_regions[:, 1]
            fence_region_area = np.sum(fence_regions_size_x * fence_regions_size_y)

            placeable_area = (
                max(self.total_space_area, self.area - self.total_fixed_node_area) - fence_region_area
            )

        ### recompute target density based on the region utilization
        utilization = min(total_movable_node_area / placeable_area, 1.0)
        if target_density < utilization:
            ### add a few fillers to avoid divergence
            target_density_fence_region = min(1, utilization + 0.01)
        else:
            target_density_fence_region = target_density

        target_density_fence_region = max(0.35, target_density_fence_region)

        total_filler_node_area = max(placeable_area * target_density_fence_region - total_movable_node_area, 0.0)

        num_filler = int(round(total_filler_node_area / (filler_size_x * filler_size_y)))
        logging.info(
            "Region:%2d movable_node_area =%10.1f, placeable_area =%10.1f, utilization =%.3f, filler_node_area =%10.1f, #fillers =%8d, filler sizes =%2.4gx%g\n"
            % (
                region_id,
                total_movable_node_area,
                placeable_area,
                utilization,
                total_filler_node_area,
                num_filler,
                filler_size_x,
                filler_size_y,
            )
        )

        return (
            num_filler,
            target_density_fence_region,
            filler_size_x,
            filler_size_y,
            total_movable_node_area,
            np.sum(fence_region_mask.astype(np.float32)),
        )

    def initialize(self, params):
        """
        @brief initialize data members after reading
        @param params parameters
        """

        # shift and scale
        # adjust shift_factor and scale_factor if not set
        params.shift_factor[0] = self.xl
        params.shift_factor[1] = self.yl
        logging.info("set shift_factor = (%g, %g), as original row bbox = (%g, %g, %g, %g)" 
                % (params.shift_factor[0], params.shift_factor[1], self.xl, self.yl, self.xh, self.yh))
        if params.scale_factor == 0.0 or self.site_width != 1.0:
            params.scale_factor = 1.0 / self.site_width
        logging.info("set scale_factor = %g, as site_width = %g" % (params.scale_factor, self.site_width))
        self.scale(params.shift_factor, params.scale_factor)

        content = """
================================= Benchmark Statistics =================================
#nodes = %d, #terminals = %d, # terminal_NIs = %d, #movable = %d, #nets = %d
die area = (%g, %g, %g, %g) %g
row height = %g, site width = %g
""" % (
                self.num_physical_nodes, self.num_terminals, self.num_terminal_NIs, self.num_movable_nodes, len(self.net_names),
                self.xl, self.yl, self.xh, self.yh, self.area,
                self.row_height, self.site_width
                )

        # set num_movable_pins
        if self.num_movable_pins is None:
            self.num_movable_pins = 0
            for node_id in self.pin2node_map:
                if node_id < self.num_movable_nodes:
                    self.num_movable_pins += 1
        content += "#pins = %d, #movable_pins = %d\n" % (self.num_pins, self.num_movable_pins)
        # set total cell area
        self.total_movable_node_area = float(np.sum(self.node_size_x[:self.num_movable_nodes]*self.node_size_y[:self.num_movable_nodes]))
        # total fixed node area should exclude the area outside the layout and the area of terminal_NIs
        self.total_fixed_node_area = float(np.sum(
                np.maximum(
                    np.minimum(self.node_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs] + self.node_size_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.xh)
                    - np.maximum(self.node_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.xl),
                    0.0) * np.maximum(
                        np.minimum(self.node_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs] + self.node_size_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.yh)
                        - np.maximum(self.node_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.yl),
                        0.0)
                ))
        content += "total_movable_node_area = %g, total_fixed_node_area = %g, total_space_area = %g\n" % (self.total_movable_node_area, self.total_fixed_node_area, self.total_space_area)

        target_density = min(self.total_movable_node_area / self.total_space_area, 1.0)
        if target_density > params.target_density:
            logging.warning("target_density %g is smaller than utilization %g, ignored" % (params.target_density, target_density))
            params.target_density = target_density
        content += "utilization = %g, target_density = %g\n" % (self.total_movable_node_area / self.total_space_area, params.target_density)

        # calculate fence region virtual macro
        if len(self.regions) > 0:
            virtual_macro_for_fence_region = [
                fence_region.slice_non_fence_region(
                    region,
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    merge=True,
                    plot=False,
                    figname=f"vmacro_{region_id}_merged.png",
                    device="cpu",
                    macro_pos_x=self.node_x[self.num_movable_nodes : self.num_movable_nodes + self.num_terminals],
                    macro_pos_y=self.node_y[self.num_movable_nodes : self.num_movable_nodes + self.num_terminals],
                    macro_size_x=self.node_size_x[
                        self.num_movable_nodes : self.num_movable_nodes + self.num_terminals
                    ],
                    macro_size_y=self.node_size_y[
                        self.num_movable_nodes : self.num_movable_nodes + self.num_terminals
                    ],
                )
                .cpu()
                .numpy()
                for region_id, region in enumerate(self.regions)
            ]
            virtual_macro_for_non_fence_region = np.concatenate(self.regions, 0)
           
            self.virtual_macro_fence_region = virtual_macro_for_fence_region + [virtual_macro_for_non_fence_region]


        if len(self.regions) > 0:
            ### calculate fillers if there is fence region
            self.filler_size_x_fence_region = []
            self.filler_size_y_fence_region = []
            self.num_filler_nodes = 0
            self.num_filler_nodes_fence_region = []
            self.num_movable_nodes_fence_region = []
            self.total_movable_node_area_fence_region = []
            self.target_density_fence_region = []
            self.filler_start_map = None
            filler_node_size_x_list = []
            filler_node_size_y_list = []
            self.total_filler_node_area = 0
            for i in range(len(self.regions) + 1):
                (
                    num_filler_i,
                    target_density_i,
                    filler_size_x_i,
                    filler_size_y_i,
                    total_movable_node_area_i,
                    num_movable_nodes_i,
                ) = self.calc_num_filler_for_fence_region(i, self.node2fence_region_map, params.target_density)
                self.num_movable_nodes_fence_region.append(num_movable_nodes_i)
                self.num_filler_nodes_fence_region.append(num_filler_i)
                self.total_movable_node_area_fence_region.append(total_movable_node_area_i)
                self.target_density_fence_region.append(target_density_i)
                self.filler_size_x_fence_region.append(filler_size_x_i)
                self.filler_size_y_fence_region.append(filler_size_y_i)
                self.num_filler_nodes += num_filler_i
                filler_node_size_x_list.append(
                    np.full(num_filler_i, fill_value=filler_size_x_i, dtype=self.node_size_x.dtype)
                )
                filler_node_size_y_list.append(
                    np.full(num_filler_i, fill_value=filler_size_y_i, dtype=self.node_size_y.dtype)
                )
                filler_node_area_i = num_filler_i * (filler_size_x_i * filler_size_y_i)
                self.total_filler_node_area += filler_node_area_i
                content += "Region: %2d filler_node_area = %10.2f, #fillers = %8d, filler sizes = %2.4gx%g\n" % (
                    i,
                    filler_node_area_i,
                    num_filler_i,
                    filler_size_x_i,
                    filler_size_y_i,
                )

            self.total_movable_node_area_fence_region = np.array(self.total_movable_node_area_fence_region)
            self.num_movable_nodes_fence_region = np.array(self.num_movable_nodes_fence_region)


        if params.enable_fillers:
            # the way to compute this is still tricky; we need to consider place_io together on how to
            # summarize the area of fixed cells, which may overlap with each other.
            if len(self.regions) > 0:
                self.filler_start_map = np.cumsum([0] + self.num_filler_nodes_fence_region)
                self.num_filler_nodes_fence_region = np.array(self.num_filler_nodes_fence_region)
                self.node_size_x = np.concatenate([self.node_size_x] + filler_node_size_x_list)
                self.node_size_y = np.concatenate([self.node_size_y] + filler_node_size_y_list)
                content += "total_filler_node_area = %10.2f, #fillers = %8d, average filler sizes = %2.4gx%g\n" % (
                    self.total_filler_node_area,
                    self.num_filler_nodes,
                    self.total_filler_node_area / self.num_filler_nodes / self.row_height,
                    self.row_height,
                )
            else:
                node_size_order = np.argsort(self.node_size_x[: self.num_movable_nodes])
                range_lb = int(self.num_movable_nodes*0.05)
                range_ub = int(self.num_movable_nodes*0.95)
                if range_lb >= range_ub: # when there are too few cells, i.e., <= 1
                    filler_size_x = 0
                else:
                    filler_size_x = np.mean(self.node_size_x[node_size_order[range_lb:range_ub]])
                filler_size_y = self.row_height
                placeable_area = max(self.area - self.total_fixed_node_area, self.total_space_area)
                content += "use placeable_area = %g to compute fillers\n" % (placeable_area)
                self.total_filler_node_area = max(
                    placeable_area * params.target_density - self.total_movable_node_area, 0.0
                )
                filler_area = filler_size_x * filler_size_y
                if filler_area == 0: 
                    self.num_filler_nodes = 0
                else:
                    self.num_filler_nodes = int(round(self.total_filler_node_area / filler_area))
                    self.node_size_x = np.concatenate(
                        [
                            self.node_size_x,
                            np.full(self.num_filler_nodes, fill_value=filler_size_x, dtype=self.node_size_x.dtype),
                        ]
                    )
                    self.node_size_y = np.concatenate(
                        [
                            self.node_size_y,
                            np.full(self.num_filler_nodes, fill_value=filler_size_y, dtype=self.node_size_y.dtype),
                        ]
                    )
                content += "total_filler_node_area = %g, #fillers = %d, filler sizes = %gx%g\n" % (
                    self.total_filler_node_area,
                    self.num_filler_nodes,
                    filler_size_x,
                    filler_size_y,
                )
        else:
            self.total_filler_node_area = 0
            self.num_filler_nodes = 0
            filler_size_x, filler_size_y = 0, 0
            if len(self.regions) > 0:
                self.filler_start_map = np.zeros(len(self.regions) + 2, dtype=np.int32)
                self.num_filler_nodes_fence_region = np.zeros(len(self.num_filler_nodes_fence_region))

            content += "total_filler_node_area = %g, #fillers = %d, filler sizes = %gx%g\n" % (
                self.total_filler_node_area,
                self.num_filler_nodes,
                filler_size_x,
                filler_size_y,
            )

        # set number of bins 
        # derive bin dimensions by keeping the aspect ratio 
        self.initialize_num_bins(params)
        # set bin size 
        self.bin_size_x = (self.xh - self.xl) / self.num_bins_x
        self.bin_size_y = (self.yh - self.yl) / self.num_bins_y

        content += "num_bins = %dx%d, bin sizes = %gx%g\n" % (self.num_bins_x, self.num_bins_y, self.bin_size_x / self.row_height, self.bin_size_y / self.row_height)

        if params.routability_opt_flag:
            content += "================================== routing information =================================\n"
            content += "routing grids (%d, %d)\n" % (self.num_routing_grids_x, self.num_routing_grids_y)
            content += "routing grid sizes (%g, %g)\n" % (self.routing_grid_size_x, self.routing_grid_size_y)
            content += "routing capacity H/V (%g, %g) per tile\n" % (self.unit_horizontal_capacity * self.routing_grid_size_y, self.unit_vertical_capacity * self.routing_grid_size_x)
        content += "========================================================================================"

        logging.info(content)

    def write(self, params, filename, sol_file_format=None):
        """
        @brief write placement solution
        @param filename output file name
        @param sol_file_format solution file format, DEF|DEFSIMPLE|BOOKSHELF|BOOKSHELFALL
        """
        tt = time.time()
        logging.info("writing to %s" % (filename))
        if sol_file_format is None:
            if filename.endswith(".def"):
                sol_file_format = place_io.SolutionFileFormat.DEF
            else:
                sol_file_format = place_io.SolutionFileFormat.BOOKSHELF

        # unscale locations
        node_x, node_y = self.unscale_pl(params.shift_factor, params.scale_factor)

        # Global placement may have floating point positions.
        # Currently only support BOOKSHELF format.
        # This is mainly for debug.
        if not params.legalize_flag and not params.detailed_place_flag and sol_file_format == place_io.SolutionFileFormat.BOOKSHELF:
            self.write_pl(params, filename, node_x, node_y)
        else:
            place_io.PlaceIOFunction.write(self.rawdb, filename, sol_file_format, node_x, node_y)
        logging.info("write %s takes %.3f seconds" % (str(sol_file_format), time.time()-tt))

    def read_pl(self, params, pl_file):
        """
        @brief read .pl file
        @param pl_file .pl file
        """
        tt = time.time()
        logging.info("reading %s" % (pl_file))
        count = 0
        with open(pl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("UCLA"):
                    continue
                # node positions
                pos = re.search(r"(\w+)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*:\s*(\w+)", line)
                if pos:
                    node_id = self.node_name2id_map[pos.group(1)]
                    self.node_x[node_id] = float(pos.group(2))
                    self.node_y[node_id] = float(pos.group(6))
                    self.node_orient[node_id] = pos.group(10)
                    orient = pos.group(4)
        if params.shift_factor[0] != 0 or params.shift_factor[1] != 0 or params.scale_factor != 1.0:
            self.scale_pl(params.shift_factor, params.scale_factor)
        logging.info("read_pl takes %.3f seconds" % (time.time()-tt))

    def write_pl(self, params, pl_file, node_x, node_y):
        """
        @brief write .pl file
        @param pl_file .pl file
        """
        tt = time.time()
        logging.info("writing to %s" % (pl_file))
        content = "UCLA pl 1.0\n"
        str_node_names = np.array(self.node_names).astype(np.str)
        str_node_orient = np.array(self.node_orient).astype(np.str)
        for i in range(self.num_movable_nodes):
            content += "\n%s %g %g : %s" % (
                    str_node_names[i],
                    node_x[i],
                    node_y[i],
                    str_node_orient[i]
                    )
        # use the original fixed cells, because they are expanded if they contain shapes
        fixed_node_indices = list(self.rawdb.fixedNodeIndices())
        for i, node_id in enumerate(fixed_node_indices):
            content += "\n%s %g %g : %s /FIXED" % (
                    str(self.rawdb.nodeName(node_id)),
                    float(self.rawdb.node(node_id).xl()),
                    float(self.rawdb.node(node_id).yl()),
                    "N" # still hard-coded
                    )
        for i in range(self.num_movable_nodes + self.num_terminals, self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs):
            content += "\n%s %g %g : %s /FIXED_NI" % (
                    str_node_names[i],
                    node_x[i],
                    node_y[i],
                    str_node_orient[i]
                    )
        with open(pl_file, "w") as f:
            f.write(content)
        logging.info("write_pl takes %.3f seconds" % (time.time()-tt))

    def write_nets(self, params, net_file):
        """
        @brief write .net file
        @param params parameters
        @param net_file .net file
        """
        tt = time.time()
        logging.info("writing to %s" % (net_file))
        content = "UCLA nets 1.0\n"
        content += "\nNumNets : %d" % (len(self.net2pin_map))
        content += "\nNumPins : %d" % (len(self.pin2net_map))
        content += "\n"

        for net_id in range(len(self.net2pin_map)):
            pins = self.net2pin_map[net_id]
            content += "\nNetDegree : %d %s" % (len(pins), self.net_names[net_id].decode())
            for pin_id in pins:
                content += "\n\t%s %s : %d %d" % (self.node_names[self.pin2node_map[pin_id]].decode(), self.pin_direct[pin_id].decode(), self.pin_offset_x[pin_id]/params.scale_factor, self.pin_offset_y[pin_id]/params.scale_factor)

        with open(net_file, "w") as f:
            f.write(content)
        logging.info("write_nets takes %.3f seconds" % (time.time()-tt))

    def apply(self, params, node_x, node_y):
        """
        @brief apply placement solution and update database
        """
        # assign solution
        self.node_x[:self.num_movable_nodes] = node_x[:self.num_movable_nodes]
        self.node_y[:self.num_movable_nodes] = node_y[:self.num_movable_nodes]

        # unscale locations
        node_x, node_y = self.unscale_pl(params.shift_factor, params.scale_factor)

        # update raw database
        place_io.PlaceIOFunction.apply(self.rawdb, node_x, node_y)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")

    params = Params.Params()
    params.load(sys.argv[sys.argv[1]])
    logging.info("parameters = %s" % (params))

    db = PlaceDB()
    db(params)

    db.print_node(1)
    db.print_net(1)
    db.print_row(1)