import os 
import sys 
import torch 
from torch.autograd import Function
#DBG
import pdb

import dreamplace.ops.draw_place.draw_place_cpp as draw_place_cpp
import dreamplace.ops.draw_place.PlaceDrawer as PlaceDrawer 

class DrawPlaceFunction(Function):
    @staticmethod
    def forward(
            pos, 
            node_size_x, node_size_y, 
            pin_offset_x, pin_offset_y, 
            pin2node_map, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            bin_size_x, bin_size_y, 
            num_movable_nodes, num_filler_nodes, 
            filename,
            watermark_cells=None
            ):

        if watermark_cells == None: #Use empty file
            watermark_cells = torch.zeros(int(pos.numel()/2), dtype=torch.int, device=pos.device)

        ret = draw_place_cpp.forward(
                pos, 
                node_size_x, node_size_y, 
                pin_offset_x, pin_offset_y, 
                pin2node_map, 
                xl, yl, xh, yh, 
                site_width, row_height, 
                bin_size_x, bin_size_y, 
                num_movable_nodes, num_filler_nodes,
                watermark_cells,
                filename
                )
        # if C/C++ API failed, try with python implementation 
        if not filename.endswith(".gds") and not ret:
            ret = PlaceDrawer.PlaceDrawer.forward(
                    pos, 
                    node_size_x, node_size_y, 
                    pin_offset_x, pin_offset_y, 
                    pin2node_map, 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    bin_size_x, bin_size_y, 
                    num_movable_nodes, num_filler_nodes, filename,
                    watermark_cells
                    )
        return ret 

class DrawPlace(object):
    """ 
    @brief Draw placement
    """
    def __init__(self, placedb):
        """
        @brief initialization 
        """
        self.node_size_x = torch.from_numpy(placedb.node_size_x)
        self.node_size_y = torch.from_numpy(placedb.node_size_y)
        self.pin_offset_x = torch.from_numpy(placedb.pin_offset_x)
        self.pin_offset_y = torch.from_numpy(placedb.pin_offset_y)
        self.pin2node_map = torch.from_numpy(placedb.pin2node_map)
        self.xl = placedb.xl 
        self.yl = placedb.yl 
        self.xh = placedb.xh 
        self.yh = placedb.yh 
        self.site_width = placedb.site_width
        self.row_height = placedb.row_height 
        self.bin_size_x = placedb.bin_size_x 
        self.bin_size_y = placedb.bin_size_y
        self.num_movable_nodes = placedb.num_movable_nodes
        self.num_filler_nodes = placedb.num_filler_nodes

    def forward(self, pos, filename, watermark_cells=None): 
        """ 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        return DrawPlaceFunction.forward(
                pos, 
                self.node_size_x, 
                self.node_size_y, 
                self.pin_offset_x, 
                self.pin_offset_y, 
                self.pin2node_map, 
                self.xl, 
                self.yl, 
                self.xh, 
                self.yh, 
                self.site_width, 
                self.row_height, 
                self.bin_size_x, 
                self.bin_size_y, 
                self.num_movable_nodes, 
                self.num_filler_nodes, 
                filename,
                watermark_cells
                )

    def __call__(self, pos, filename, watermark_cells=None): 
        """
        @brief top API 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        return self.forward(pos, filename, watermark_cells)