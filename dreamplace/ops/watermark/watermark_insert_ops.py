import torch
import random
import math
import numpy as np
import logging
import time
logger = logging.getLogger(__name__)

class WM_insert(object):
    '''
    @brief Watermark insertion
    '''
    def __init__(self, pos, params, placedb, data_collections, device):
        '''
        @brief initialization
        '''
        self.pos = pos
        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.device = device
        random.seed(self.params.seed)
        if self.params.watermark_type == "combine":
            self.wm_cell_num = self.params.watermark_num
        else:
            self.wm_cell_num = self.params.watermark_num
        self.keys = [random.randint(0, 1) for _ in range(self.wm_cell_num)]
        self.total_cell_num = self.placedb.num_movable_nodes + self.placedb.num_terminals + self.placedb.num_terminal_NIs
        self.node_size_x = self.data_collections.node_size_x[:self.total_cell_num]
        self.node_size_y = self.data_collections.node_size_y[:self.total_cell_num]
        
        print("self.node_size_y", self.node_size_y[self.placedb.num_movable_nodes:self.total_cell_num])
    def get_nearest_dist(self, x_position, y_position):
        #group cells by y_position
        cell_group = {}
        # group cells with the same y position into dict
        x_position_size = x_position + self.node_size_x

        x_position = x_position.detach().cpu().numpy()
        x_position_size = x_position_size.detach().cpu().numpy()
        macro_pos_y = y_position[self.placedb.num_movable_nodes:self.total_cell_num]
        y_position = y_position.detach().cpu().numpy()

        min_y = int(min(y_position[:self.placedb.num_movable_nodes]))
        max_y = int(max(y_position[:self.placedb.num_movable_nodes]))

        macro_pos_y_upper = macro_pos_y + self.node_size_y[self.placedb.num_movable_nodes:self.total_cell_num]
        macro_pos_y_upper = macro_pos_y_upper.detach().cpu().numpy()
        macro_pos_y = macro_pos_y.detach().cpu().numpy()

        print("min_y", min_y)
        print("max_y", max_y)
       
        for i in range(min_y, max_y, int(self.placedb.row_height)):
            idx_list = np.where(y_position == i)[0]
            macro_idx = np.where((macro_pos_y - self.placedb.row_height <= i) & (macro_pos_y_upper + self.placedb.row_height >=i ))[0]
            idx_list = np.concatenate((idx_list, macro_idx+self.placedb.num_movable_nodes), axis=0)
 
            if len(idx_list) == 0:
                continue
            else:
                cell_group[i] = [] #[[idx, x_position[idx], x_position_size[idx]] for idx in idx_list]
                for idx in idx_list:
                    if idx > self.placedb.num_movable_nodes:
                        height_num = 1
                    else:
                        height_num = self.node_size_y[idx]/self.placedb.row_height
                    if height_num ==1:
                        cell_group[i].append([idx, x_position[idx], x_position_size[idx]])
                        
                    else:
                        for k in range(int(height_num)):
                            if i + k*self.placedb.row_height not in list(cell_group.keys()):
                                cell_group[i+k*self.placedb.row_height] = []
                            cell_group[i+k*self.placedb.row_height].append([idx, x_position[idx], x_position_size[idx]])
                 
        # add macros into cell_group
        #if self.placedb.num_terminals > 0:
        #    for i in range(len(self.macro_pos_y)):
        #        cur_height = self.macro_pos_y[i]
        #        if cur_height in cell_group:
        #            height_num = self.macro_pos_y[i]/self.placedb.row_height
        #            for k in range(int(height_num)):
        #                cell_group[cur_height-k*self.placedb.row_height].append([i+self.placedb.num_movable_nodes, self.macro_pos_x[i], self.macro_pos_x[i]+self.macro_size_x[i]])

        # get max x position difference for each y position
        position_diff_ori_x_pos = torch.zeros(len(x_position))
        position_diff_ori_x_neg = torch.zeros(len(x_position))
        position_diff_ori_y_pos = torch.zeros(len(y_position))
        position_diff_ori_y_neg = torch.zeros(len(y_position))
        # sort cell group by y position
        cell_group = dict(sorted(cell_group.items(), key=lambda item: item[0]))
        cell_group_room = {}
        for key in cell_group:

            cur_info = cell_group[key]
            cur_idex, cur_position_list = [i[0] for i in cur_info], [[i[1], i[2]] for i in cur_info]
            # sort cur_position_list by x position
            cur_position_list = sorted(cur_position_list, key=lambda item: item[0])

            # get complement cur_position_list
            cur_position_list_complement = []
            for i in range(len(cur_position_list)-1):
                if cur_position_list[i+1][0] > cur_position_list[i][1]:
                    cur_position_list_complement.append([cur_position_list[i][1], cur_position_list[i+1][0]])
            if len(cur_position_list_complement) != 0:
                cell_group_room[key] = cur_position_list_complement

        #print("cell_group_room", cell_group_room)

        for key in cell_group:

            if key in list(cell_group_room.keys()):
                cur_position_list_complement = cell_group_room[key]
            else:
                continue
            cur_info = cell_group[key]
            cur_idex = [i[0] for i in cur_info]
           
            cur_position_list_complement1 = [cur_position_list_complement[i][0] for i in range(len(cur_position_list_complement))]
            cur_position_list_complement2 = [cur_position_list_complement[i][1] for i in range(len(cur_position_list_complement))]
            
            if key+self.placedb.row_height in list(cell_group_room.keys()):
                upper_complement = cell_group_room[key+self.placedb.row_height]
                upper_complement1 = [upper_complement[i][0] for i in range(len(upper_complement))]
                upper_complement2 = [upper_complement[i][1] for i in range(len(upper_complement))]
                
            else:
                upper_complement = None
            if key-self.placedb.row_height in list(cell_group_room.keys()):
                lower_complement = cell_group_room[key-self.placedb.row_height]
                lower_complement1 = [lower_complement[i][0] for i in range(len(lower_complement))]
                lower_complement2 = [lower_complement[i][1] for i in range(len(lower_complement))]
               
            else:
                lower_complement = None
            for idx in cur_idex:
                if idx >= self.placedb.num_movable_nodes:
                    continue
                cur_list_pos = x_position[idx]
                cur_list_pos_right = x_position_size[idx]
                cur_height = self.node_size_y[idx]/self.placedb.row_height

                #print("cur_list_pos", cur_list_pos, cur_list_pos_right)
                
                if cur_list_pos_right in cur_position_list_complement1:
                    idx_pos = np.where(np.array(cur_position_list_complement1) == cur_list_pos_right)[0][0]
                    if cur_height== 1:
                        cur_pos = (cur_position_list_complement[idx_pos][1] - cur_list_pos_right).item()
                        position_diff_ori_x_pos[idx] = cur_pos
                    else:
                        cur_pos = (cur_position_list_complement[idx_pos][1] - cur_list_pos_right).item()
                        for k in range(int(cur_height)):
                            if key+k*self.placedb.row_height in list(cell_group_room.keys()):
                                prev_complement = cell_group_room[key+k*self.placedb.row_height]
                                prev_complement1 = [prev_complement[i][0] for i in range(len(prev_complement))]
                                prev_complement2 = [prev_complement[i][1] for i in range(len(prev_complement))]
                                mask1 = cur_list_pos_right < prev_complement2
                                mask2 = cur_list_pos_right > prev_complement1
                                mask = np.where(mask1 *mask2)[0]
                                if len(mask) > 0:
                                    mask = mask[0]
                                    cur_pos = min(cur_pos, (prev_complement2[mask] - cur_list_pos_right).item())
                                else:
                                    cur_pos = 0
                            else:
                                cur_pos = 0
                        cur_pos = cur_pos
                        position_diff_ori_x_pos[idx] = cur_pos

                    #print("position_diff_ori_x_pos", idx, cur_position_list_complement[idx_pos][1], cur_position_list_complement[idx_pos][0])
                if cur_list_pos in cur_position_list_complement2:
                    idx_pos = np.where(np.array(cur_position_list_complement2) == cur_list_pos)[0][0]
                    if cur_height  == 1:
                        cur_pos = (cur_list_pos - cur_position_list_complement[idx_pos][0]).item()
                        position_diff_ori_x_neg[idx] = cur_pos
                    else:
                        cur_pos = (cur_list_pos - cur_position_list_complement[idx_pos][0]).item()
                        for k in range(int(cur_height)):
                            if key+k*self.placedb.row_height in list(cell_group_room.keys()):
                                prev_complement = cell_group_room[key+k*self.placedb.row_height]
                                prev_complement1 = [prev_complement[i][0] for i in range(len(prev_complement))]
                                prev_complement2 = [prev_complement[i][1] for i in range(len(prev_complement))]
                                mask1 = cur_list_pos_right < prev_complement2
                                mask2 = cur_list_pos_right > prev_complement1
                                mask = np.where(mask1 *mask2)[0]
                                if len(mask) > 0:
                                    mask = mask[0]
                                    cur_pos = min(cur_pos, (cur_list_pos - prev_complement1[mask]).item())
                                else:
                                    cur_pos = 0
                            else:
                                cur_pos = 0
                        cur_pos = cur_pos
                        position_diff_ori_x_neg[idx] = cur_pos
                    #print("position_diff_ori_x_neg", idx, cur_position_list_complement[idx_pos][1], cur_position_list_complement[idx_pos][0])

                if upper_complement is not None:
                    mask1 = cur_list_pos_right < upper_complement2 
                    mask2 = cur_list_pos > upper_complement1
                    mask = mask1 * mask2
                    if True in mask:
                        position_diff_ori_y_pos[idx] = self.placedb.row_height
                        #print("position_diff_ori_y_pos", idx, upper_complement, cur_list_pos, cur_list_pos_right, mask)
                
                if lower_complement is not None:
                    mask1 =  cur_list_pos_right < lower_complement2 
                    mask2 =  cur_list_pos > lower_complement1
                    mask = mask1 * mask2
                    if True in mask:
                        position_diff_ori_y_neg[idx] = self.placedb.row_height
                        #print("position_diff_ori_y_neg", idx, position_diff_ori_y_neg[idx])

        print(position_diff_ori_x_pos)
        print(position_diff_ori_x_neg)
        print(position_diff_ori_y_pos)
        print(position_diff_ori_y_neg)
        return position_diff_ori_x_pos, position_diff_ori_x_neg, position_diff_ori_y_pos, position_diff_ori_y_neg  
    
    def random_watermark_constraint(self):
        wm_pos = torch.zeros_like(self.pos)
        wm_pos.data.copy_(self.pos.data)

        print("self.placedb.num_terminals", self.placedb.num_movable_nodes, self.placedb.num_terminals, self.placedb.num_terminal_NIs)
        x_position = self.pos[:self.total_cell_num]
        y_position = self.pos[self.placedb.num_nodes:self.placedb.num_nodes+self.total_cell_num]
        st_time = time.time()
        x_pos, x_neg, y_pos, y_neg = self.get_nearest_dist(x_position, y_position)
        et_time = time.time()
        
        x_candidate = x_pos + x_neg
        x_candidate = torch.where(x_candidate != 0)[0]
        x_candidate = x_candidate[torch.randperm(x_candidate.size(0))]
        
        y_candidate = y_pos + y_neg
        y_candidate = torch.where(y_candidate != 0)[0]
        y_candidate = y_candidate[torch.randperm(y_candidate.size(0))]

        print("x_candidate", x_candidate, len(x_candidate))
        print("y_candidate", y_candidate, len(y_candidate))

        wm_cell_idx = torch.zeros(self.wm_cell_num, dtype=torch.long)

        x_start = 0
        y_start = 0
        for i in range(self.wm_cell_num):
            cur_key = self.keys[i]
            if cur_key == 0:
                if x_start >= x_candidate.size(0):
                    break

                if x_candidate[x_start] in wm_cell_idx:
                    x_start += 1
                    continue

                wm_cell_idx[i] = x_candidate[x_start]
                cur_idx = x_candidate[x_start]

                if x_pos[cur_idx] != 0:
                    if self.params.watermark_step < x_pos[cur_idx] and wm_pos[cur_idx] + self.params.watermark_step < self.placedb.site_width:
                        wm_pos[cur_idx] = wm_pos[cur_idx] + self.params.watermark_step
                        x_start += 1
                    else:
                        x_start += 1
                        continue
                elif x_neg[cur_idx] != 0:
                    if self.params.watermark_step < x_neg[cur_idx] and wm_pos[cur_idx] - self.params.watermark_step > 0:
                        wm_pos[cur_idx] = wm_pos[cur_idx] - self.params.watermark_step
                        x_start += 1
                    else:
                        x_start += 1
                        continue
            
            elif cur_key == 1:
                if y_start >= y_candidate.size(0):
                    break
                if y_candidate[y_start] in wm_cell_idx:
                    y_start += 1
                    continue

                wm_cell_idx[i] = y_candidate[y_start]
                cur_idx = y_candidate[y_start]

                if y_pos[cur_idx] != 0:
                    wm_pos[self.placedb.num_nodes+cur_idx] = wm_pos[self.placedb.num_nodes+cur_idx] + self.placedb.row_height
                elif y_neg[cur_idx] != 0:
                    wm_pos[self.placedb.num_nodes+cur_idx] = wm_pos[self.placedb.num_nodes+cur_idx] - self.placedb.row_height
    
                y_start += 1
            
            else:
                raise NotImplementedError
        

        print("current keys", self.keys)
        print("wm_cell_idx", wm_cell_idx, len(wm_cell_idx))

        print("node_orient", self.placedb.node_orient, len(self.placedb.node_orient))
        map_dict = {}
        for i in range(len(wm_cell_idx)):
            map_dict[wm_cell_idx[i].item()] = self.keys[i]
        print("map_dict", map_dict, len(map_dict))

        #assert len(map_dict) == len(wm_cell_idx)
        return wm_pos, wm_cell_idx, et_time-st_time
    
    def get_nearest_dist_fence_region(self, x_position_region, y_position_region, x_position, y_position, wm_cells):
        #group cells by y_position
        cell_group = {}
        # group cells with the same y position into dict
        x_start, x_end = int(self.params.wm_start[0][0]), int(self.params.wm_start[0][0]+self.params.fence_region_size[0])
        x_position_size = x_position_region + self.node_size_x[wm_cells]
        x_position_size_all = x_position + self.node_size_x

        x_position = x_position.detach().cpu().numpy()
        x_position_region = x_position_region.detach().cpu().numpy()
        x_position_size = x_position_size.detach().cpu().numpy()
        x_position_size_all = x_position_size_all.detach().cpu().numpy()

        macro_pos_y = y_position[self.placedb.num_movable_nodes:self.total_cell_num]
        y_position = y_position.detach().cpu().numpy()
        y_position_region = y_position_region.detach().cpu().numpy()


        # get nearest number in y_position with self.params.wm_start[0][1]
        num = np.abs(y_position[:self.placedb.num_movable_nodes] - self.params.wm_start[0][1])
        num = np.argmin(num)
        min_y = int(y_position[num])
        max_y = int(min_y+self.params.fence_region_size[0])

        macro_pos_y_upper = macro_pos_y + self.node_size_y[self.placedb.num_movable_nodes:self.total_cell_num]
        macro_pos_y_upper = macro_pos_y_upper.detach().cpu().numpy()
        macro_pos_y = macro_pos_y.detach().cpu().numpy()

        print("min_y", min_y)
        print("max_y", max_y)

        print("x", x_start, x_end)
        print("wm_cells", wm_cells)
        print("y_position_region", y_position_region)
    
        for i in range(min_y, max_y, int(self.placedb.row_height)):
            idx_list = np.where(y_position_region == i)[0]
            #print("idx_list1", idx_list)
            #macro_idx = np.where((macro_pos_y - self.placedb.row_height <= i) & (macro_pos_y_upper + self.placedb.row_height >=i ))[0]
            #macro_idx = macro_idx + self.placedb.num_movable_nodes
            #idx_list = np.concatenate((idx_list, macro_idx), axis=0)
            #print("idx_list2", idx_list)
            if len(idx_list) == 0:
                cell_group[i] = []
                continue
            else:
                cell_group[i] = [] #[[idx, x_position[idx], x_position_size[idx]] for idx in idx_list]
                for idx in idx_list:
                    if idx > self.placedb.num_movable_nodes:
                        height_num = 1
                    else:
                        height_num = self.node_size_y[idx]/self.placedb.row_height
                    if height_num ==1:
                        #if idx in macro_idx:
                        #    cell_group[i].append([idx, x_position[idx], x_position_size_all[idx]])
                        #else:
                        cell_group[i].append([idx, x_position_region[idx], x_position_size[idx]])
                        
                    else:
                        for k in range(int(height_num)):
                            if i + k*self.placedb.row_height not in list(cell_group.keys()):
                                cell_group[i+k*self.placedb.row_height] = []
                            
                            #if idx in macro_idx:
                            #    cell_group[i+k*self.placedb.row_height].append([idx, x_position[idx], x_position_size_all[idx]])
                            #else:   
                            cell_group[i+k*self.placedb.row_height].append([idx, x_position_region[idx], x_position_size[idx]])
        
        print("cell_group", cell_group)
        # get max x position difference for each y position
        position_diff_ori_x_pos = torch.zeros(len(x_position_region))
        position_diff_ori_x_neg = torch.zeros(len(x_position_region))
        position_diff_ori_y_pos = torch.zeros(len(y_position_region))
        position_diff_ori_y_neg = torch.zeros(len(y_position_region))
        # sort cell group by y position
        cell_group = dict(sorted(cell_group.items(), key=lambda item: item[0]))
        cell_group_room = {}
        for key in cell_group:

            cur_info = cell_group[key]

            if len(cur_info) == 0:
                cur_position_list_complement = []
                cur_position_list_complement.append([x_start, x_end])
                cell_group_room[key] = cur_position_list_complement
                continue
            cur_idex, cur_position_list = [i[0] for i in cur_info], [[i[1], i[2]] for i in cur_info]
            # sort cur_position_list by x position
            cur_position_list = sorted(cur_position_list, key=lambda item: item[0])

            # get complement cur_position_list
            cur_position_list_complement = []
            for i in range(len(cur_position_list)-1):
                if i == 0:
                    if cur_position_list[i][0] > x_start:
                        cur_position_list_complement.append([x_start, cur_position_list[i][0]])
                if cur_position_list[i+1][0] > cur_position_list[i][1]:
                    cur_position_list_complement.append([cur_position_list[i][1], cur_position_list[i+1][0]])
                if i == len(cur_position_list)-2:
                    if cur_position_list[i+1][1] < x_end:
                        cur_position_list_complement.append([cur_position_list[i+1][1], x_end])
            if len(cur_position_list_complement) != 0:
                cell_group_room[key] = cur_position_list_complement

        print("cell_group_room", cell_group_room)

        for key in cell_group:

            if key in list(cell_group_room.keys()):
                cur_position_list_complement = cell_group_room[key]
            else:
                continue
            cur_info = cell_group[key]
            cur_idex = [i[0] for i in cur_info]
           
            cur_position_list_complement1 = [cur_position_list_complement[i][0] for i in range(len(cur_position_list_complement))]
            cur_position_list_complement2 = [cur_position_list_complement[i][1] for i in range(len(cur_position_list_complement))]
            
            if key+self.placedb.row_height in list(cell_group_room.keys()):
                upper_complement = cell_group_room[key+self.placedb.row_height]
                upper_complement1 = [upper_complement[i][0] for i in range(len(upper_complement))]
                upper_complement2 = [upper_complement[i][1] for i in range(len(upper_complement))]
                
            else:
                upper_complement = None
            if key-self.placedb.row_height in list(cell_group_room.keys()):
                lower_complement = cell_group_room[key-self.placedb.row_height]
                lower_complement1 = [lower_complement[i][0] for i in range(len(lower_complement))]
                lower_complement2 = [lower_complement[i][1] for i in range(len(lower_complement))]
               
            else:
                lower_complement = None
            for idx in cur_idex:
                if idx >= self.placedb.num_movable_nodes:
                    continue
                cur_list_pos = x_position_region[idx]
                cur_list_pos_right = x_position_size[idx]
                cur_height = self.node_size_y[idx]/self.placedb.row_height

                #print("cur_list_pos", cur_list_pos, cur_list_pos_right)
                
                if cur_list_pos_right in cur_position_list_complement1:
                    idx_pos = np.where(np.array(cur_position_list_complement1) == cur_list_pos_right)[0][0]
                    if cur_height== 1:
                        cur_pos = (cur_position_list_complement[idx_pos][1] - cur_list_pos_right).item()
                        position_diff_ori_x_pos[idx] = cur_pos
                    else:
                        cur_pos = (cur_position_list_complement[idx_pos][1] - cur_list_pos_right).item()
                        for k in range(int(cur_height)):
                            if key+k*self.placedb.row_height in list(cell_group_room.keys()):
                                prev_complement = cell_group_room[key+k*self.placedb.row_height]
                                prev_complement1 = [prev_complement[i][0] for i in range(len(prev_complement))]
                                prev_complement2 = [prev_complement[i][1] for i in range(len(prev_complement))]
                                mask1 = cur_list_pos_right < prev_complement2
                                mask2 = cur_list_pos_right > prev_complement1
                                mask = np.where(mask1 *mask2)[0]
                                if len(mask) > 0:
                                    mask = mask[0]
                                    cur_pos = min(cur_pos, (prev_complement2[mask] - cur_list_pos_right).item())
                                else:
                                    cur_pos = 0
                            else:
                                cur_pos = 0
                        cur_pos = cur_pos
                        position_diff_ori_x_pos[idx] = cur_pos

                    #print("position_diff_ori_x_pos", idx, cur_position_list_complement[idx_pos][1], cur_position_list_complement[idx_pos][0])
                if cur_list_pos in cur_position_list_complement2:
                    idx_pos = np.where(np.array(cur_position_list_complement2) == cur_list_pos)[0][0]
                    if cur_height  == 1:
                        cur_pos = (cur_list_pos - cur_position_list_complement[idx_pos][0]).item()
                        position_diff_ori_x_neg[idx] = cur_pos
                    else:
                        cur_pos = (cur_list_pos - cur_position_list_complement[idx_pos][0]).item()
                        for k in range(int(cur_height)):
                            if key+k*self.placedb.row_height in list(cell_group_room.keys()):
                                prev_complement = cell_group_room[key+k*self.placedb.row_height]
                                prev_complement1 = [prev_complement[i][0] for i in range(len(prev_complement))]
                                prev_complement2 = [prev_complement[i][1] for i in range(len(prev_complement))]
                                mask1 = cur_list_pos_right < prev_complement2
                                mask2 = cur_list_pos_right > prev_complement1
                                mask = np.where(mask1 *mask2)[0]
                                if len(mask) > 0:
                                    mask = mask[0]
                                    cur_pos = min(cur_pos, (cur_list_pos - prev_complement1[mask]).item())
                                else:
                                    cur_pos = 0
                            else:
                                cur_pos = 0
                        cur_pos = cur_pos
                        position_diff_ori_x_neg[idx] = cur_pos
                    #print("position_diff_ori_x_neg", idx, cur_position_list_complement[idx_pos][1], cur_position_list_complement[idx_pos][0])

                if upper_complement is not None:
                    mask1 = cur_list_pos_right < upper_complement2 
                    mask2 = cur_list_pos > upper_complement1
                    mask = mask1 * mask2
                    if True in mask:
                        position_diff_ori_y_pos[idx] = self.placedb.row_height
                        #print("position_diff_ori_y_pos", idx, upper_complement, cur_list_pos, cur_list_pos_right, mask)
                
                if lower_complement is not None:
                    mask1 =  cur_list_pos_right < lower_complement2 
                    mask2 =  cur_list_pos > lower_complement1
                    mask = mask1 * mask2
                    if True in mask:
                        position_diff_ori_y_neg[idx] = self.placedb.row_height
                        #print("position_diff_ori_y_neg", idx, position_diff_ori_y_neg[idx])

        print(position_diff_ori_x_pos)
        print(position_diff_ori_x_neg)
        print(position_diff_ori_y_pos)
        print(position_diff_ori_y_neg)
        return position_diff_ori_x_pos, position_diff_ori_x_neg, position_diff_ori_y_pos, position_diff_ori_y_neg  
    
    def random_watermark_constraint_combine(self):
        
        wm_cells = self.params.wm_cells[0]
        wm_cells_y = [i+self.placedb.num_nodes for i in wm_cells]
        wm_pos = torch.zeros_like(self.pos)
        wm_pos.data.copy_(self.pos.data)

        x_position_region = self.pos[wm_cells]
        y_position_region = self.pos[wm_cells_y]

        x_position = self.pos[:self.total_cell_num]
        y_position = self.pos[self.placedb.num_nodes:self.placedb.num_nodes+self.total_cell_num]

        st_time = time.time()
        x_pos, x_neg, y_pos, y_neg = self.get_nearest_dist_fence_region(x_position_region, y_position_region, 
                                                                        x_position, y_position, wm_cells)
        et_time = time.time()

        x_candidate = x_pos + x_neg
        x_candidate_idx = torch.where(x_candidate != 0)[0]
        # get overlap of wm_cells and x_candidate
        #import pdb; pdb.set_trace()
        wm_cells = torch.tensor(wm_cells)
        x_candidate = wm_cells[x_candidate_idx]
        #x_candidate_idx = x_candidate_idx.tolist()
        #x_candidate = wm_cells[x_candidate_idx]
        #print("x_candidate2", x_candidate, len(x_candidate))
        # convert x_candidate to tensor
        x_candidate = torch.tensor(x_candidate)
        #print("x_candidate3", x_candidate, len(x_candidate))
        x_candidate = x_candidate[torch.randperm(x_candidate.size(0))]
        
        y_candidate = y_pos + y_neg
        y_candidate_idx = torch.where(y_candidate != 0)[0]
        y_candidate = wm_cells[y_candidate_idx]
        y_candidate = torch.tensor(y_candidate)
        y_candidate = y_candidate[torch.randperm(y_candidate.size(0))]

            
        print("x_candidate", x_candidate, len(x_candidate))
        print("y_candidate", y_candidate, len(y_candidate))

        wm_cell_idx = torch.zeros(self.wm_cell_num, dtype=torch.long)
        region_left = self.params.wm_start[0][0]
        region_right = self.params.wm_start[0][0] +self.params.fence_region_size[0]

        x_start = 0
        y_start = 0
        count = 0
        for _ in range(len(y_candidate)+len(x_candidate)):
            cur_key = self.keys[count]
            if cur_key == 0:
                if x_start >= x_candidate.size(0) and y_start >= y_candidate.size(0):
                    break
                elif x_start >= x_candidate.size(0):
                    self.keys[count] = 1
                    continue

                if x_candidate[x_start] in wm_cell_idx:
                    x_start += 1
                    continue

                cur_idx = x_candidate[x_start]
                
                if x_pos[x_start] != 0:
                    if self.params.watermark_step < x_pos[x_start] and wm_pos[cur_idx] + self.params.watermark_step < region_right:
                        wm_pos[cur_idx] = wm_pos[cur_idx] + self.params.watermark_step
                    else:
                        x_start += 1
                        continue
                elif x_neg[x_start] != 0:
                    if self.params.watermark_step < x_neg[x_start] and wm_pos[cur_idx] - self.params.watermark_step > region_left:
                        wm_pos[cur_idx] = wm_pos[cur_idx] - self.params.watermark_step
                    else:
                        x_start += 1
                        continue

                wm_cell_idx[count] = x_candidate[x_start]
                count = count + 1
                x_start += 1
                if count == self.wm_cell_num:
                    break
            elif cur_key == 1:
                if y_start >= y_candidate.size(0) and x_start >= x_candidate.size(0):
                    break
                elif y_start >= y_candidate.size(0):
                    self.keys[count] = 0
                    continue
                if y_candidate[y_start] in wm_cell_idx:
                    y_start += 1
                    continue
                print("y_candidate[y_start]", y_candidate[y_start], y_start)
                cur_idx = y_candidate[y_start]

                if y_pos[y_start] != 0:
                    wm_pos[self.placedb.num_nodes+cur_idx] = wm_pos[self.placedb.num_nodes+cur_idx] + self.placedb.row_height
                elif y_neg[y_start] != 0:
                    wm_pos[self.placedb.num_nodes+cur_idx] = wm_pos[self.placedb.num_nodes+cur_idx] - self.placedb.row_height
                
                wm_cell_idx[count] = y_candidate[y_start]
                count = count + 1
                y_start += 1
                if count == self.wm_cell_num:
                    break
            
            else:
                raise NotImplementedError
        

        print("current keys", self.keys)
        print("wm_cell_idx", wm_cell_idx, len(wm_cell_idx))

        # remove zero from wm_cell_idx
        wm_cell_idx = wm_cell_idx[torch.where(wm_cell_idx != 0)[0]]

        print("wm_cell_idx", wm_cell_idx)
        map_dict = {}
        for i in range(len(wm_cell_idx)):
            map_dict[wm_cell_idx[i].item()] = self.keys[i]
        print("map_dict", map_dict, len(map_dict))

        assert len(map_dict) == len(wm_cell_idx)
        return wm_pos, wm_cell_idx, et_time-st_time

    def map_watermark_constraint(self):
        wm_pos = torch.zeros_like(self.pos)
        wm_pos.data.copy_(self.pos.data)

        x_position = self.pos[:self.placedb.num_nodes]
        x_position_diff = self.get_nearest_dist(x_position)
        y_position = self.pos[self.placedb.num_nodes:]
        y_position_diff = self.get_nearest_dist(y_position)

        print("total num of nodes", self.placedb.num_nodes)
        print("non-fixed nodes", self.placedb.num_movable_nodes)
        print("non zero x position diff", torch.nonzero(x_position_diff).size(0))
        print("non zero y position diff", torch.nonzero(y_position_diff).size(0))
        
        wm_cell_idx = torch.zeros(self.wm_cell_num, dtype=torch.long)
        gap = self.placedb.num_movable_nodes // self.wm_cell_num
        count = 0
        
        print("gap", gap, self.placedb.num_movable_nodes, self.wm_cell_num)
        for i in range(self.wm_cell_num):
            start_idx = i * gap
            end_idx = (i+1) * gap
            cur_key = self.keys[i]
            idx_range = list(range(start_idx, end_idx))
            random.shuffle(idx_range)
            for cur_idx in idx_range:
                if cur_key == 0 and x_position_diff[cur_idx] != 0:
                    wm_cell_idx[i] = cur_idx
                    wm_pos[cur_idx] = wm_pos[cur_idx] + x_position_diff[cur_idx]
                    break
                elif cur_key == 1 and y_position_diff[cur_idx] != 0:
                    wm_cell_idx[i] = cur_idx
                    if y_position_diff[cur_idx] > 0:
                        wm_pos[self.placedb.num_nodes+cur_idx] = wm_pos[self.placedb.num_nodes+cur_idx] + self.placedb.row_height
                    else:
                        wm_pos[self.placedb.num_nodes+cur_idx] = wm_pos[self.placedb.num_nodes+cur_idx] - self.placedb.row_height
                    break
                else:
                    if x_position_diff[cur_idx] != 0:
                        wm_cell_idx[i] = cur_idx
                        wm_pos[cur_idx] = wm_pos[cur_idx] + x_position_diff[cur_idx]
                        self.keys[i] = 0
                        break
                    elif y_position_diff[cur_idx] != 0:
                        wm_cell_idx[i] = cur_idx
                        if y_position_diff[cur_idx] > 0:
                            wm_pos[self.placedb.num_nodes+cur_idx] = wm_pos[self.placedb.num_nodes+cur_idx] + self.placedb.row_height
                        else:
                            wm_pos[self.placedb.num_nodes+cur_idx] = wm_pos[self.placedb.num_nodes+cur_idx] - self.placedb.row_height
                        self.keys[i] = 1
                        break
        print("wm_cell_idx", wm_cell_idx)
        return wm_pos, wm_cell_idx

    def random_watermark(self):
        wm_pos = torch.zeros_like(self.pos)
        wm_pos.data.copy_(self.pos.data)

        print("placedb.row_height", self.placedb.row_height)

        x_position = self.pos[:self.placedb.num_movable_nodes]
        x_position_diff = self.get_nearest_dist(x_position)
        y_position = self.pos[self.placedb.num_nodes:self.placedb.num_nodes+self.placedb.num_movable_nodes]
        y_position_diff = self.get_nearest_dist(y_position)


        if self.params.watermark_type == "combine":
            wm_cell_idx = np.random.permutation(self.params.wm_cells[0])
        else:
            wm_cell_idx = torch.randperm(self.placedb.num_movable_nodes)

        wm_dist  = torch.zeros_like(wm_cell_idx)
        for i in range(self.wm_cell_num):
            if self.keys[i] == 0:
                random_dist = random.uniform(-self.params.watermark_step, self.params.watermark_step)
                wm_pos[wm_cell_idx[i]] = wm_pos[wm_cell_idx[i]] + random_dist
            else:
                sign = random.uniform(-self.params.watermark_step, self.params.watermark_step)
                if sign > 0:
                    random_dist = self.placedb.row_height
                else:
                    random_dist = -self.placedb.row_height
                wm_pos[self.placedb.num_nodes+wm_cell_idx[i]] = wm_pos[self.placedb.num_nodes+wm_cell_idx[i]] + random_dist
        return wm_pos, wm_cell_idx
    
    def scatter_watermark(self):
        wm_pos = torch.zeros_like(self.pos)
        wm_pos.data.copy_(self.pos.data)

        x_position = self.pos[:self.total_cell_num]
        y_position = self.pos[self.placedb.num_nodes:self.placedb.num_nodes+self.total_cell_num]
        st_time = time.time()
        x_pos, x_neg, y_pos, y_neg = self.get_nearest_dist(x_position, y_position)
        et_time = time.time()
        
        #x_candidate = x_pos + x_neg
        #x_candidate = torch.where(x_candidate != 0)[0]
        #x_candidate = x_candidate[torch.randperm(x_candidate.size(0))]
        
        x_pos = torch.where(x_pos != 0)[0]
        x_pos = x_pos[torch.randperm(x_pos.size(0))]
        x_neg = torch.where(x_neg != 0)[0]
        x_neg = x_neg[torch.randperm(x_neg.size(0))]

       
        wm_cell_idx = torch.zeros(self.wm_cell_num, dtype=torch.long)

        x_start = 0
        y_start = 0
        count1 = 0
        count0 = 0
        for i in range(self.wm_cell_num):
            cur_key = self.keys[i]
            if cur_key == 1:
                cur_idx = x_pos[count1]
                wm_cell_idx[i] = cur_idx
                wm_pos[cur_idx] = wm_pos[cur_idx] + self.params.watermark_step
                count1 += 1
            else:
                cur_idx = x_neg[count0]
                wm_cell_idx[i] = cur_idx
                wm_pos[cur_idx] = wm_pos[cur_idx] - self.params.watermark_step
                count0 += 1

        print("current keys", self.keys)
        print("wm_cell_idx", wm_cell_idx, len(wm_cell_idx))

        map_dict = {}
        for i in range(len(wm_cell_idx)):
            map_dict[wm_cell_idx[i].item()] = self.keys[i]
 
        #assert len(map_dict) == len(wm_cell_idx)
        return wm_pos, wm_cell_idx, et_time-st_time
                
        
    def insert(self):
        if self.params.watermark_method == "random_perturb_constraint" and self.params.wm_start is None:
            logger.info("Watermark with: random perturbation constraint under detailed wm mode")
            self.wm_pos, wm_cell_idx, sub_time = self.random_watermark_constraint()
            return self.wm_pos, wm_cell_idx, self.keys, sub_time
        elif self.params.watermark_method == "random_perturb_constraint" and self.params.wm_start is not None:
            logger.info("Watermark with: random perturbation constraint under combined wm mode")
            self.wm_pos, wm_cell_idx, sub_time = self.random_watermark_constraint_combine()
            return self.wm_pos, wm_cell_idx, self.keys, sub_time
        elif self.params.watermark_method == "random_perturb":
            logger.info("Watermark with: random perturbation")
            self.wm_pos, wm_cell_idx = self.random_watermark()
            return self.wm_pos, wm_cell_idx, self.keys
        elif self.params.watermark_method == "map_perturb_constraint":
            logger.info("Watermark with: map perturbation constraint")
            self.wm_pos, wm_cell_idx = self.map_watermark_constraint()
            return self.wm_pos, wm_cell_idx, self.keys
        elif self.params.watermark_method == "scatter":
            logger.info("Watermark with: scatter")
            self.wm_pos, wm_cell_idx, sub_time = self.scatter_watermark()
            return self.wm_pos, wm_cell_idx, self.keys, sub_time
        else:
            logger.info("No watermark is inserted")
            return self.pos, self.keys