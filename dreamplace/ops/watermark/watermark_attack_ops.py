import torch
import math
import re
import time
import os
import random
import numpy as np
import logging
logger = logging.getLogger(__name__)

class  WM_attack(object):
    '''
    @brief Watermark attack
    '''
    def  __init__(self, pos, params, placedb, data_collections, device, ratio):
        '''
        @brief initialization
        '''
        
        self.pos = pos
        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.device = device
        self.ratio = ratio
        self.x_pos, self.x_neg, self.y_pos, self.y_neg = None, None, None, None
        
        self.total_cell_num = self.placedb.num_movable_nodes + self.placedb.num_terminals + self.placedb.num_terminal_NIs
        self.node_size_x = self.data_collections.node_size_x[:self.total_cell_num]
        self.node_size_y = self.data_collections.node_size_y[:self.total_cell_num]
        random.seed(self.params.attack_seed)
        torch.manual_seed(self.params.attack_seed)
   
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
    
    
    def random_attack_constraint(self):
        # change random seed
        torch.manual_seed(self.params.attack_seed*100)
        random.seed(self.params.attack_seed*100)

        attack_pos = torch.zeros_like(self.pos)
        attack_pos.data.copy_(self.pos.data)
        print("self.placedb.num_terminals", self.placedb.num_movable_nodes, self.placedb.num_terminals, self.placedb.num_terminal_NIs)
        x_position = self.pos[:self.total_cell_num]
        y_position = self.pos[self.placedb.num_nodes:self.placedb.num_nodes+self.total_cell_num]
        if self.x_pos is None:
            x_pos, x_neg, y_pos, y_neg = self.get_nearest_dist(x_position, y_position)
            self.x_pos, self.x_neg, self.y_pos, self.y_neg = x_pos, x_neg, y_pos, y_neg
        else:
            x_pos, x_neg, y_pos, y_neg = self.x_pos, self.x_neg, self.y_pos, self.y_neg
        
        x_candidate = x_pos + x_neg
        x_candidate = torch.where(x_candidate != 0)[0]
        x_candidate = x_candidate[torch.randperm(x_candidate.size(0))]
        
        y_candidate = y_pos + y_neg
        y_candidate = torch.where(y_candidate != 0)[0]
        y_candidate = y_candidate[torch.randperm(y_candidate.size(0))]

        print("x_candidate", x_candidate, len(x_candidate))
        print("y_candidate", y_candidate, len(y_candidate))

        self.attack_num = int(self.placedb.num_movable_nodes*self.ratio)
        #self.attack_num = int(10*self.params.attack_num)
        wm_cell_idx = torch.zeros(self.attack_num , dtype=torch.long)
        self.keys = [random.randint(0, 1) for _ in range(self.attack_num)]

        x_start = 0
        y_start = 0
        for i in range(self.attack_num):
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
                    if self.params.attack_step < x_pos[cur_idx] and attack_pos[cur_idx] + self.params.attack_step < self.placedb.site_width:
                        attack_pos[cur_idx] = attack_pos[cur_idx] + self.params.attack_step
                        x_start += 1
                    else:
                        x_start += 1
                        continue
                elif x_neg[cur_idx] != 0:
                    if self.params.attack_step < x_neg[cur_idx] and attack_pos[cur_idx] - self.params.attack_step > 0:
                        attack_pos[cur_idx] = attack_pos[cur_idx] - self.params.attack_step
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
                    attack_pos[self.placedb.num_nodes+cur_idx] = attack_pos[self.placedb.num_nodes+cur_idx] + self.placedb.row_height
                elif y_neg[cur_idx] != 0:
                    attack_pos[self.placedb.num_nodes+cur_idx] = attack_pos[self.placedb.num_nodes+cur_idx] - self.placedb.row_height
    
                y_start += 1
            
            else:
                raise NotImplementedError
        

        print("attack keys", self.keys)
        print("attack index", wm_cell_idx, len(wm_cell_idx))
        print("moved pos", torch.sum(attack_pos-self.pos))
        map_dict = {}
        for i in range(len(wm_cell_idx)):
            map_dict[wm_cell_idx[i].item()] = self.keys[i]
      
        return attack_pos
    
    def random_attack_row(self):
        attack_pos = torch.zeros_like(self.pos)
        attack_pos.data.copy_(self.pos.data)

        y_position = self.pos[self.placedb.num_nodes:]
        y_position_diff = self.get_nearest_dist(y_position)

        random.seed(self.params.attack_seed)
        self.attack_cell_num = int(self.ratio * self.placedb.num_movable_nodes)
        attack_cell_idx = torch.randperm(self.placedb.num_movable_nodes)

        count = 0
        for i in range(len(attack_cell_idx)):
            cur_cell = attack_cell_idx[i]
            ## watermark y axis
            if y_position_diff[cur_cell] == 0:
                continue
            elif y_position_diff[cur_cell] > 0:
                attack_pos[self.placedb.num_nodes+cur_cell] = attack_pos[self.placedb.num_nodes+cur_cell] + self.placedb.row_height
                count = count + 1
            else:
                attack_pos[self.placedb.num_nodes+cur_cell] = attack_pos[self.placedb.num_nodes+cur_cell] - self.placedb.row_height
                count = count + 1
            if count == self.attack_cell_num:
                break
        print("attack cell num: ", self.ratio, count, self.attack_cell_num) 
        return attack_pos
    
    def swap_pairs_attack(self):
        print("currently using swap pair attack")
        attack_pos = torch.zeros_like(self.pos)
        attack_pos.data.copy_(self.pos.data)

        random.seed(self.params.attack_seed)
        self.attack_cell_num = int(self.ratio * self.placedb.num_movable_nodes)
        #self.attack_cell_num = int(self.ratio)
        self.attack_cell_num = self.attack_cell_num - self.attack_cell_num % 2
        attack_cell_idx = torch.randperm(self.placedb.num_movable_nodes)[:self.attack_cell_num]

        print("self.ratio ", self.ratio )
        for i in range(0, self.attack_cell_num, 2):
            cell_idx1 = attack_cell_idx[i]
            cell_idx2 = attack_cell_idx[i+1]
            cell_idx1x = self.pos[cell_idx1]
            cell_idx1y = self.pos[self.placedb.num_nodes+cell_idx1]
            cell_idx2x = self.pos[cell_idx2]
            cell_idx2y = self.pos[self.placedb.num_nodes+cell_idx2]

            attack_pos[cell_idx1] = cell_idx2x
            attack_pos[self.placedb.num_nodes+cell_idx1] = cell_idx2y
            attack_pos[cell_idx2] = cell_idx1x
            attack_pos[self.placedb.num_nodes+cell_idx2] = cell_idx1y

        return attack_pos
    
    def random_attack(self):
        attack_pos = torch.zeros_like(self.pos)
        attack_pos.data.copy_(self.pos.data)

        random.seed(self.params.attack_seed)
        self.attack_cell_num = int(self.ratio * self.placedb.num_movable_nodes)
        attack_cell_idx = torch.randperm(self.placedb.num_movable_nodes)[:self.attack_cell_num]
        self.keys = [random.randint(0, 1) for _ in range(self.attack_cell_num)]

        count = 0
        for i in range(self.attack_cell_num):
            if self.keys[i] == 0:
                random_dist = random.uniform(-self.params.attack_step, self.params.attack_step)
                attack_pos[attack_cell_idx[i]] = attack_pos[attack_cell_idx[i]] + random_dist
            else:
                random_dist = random.uniform(-self.params.attack_step, self.params.attack_step)
                #attack_pos[self.placedb.num_nodes+attack_cell_idx[i]] = attack_pos[self.placedb.num_nodes+attack_cell_idx[i]] + random_dist
                if random_dist > 0:
                    attack_pos[self.placedb.num_nodes+attack_cell_idx[i]] = attack_pos[self.placedb.num_nodes+attack_cell_idx[i]] + self.placedb.row_height
                else:
                    attack_pos[self.placedb.num_nodes+attack_cell_idx[i]] = attack_pos[self.placedb.num_nodes+attack_cell_idx[i]] - self.placedb.row_height
        print("attack cell num: ", count, self.attack_cell_num) 
        return attack_pos
    

    def get_cell_in_region(self, position, sizex, sizey, x, y, width, height, params):
        '''
        position: axis of cells
        sizex: size of cells in x direction
        '''
        cell_idx = []
        node_num_prev = len(sizex)

        print("position", position.size(), sizex.size(), sizey.size(), self.placedb.num_movable_nodes, node_num_prev)
        x_position_right = position[:self.placedb.num_movable_nodes]+ sizex[:self.placedb.num_movable_nodes]
        x_position_left = position[:self.placedb.num_movable_nodes] #+ sizex[:self.num_movable_nodes]/2

        #print("len(x)", len(sizex), len(sizey))
        #print("self.num_physical_nodes", self.num_physical_nodes, x_position_left.size(), sizex.size())

        y_position_up = position[node_num_prev:node_num_prev+self.placedb.num_movable_nodes] + sizey[:self.placedb.num_movable_nodes]
        y_position_down = position[node_num_prev:node_num_prev+self.placedb.num_movable_nodes] #+ sizey[:self.num_movable_nodes]/2

        #x_inter_idx = np.where((x_position_left > x-width/2) & (x_position_right < (x+width/2)))[0]
        x_inter_idx = np.where((x_position_right < x+width) & ( x_position_left> (x)))[0]
        #y_inter_idx = np.where((y_position_down > y-height/2) & (y_position_up < (y+height/2)))[0]
        y_inter_idx = np.where((y_position_up < y+height) & (y_position_down > (y)))[0]


        # get cell name from cell idx # (7446.), tensor(7020.)
        cell_idx = np.intersect1d(x_inter_idx, y_inter_idx)

        total_region = width * height
        cell_region = torch.sum(sizex[cell_idx] * sizey[cell_idx]).item()
        #print("fraction", cell_region/total_region)
        return cell_idx, cell_region/total_region, len(cell_idx)

    def get_region_score(self, save_pos, sizex, sizey, id_x, id_y, fence_region_size, params):
        # score 1: num of cells within region; score 2: area of cells in region; score 3: 
        # put all tensors to cpu
        self.x_position_left_stand = self.x_position_left_stand.cpu()
        self.x_position_right_stand = self.x_position_right_stand.cpu()
        self.y_position_up_stand = self.y_position_up_stand.cpu()
        self.y_position_down_stand = self.y_position_down_stand.cpu()
        self.x_position_left_macro = self.x_position_left_macro.cpu()
        self.x_position_right_macro = self.x_position_right_macro.cpu()
        self.y_position_up_macro = self.y_position_up_macro.cpu()
        self.y_position_down_macro = self.y_position_down_macro.cpu()
        self.macro_size_x = self.macro_size_x.cpu()
        self.macro_size_y = self.macro_size_y.cpu()
        fence_region_size = fence_region_size.cpu()

        x_inter_idx = np.where((self.x_position_left_stand > id_x) & (self.x_position_right_stand < id_x+fence_region_size))[0] # find who is within region
        y_inter_idx = np.where((self.y_position_down_stand > id_y) & (self.y_position_up_stand < id_y+fence_region_size))[0]
        x_inter_idx_overlap = np.where((self.x_position_right_stand >= id_x) & (self.x_position_left_stand <= id_x+fence_region_size))[0] # find who has overlap with region
        y_inter_idx_overlap = np.where((self.y_position_up_stand >= id_y) & (self.y_position_down_stand <= id_y+fence_region_size))[0]

        x_macro_inter_idx = np.where((self.x_position_left_macro > id_x) & (self.x_position_right_macro < id_x+fence_region_size))[0] # find who is within region
        y_macro_inter_idx = np.where((self.y_position_down_macro > id_y) & (self.y_position_up_macro < id_y+fence_region_size))[0]
        x_macro_inter_overlap = np.where((self.x_position_right_macro>= id_x) & (self.x_position_left_macro <= id_x+fence_region_size))[0] #  find who has overlap with region
        y_macro_inter_overlap = np.where((self.y_position_up_macro >= id_y) & (self.y_position_down_macro <= id_y+fence_region_size))[0]
        
        macro_inter_idx = np.intersect1d(x_macro_inter_idx, y_macro_inter_idx)
        macro_inter_overlap = np.intersect1d(x_macro_inter_overlap, y_macro_inter_overlap)
        macro_inter_overlap = np.setdiff1d(macro_inter_overlap, macro_inter_idx)

        cell_idx = np.intersect1d(x_inter_idx, y_inter_idx)
        #print("cell_idx", len(cell_idx), "cell_idx_overlap", len(cell_idx_overlap), "macro_inter_idx", len(macro_inter_idx))  
        cell_idx_overlap = np.intersect1d(x_inter_idx_overlap, y_inter_idx_overlap)
        # excluse cell_idx in cell_idx_overlap
        cell_idx_overlap = np.setdiff1d(cell_idx_overlap, cell_idx)

        #if len(cell_idx) <=5:
        #    return float('inf'), None
        if len(cell_idx) == 0:
            return float('inf'), None

        score1 = params.watermark_num/len(cell_idx) # smaller, better

        size_within_region = torch.sum(sizex[cell_idx] * sizey[cell_idx]).item()  # standard cells
        macro_size_within_region = torch.sum(self.macro_size_x[macro_inter_idx] * self.macro_size_y[macro_inter_idx]).item() # macro cells

        score2 = (size_within_region+macro_size_within_region)/fence_region_size**2 # smaller, better
        
        
        overlap_region = 0
        for i in cell_idx_overlap:
            # overlap at lower left corner
            if self.x_position_left_stand[i] < id_x and self.y_position_down_stand[i] < id_y:
                overlap_region += (self.x_position_right_stand[i]-id_x)*(self.y_position_up_stand[i]-id_y)
            elif self.x_position_left_stand[i] < id_x and self.y_position_down_stand[i] >= id_y:
                min_y = min(self.y_position_up_stand[i], id_y+fence_region_size)
                overlap_region += (self.x_position_right_stand[i]-id_x)*(min_y-self.y_position_down_stand[i])
            elif self.x_position_left_stand[i] >= id_x and self.y_position_down_stand[i] < id_y:
                min_x = min(self.x_position_right_stand[i], id_x+fence_region_size)
                overlap_region += (min_x-self.x_position_left_stand[i])*(self.y_position_up_stand[i]-id_y)
            elif self.x_position_left_stand[i] >= id_x and self.y_position_down_stand[i] >= id_y:
                min_x = min(self.x_position_right_stand[i], id_x+fence_region_size)
                min_y = min(self.y_position_up_stand[i], id_y+fence_region_size)
                overlap_region += (min_x-self.x_position_left_stand[i])*(min_y-self.y_position_down_stand[i])
        score3 = overlap_region/fence_region_size**2 # smaller, better
        # convert score3 to tensor if it is
        if isinstance(score3, torch.Tensor):
            score3 = score3.item()
        
        #print("current position", id_x, id_y)
        #print("current score", score1, score2, score3) #1:5:10； 50/10 test 7
        score = params.alpha_weight*score1+params.beta_weight*score2+params.gamma_weight*score3
        return score, cell_idx
        #return score1+score2+score3
        #return score3
    
    def check_in_fence_region(self, id_x, id_y, fence_region_size):
        #print("self.x_fence_right", self.x_fence_right, id_x+fence_region_size)
        x_iter_idx = np.where((id_x+fence_region_size > self.x_fence_left) & (id_x < self.x_fence_right))[0]
        y_iter_idx = np.where((id_y+fence_region_size > self.y_fence_low) & (id_y < self.y_fence_up))[0]

        iter_idx = np.intersect1d(x_iter_idx, y_iter_idx)

        if len(iter_idx) > 0:
            return True
        
        y_macro_iter_idx = np.where((id_y+fence_region_size > self.y_position_down_macro) & (id_y < self.y_position_up_macro))[0]
        x_macro_iter_idx = np.where((id_x+fence_region_size > self.x_position_left_macro) & (id_x < self.x_position_right_macro))[0]
        macro_iter_idx = np.intersect1d(x_macro_iter_idx, y_macro_iter_idx)
        if len(macro_iter_idx) > 0:
            return True
        return False

    def score_fence_region(self, save_pos, sizex, sizey, fence_region_size, fence_region_stride, params):
        node_num_prev = len(sizex)
        cell_nums = self.placedb.num_movable_nodes + self.placedb.num_terminals + self.placedb.num_terminal_NIs

        self.x_position_left = save_pos[:cell_nums]
        self.y_position_down = save_pos[node_num_prev : node_num_prev+cell_nums]
        self.x_position_right = save_pos[:cell_nums] + sizex[:cell_nums]
        self.y_position_up = save_pos[node_num_prev : node_num_prev+cell_nums] + sizey[:cell_nums]

        self.x_position_left_stand = self.x_position_left[:self.placedb.num_movable_nodes]
        self.y_position_down_stand = self.y_position_down[:self.placedb.num_movable_nodes]
        self.x_position_right_stand = self.x_position_right[:self.placedb.num_movable_nodes]
        self.y_position_up_stand = self.y_position_up[:self.placedb.num_movable_nodes]

        self.x_position_left_macro = self.x_position_left[self.placedb.num_movable_nodes:cell_nums]
        self.y_position_down_macro = self.y_position_down[self.placedb.num_movable_nodes:cell_nums]
        self.x_position_right_macro = self.x_position_right[self.placedb.num_movable_nodes:cell_nums]
        self.y_position_up_macro = self.y_position_up[self.placedb.num_movable_nodes:cell_nums]

        self.macro_size_x = sizex[self.placedb.num_movable_nodes:self.placedb.num_movable_nodes + self.placedb.num_terminals + self.placedb.num_terminal_NIs]
        self.macro_size_y = sizey[self.placedb.num_movable_nodes:self.placedb.num_movable_nodes + self.placedb.num_terminals + self.placedb.num_terminal_NIs]

        if len(self.placedb.flat_region_boxes) > 0:
            self.x_fence_left = torch.tensor([self.placedb.flat_region_boxes[i][0]/self.placedb.constant  for i in range(len(self.placedb.flat_region_boxes))])
            self.y_fence_low = torch.tensor([self.placedb.flat_region_boxes[i][1]/self.placedb.constant  for i in range(len(self.placedb.flat_region_boxes))])
            self.x_fence_right = torch.tensor([self.placedb.flat_region_boxes[i][2]/self.placedb.constant  for i in range(len(self.placedb.flat_region_boxes))])
            self.y_fence_up = torch.tensor([self.placedb.flat_region_boxes[i][3]/self.placedb.constant  for i in range(len(self.placedb.flat_region_boxes))])
            
            
        #min_x = max(0,  int(torch.min(self.x_position_left).item()))
        min_x = int(torch.min(self.x_position_left).item())
        max_x = int(torch.max(self.x_position_left).item())

        #min_y = max(0, int(torch.min(self.y_position_down).item()))
        min_y = int(torch.min(self.y_position_down).item())
        max_y = int(torch.max(self.y_position_down).item())

        start_idx = []
        cell_idxs = []
        scores = []
        #pdb.set_trace()
        for id_x in range(min_x, max_x-int(fence_region_size), int(fence_region_stride)):
            for id_y in range(min_y, max_y-int(fence_region_size), int(fence_region_stride)):
                ## check if the region is in fence region
                #print("fence region", len(self.flat_region_boxes), id_x, id_y, self.check_in_fence_region(id_x, id_y, fence_region_size))
                #if len(self.placedb.flat_region_boxes) > 0:
                #    if self.check_in_fence_region(id_x, id_y, fence_region_size): 
                #        continue
                #import pdb; pdb.set_trace()
                score, cell_idx = self.get_region_score(save_pos, sizex, sizey, id_x, id_y, fence_region_size, params)
                
                start_idx.append([id_x, id_y])
                cell_idxs.append(cell_idx)
                scores.append(score)
        print("self.ratio ", self.ratio )
        # get minimum score of 5 region
        if self.ratio == 1:
            min_idx = np.argmin(scores)
            assert scores[min_idx] is not float('inf'), "no valid region"
            return cell_idxs[min_idx]
        elif self.ratio == 5:
            min_idx = np.argsort(scores)[:5]
            final_cell = []
            for i in min_idx:
                if scores[i] is not float('inf'):
                    final_cell.extend(cell_idxs[i].tolist())
            return final_cell
    


    def adaptive_region(self):
        attack_pos = torch.zeros_like(self.pos)
        attack_pos.data.copy_(self.pos.data)

        # given attack_pos, self.node_size_x, self.node_size_y, find the region that is rectangular
        if self.params.watermark_type == "row_parity" or self.params.watermark_type == "detail" or self.params.watermark_type == "scatter_wm":
            height = torch.min(self.node_size_y[:self.placedb.num_movable_nodes])
            self.params.fence_region_size[0] = self.params.fence_region_size[0]*height
            self.params.fence_region_stride = self.params.fence_region_stride*height
        if self.params.watermark_type == "gnn":
            height = torch.min(self.node_size_y[:self.placedb.num_movable_nodes])
            if self.total_cell_num > 200000:
                self.params.fence_region_size[0] = 20*10*height
                self.params.fence_region_stride = 10*10*height
            else:
                self.params.fence_region_size[0] = 10*height
                self.params.fence_region_stride = 5*height
        self.node_size_x = self.node_size_x.cpu()
        self.node_size_y = self.node_size_y.cpu()
        cell_index = self.score_fence_region(attack_pos, self.node_size_x, self.node_size_y, self.params.fence_region_size[0], self.params.fence_region_stride, self.params)
        #import pdb; pdb.set_trace()
        self.x_pos, self.x_neg, self.y_pos, self.y_neg = self.get_nearest_dist(attack_pos[:self.total_cell_num], attack_pos[self.placedb.num_nodes:self.placedb.num_nodes+self.total_cell_num])
        #self.x_pos, self.x_neg, self.y_pos, self.y_neg
        for idx in cell_index:
            if self.x_pos[idx] != 0:
                attack_pos[idx] = attack_pos[idx] + self.params.attack_step
                continue
            elif self.x_neg[idx] != 0:
                attack_pos[idx] = attack_pos[idx] - self.params.attack_step
                continue
            elif self.y_pos[idx] != 0:
                attack_pos[self.placedb.num_nodes+idx] = attack_pos[self.placedb.num_nodes+idx] + self.placedb.row_height
                continue
            elif self.y_neg[idx] != 0:
                attack_pos[self.placedb.num_nodes+idx] = attack_pos[self.placedb.num_nodes+idx] - self.placedb.row_height
                continue

        return attack_pos
    
    def read_pl(self, params, pl_file, attack_pos):
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
                    node_id = self.placedb.node_name2id_map[pos.group(1)]
                    attack_pos[node_id] = float(pos.group(2))
                    attack_pos[node_id+self.total_cell_num ] = float(pos.group(6))
                    orient = pos.group(4)
        if params.shift_factor[0] != 0 or params.shift_factor[1] != 0 or params.scale_factor != 1.0:
            self.scale_pl(params.shift_factor, params.scale_factor)
        logging.info("read_pl takes %.3f seconds" % (time.time()-tt))
        return attack_pos

        
    def attack(self, attack_method):

        print("current attack method", attack_method)
        if attack_method == "random_perturb":
            attack_pos = self.random_attack()
        elif attack_method == "random_perturb_constraint":
            attack_pos = self.random_attack_constraint()
        elif attack_method == "swap_pairs":
            attack_pos = self.swap_pairs_attack()
        elif attack_method == "random_perturb_row":
            attack_pos = self.random_attack_row()
        elif attack_method == "adaptive_region":
            attack_pos = self.adaptive_region()
        return attack_pos