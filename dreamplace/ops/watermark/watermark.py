##
# @file   watermark.py
# @author Ruisi Zhang
# @date   Feb 2023
# @brief  Watermark Placement
#

import math
import torch

import dreamplace.ops.watermark.watermark_insert_ops as watermark_insert_ops
import dreamplace.ops.watermark.watermark_extract_ops as watermark_extract_ops
import dreamplace.ops.watermark.watermark_attack_ops as watermark_attack_ops

def watermark_insert(pos, params, placedb, data_collections, device):
    watermark_insert_instance = watermark_insert_ops.WM_insert(pos, params, placedb, data_collections, device)
    wm_pos, wm_cell_idx, keys, sub_time = watermark_insert_instance.insert()
    return wm_pos, wm_cell_idx, keys, sub_time
    

def watermark_extract(ori_pos, wm_pos, keys, params, placedb, data_collections, device, wm_dist, wm_cell_idx=None, phase="before_attack", attack_ratio=0):
    watermark_extract_instance = watermark_extract_ops.WM_extract(ori_pos, wm_pos, keys, params, placedb, data_collections, device, wm_dist, wm_cell_idx, phase, attack_ratio)
    extract_flag, extract_dist = watermark_extract_instance.extract()
    return extract_flag, extract_dist

def watermark_attack(pos, params, attack_method, placedb, data_collections, device, ratio):
    watermark_attack_instance = watermark_attack_ops.WM_attack(pos, params, placedb, data_collections, device, ratio)
    att_pos = watermark_attack_instance.attack(attack_method)
 
    return att_pos