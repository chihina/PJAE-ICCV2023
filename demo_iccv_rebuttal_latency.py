# deep learning
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# general module
import numpy as np
import argparse
import yaml
from addict import Dict
import cv2
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore") 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import time
import sys
import json
from PIL import Image

# original module
from dataset.dataset_selector import dataset_generator
from models.model_selector import model_generator

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# define test cases
test_model_type_list = []
test_model_name_list = []

test_model_type_list.append('ISA')
test_model_name_list.append('videocoatt-isa_bbox_GT_gaze_GT')

test_model_type_list.append('DAVT')
test_model_name_list.append('videocoatt-p_p_field_deep_p_s_davt_freeze')

test_model_type_list.append('HGTD')
test_model_name_list.append('videoattentiontarget-hgt-high')

test_model_type_list.append('Ours')
test_model_name_list.append('videocoatt-p_p_field_deep_p_s_davt_scalar_weight_fix_token_only_GT')

test_sample_num = 50
time_array = np.zeros((len(test_model_type_list), test_sample_num))

for test_model_index, test_model_type in enumerate(test_model_type_list):
    test_model_name = test_model_name_list[test_model_index]
    print(test_model_type)

    print("===> Getting configuration")
    parser = argparse.ArgumentParser(description="parameters for training")
    parser.add_argument("config", type=str, help="configuration yaml file path")
    args = parser.parse_args()
    cfg_arg = Dict(yaml.safe_load(open(args.config)))
    # print(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, test_model_name, 'train*.yaml'))
    saved_yaml_file_path = glob.glob(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, test_model_name, 'train*.yaml'))[0]
    cfg = Dict(yaml.safe_load(open(saved_yaml_file_path)))
    cfg.update(cfg_arg)

    print("===> Building model")
    model_head, model_attention, model_saliency, model_fusion, cfg = model_generator(cfg)

    print("===> Building gpu configuration")
    cuda = cfg.exp_set.gpu_mode
    gpus_list = range(cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish+1)

    print("===> Building seed configuration")
    np.random.seed(cfg.exp_set.seed_num)
    torch.manual_seed(cfg.exp_set.seed_num)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms=True

    if cuda:
        model_head = model_head.cuda(gpus_list[0])
        model_saliency = model_saliency.cuda(gpus_list[0])
        model_attention = model_attention.cuda(gpus_list[0])
        model_fusion = model_fusion.cuda(gpus_list[0])
        model_head.eval()
        model_saliency.eval()
        model_attention.eval()
        model_fusion.eval()

    print("===> Loading dataset")
    mode = cfg.exp_set.mode
    test_set = dataset_generator(cfg, mode)
    test_data_loader = DataLoader(dataset=test_set,
                                    batch_size=cfg.exp_set.batch_size,
                                    shuffle=True,
                                    num_workers=cfg.exp_set.num_workers,
                                    pin_memory=True)
    print('{} demo samples found'.format(len(test_set)))

    print("===> Starting demo processing")
    for iteration, batch in enumerate(test_data_loader,1):
        if iteration > test_sample_num:
            break

        # define data id
        img_path = batch['rgb_path'][0]
        # if iteration % 10 == 0:
            # print(f'Iter:{iteration}')

        # init heatmaps
        num_people = batch['head_img'].shape[1]
        x_axis_map = torch.arange(0, cfg.exp_set.resize_width, device=f'cuda:{gpus_list[0]}').reshape(1, -1)/(cfg.exp_set.resize_width)
        x_axis_map = torch.tile(x_axis_map, (cfg.exp_set.resize_height, 1))
        y_axis_map = torch.arange(0, cfg.exp_set.resize_height, device=f'cuda:{gpus_list[0]}').reshape(-1, 1)/(cfg.exp_set.resize_height)
        y_axis_map = torch.tile(y_axis_map, (1, cfg.exp_set.resize_width))
        xy_axis_map = torch.cat((x_axis_map[None, :, :], y_axis_map[None, :, :]))[None, None, :, :, :]
        xy_axis_map = torch.tile(xy_axis_map, (cfg.exp_set.batch_size, num_people, 1, 1, 1))
        head_x_map = torch.ones((cfg.exp_set.batch_size, num_people, 1, cfg.exp_set.resize_height, cfg.exp_set.resize_width), device=f'cuda:{gpus_list[0]}')
        head_y_map = torch.ones((cfg.exp_set.batch_size, num_people, 1, cfg.exp_set.resize_height, cfg.exp_set.resize_width), device=f'cuda:{gpus_list[0]}')
        head_xy_map = torch.cat((head_x_map, head_y_map), 2)
        gaze_x_map = torch.ones((cfg.exp_set.batch_size, num_people, 1, cfg.exp_set.resize_height, cfg.exp_set.resize_width), device=f'cuda:{gpus_list[0]}')
        gaze_y_map = torch.ones((cfg.exp_set.batch_size, num_people, 1, cfg.exp_set.resize_height, cfg.exp_set.resize_width), device=f'cuda:{gpus_list[0]}')
        gaze_xy_map = torch.cat((gaze_x_map, gaze_y_map), 2)
        xy_axis_map = xy_axis_map.float()
        head_xy_map = head_xy_map.float()
        gaze_xy_map = gaze_xy_map.float()
        batch['xy_axis_map'] = xy_axis_map
        batch['head_xy_map'] = head_xy_map
        batch['gaze_xy_map'] = gaze_xy_map

        with torch.no_grad():
            # move data into gpu
            if cuda:
                for key, val in batch.items():
                    if key != 'rgb_path':
                        batch[key] = Variable(val).cuda(gpus_list[0])

            if cfg.model_params.use_position:
                input_feature = batch['head_feature'].clone() 
            else:
                input_feature = batch['head_feature'].clone()
                input_feature[:, :, :2] = input_feature[:, :, :2] * 0
            batch['input_feature'] = input_feature

            # head pose estimation
            t1_gaze = time_synchronized()
            out_head = model_head(batch)
            t2_gaze = time_synchronized()
            t_gaze = t2_gaze - t1_gaze
            # print('gaze', t2_gaze-t1_gaze)
            batch['head_img_extract'] = out_head['head_img_extract']

            if cfg.exp_params.gaze_types == 'GT':
                batch['head_vector'] = batch['head_vector_gt']
            else:
                batch['head_vector'] = out_head['head_vector']

            # change position inputs
            if cfg.model_params.use_gaze:
                batch['input_gaze'] = batch['head_vector'].clone() 
            else:
                batch['input_gaze'] = batch['head_vector'].clone() * 0

            # scene feature extraction
            t1_p_s = time_synchronized()
            out_scene_feat = model_saliency(batch)
            t2_p_s = time_synchronized()
            t_p_s = t2_p_s - t1_p_s
            # print('person_scene', t2_p_s-t1_p_s)
            batch = {**batch, **out_scene_feat}

            # joint attention estimation
            t1_p_p = time_synchronized()
            out_attention = model_attention(batch)
            t2_p_p = time_synchronized()
            t_p_p = t2_p_p - t1_p_p
            # print('pjat', t2_main-t1_main)
            batch = {**batch, **out_attention}

            # fusion network
            t1_fusion = time_synchronized()
            out_fusion = model_fusion(batch)
            t2_fusion = time_synchronized()
            t_fusion = t2_fusion - t1_fusion
            batch = {**batch, **out_fusion}

            if test_model_type == 'ISA':
                save_time = t_gaze + t_p_p
            elif test_model_type == 'DAVT':
                save_time = t_p_s
            elif test_model_type == 'HGTD':
                save_time = t_p_p
            elif test_model_type == 'Ours':
                save_time = t_gaze + t_p_s + t_p_p + t_fusion
            else:
                pass

            time_array[test_model_index, iteration-1] = save_time

# save lentency
save_time_dic = {}
save_json_file_path = 'demo_iccv_rebuttal_latency.json'

# print latency
time_array_average = np.average(time_array, axis=1)
for test_model_index, test_model_type in enumerate(test_model_type_list):
    print(test_model_type, time_array_average[test_model_index])
    save_time_dic[test_model_type] = time_array_average[test_model_index]

with open(save_json_file_path, 'w') as f:
    json.dump(save_time_dic, f)