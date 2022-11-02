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
import sys
from PIL import Image

# original module
from dataset.dataset_selector import dataset_generator
from models.model_selector import model_generator

# generate data id
def data_id_generator(img_path, cfg):
    data_id = 'unknown'
    if cfg.data.name == 'volleyball':
        video_num, seq_num, img_name = img_path.split('/')[-3:]
        img_num = img_name.split('.')[0]
        data_id = f'{video_num}_{seq_num}_{img_num}'
    elif 'videocoatt' in cfg.data.name:
        mode, seq_num, img_name = img_path.split('/')[-3:]
        img_num = img_name.split('.')[0]
        data_id = f'{mode}_{seq_num}_{img_num}'
    elif cfg.data.name == 'videoattentiontarget':
        vid_name, seq_num, img_name = img_path.split('/')[-3:]
        img_num = img_name.split('.')[0]
        data_id = f'{vid_name}_{seq_num}_{img_num}'
    elif cfg.data.name == 'toy':
        vid_name, seq_num, img_name = img_path.split('/')[-3:]
        img_num = img_name.split('.')[0]
        data_id = f'{vid_name}_{seq_num}_{img_num}'
    elif cfg.data.name == 'gazefollow':
        mode, seq_num, img_name = img_path.split('/')[-3:]
        img_num = img_name.split('.')[0]
        data_id = f'{mode}_{seq_num}_{img_num}'

    return data_id

# normalize heatmap
def norm_heatmap(img_heatmap):
    if np.min(img_heatmap) == np.max(img_heatmap):
        img_heatmap[:, :] = 0
    else: 
        img_heatmap = (img_heatmap - np.min(img_heatmap)) / (np.max(img_heatmap) - np.min(img_heatmap))
        img_heatmap *= 255

    return img_heatmap

def action_idx_to_name(action_idx):
    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
                'moving', 'setting', 'spiking', 'standing',
                'waiting']

    return ACTIONS[action_idx]

model_name_dic = {}
model_name_dic['volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_p_s_only'] = 'person_scene_joint_attention_heatmap'
# model_name_dic['volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_psfix_fusion'] = 'person_person_joint_attention_heatmap'
# model_name_dic['volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_psfix_fusion'] = 'person_scene_joint_attention_heatmap'
model_name_dic['volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_psfix_fusion'] = 'final_joint_attention_heatmap'
model_name_dic['volleyball-isa_bbox_GT_gaze_GT_act_GT'] = 'img_pred'

eval_results_list = []
for selected_model_name, use_heatmap in model_name_dic.items():
    data_id_list = []
    eval_results_model = []

    print("===> Getting configuration")
    parser = argparse.ArgumentParser(description="parameters for training")
    parser.add_argument("config", type=str, help="configuration yaml file path")
    args = parser.parse_args()
    cfg_arg = Dict(yaml.safe_load(open(args.config)))

    print("===> Making directories to save results")
    save_results_dir = os.path.join('results', cfg_arg.data.name, cfg_arg.exp_set.model_name)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    print(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, selected_model_name, 'train*.yaml'))
    saved_yaml_file_path = glob.glob(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, selected_model_name, 'train*.yaml'))[0]
    cfg = Dict(yaml.safe_load(open(saved_yaml_file_path)))
    cfg.update(cfg_arg)
    print(cfg)

    print("===> Building model")
    model_head, model_attention, model_saliency, cfg = model_generator(cfg)

    print("===> Building gpu configuration")
    cuda = cfg.exp_set.gpu_mode
    gpus_list = range(cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish+1)

    print("===> Building seed configuration")
    np.random.seed(cfg.exp_set.seed_num)
    torch.manual_seed(cfg.exp_set.seed_num)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms=True

    print("===> Loading trained model")
    weight_saved_dir = os.path.join(cfg.exp_set.save_folder, cfg.data.name, selected_model_name)
    model_head_weight_path = os.path.join(weight_saved_dir, "model_head_best.pth.tar")
    model_head.load_state_dict(torch.load(model_head_weight_path,  map_location='cuda:'+str(gpus_list[0])))

    if 'bbox_GT_gaze_GT_act_GT' in selected_model_name and 'dual-mid' in selected_model_name:
        model_saliency_weight_path = os.path.join(os.path.join(cfg.exp_set.save_folder,cfg.data.name, 'volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_p_s_only'), "model_saliency_best.pth.tar")
    else:
        model_saliency_weight_path = os.path.join(weight_saved_dir, "model_saliency_best.pth.tar")
    model_saliency.load_state_dict(torch.load(model_saliency_weight_path,  map_location='cuda:'+str(gpus_list[0])))

    model_attention_weight_path = os.path.join(weight_saved_dir, "model_gaussian_best.pth.tar")
    model_attention.load_state_dict(torch.load(model_attention_weight_path,  map_location='cuda:'+str(gpus_list[0])))

    if cuda:
        model_head = model_head.cuda(gpus_list[0])
        model_saliency = model_saliency.cuda(gpus_list[0])
        model_attention = model_attention.cuda(gpus_list[0])
        model_head.eval()
        model_saliency.eval()
        model_attention.eval()

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
    stop_iteration = 10000
    for iteration, batch in enumerate(test_data_loader,1):
        if iteration > stop_iteration:
            break

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
            out_head = model_head(batch)
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
            out_scene_feat = model_saliency(batch)
            batch = {**batch, **out_scene_feat}

            # joint attention estimation
            out_attention = model_attention(batch)
            out = {**out_head, **out_scene_feat, **out_attention, **batch}

        # get an image path
        img_path = out['rgb_path'][0]
        img = Image.open(img_path)
        original_width, original_height = img.size

        # get gt boxes
        gt_box = out['gt_box'].to('cpu').detach()[0]
        gt_x_min, gt_y_min, gt_x_max, gt_y_max = map(float, gt_box[0])
        gt_x_min, gt_x_max = map(lambda x:x*original_width, [gt_x_min, gt_x_max])
        gt_y_min, gt_y_max = map(lambda y:y*original_height, [gt_y_min, gt_y_max])
        gt_x_mid, gt_y_mid = (gt_x_min+gt_x_max)/2, (gt_y_min+gt_y_max)/2

        # define data id
        data_id = data_id_generator(img_path, cfg)
        data_id_list.append(data_id)
        print(f'Iter:{iteration}/{len(test_set)}, {data_id}')

        # get estimated joint attention coordinates
        if use_heatmap == 'img_pred':
            joint_attention_heatmap = out[use_heatmap].to('cpu').detach().numpy()[0]
        else:
            joint_attention_heatmap = out[use_heatmap].to('cpu').detach().numpy()[0,0]
        
        # joint_attention_heatmap = F.interpolate(out[use_heatmap], (original_height, original_width), mode='bilinear')
        # joint_attention_heatmap = joint_attention_heatmap[0, 0].to('cpu').detach().numpy()
        
        joint_attention_heatmap = cv2.resize(joint_attention_heatmap, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        pred_y_mid, pred_x_mid = np.unravel_index(np.argmax(joint_attention_heatmap), joint_attention_heatmap.shape)

        # calc l2 dist
        l2_dist_x = ((gt_x_mid-pred_x_mid)**2)**0.5
        l2_dist_y = ((gt_y_mid-pred_y_mid)**2)**0.5
        l2_dist_euc = (l2_dist_x**2+l2_dist_y**2)**0.5
        eval_results_model.append(l2_dist_euc)
        print(l2_dist_euc)

    eval_results_list.append(eval_results_model)

# save results as a csv file
eval_results_array = np.array(eval_results_list)
eval_results_array = eval_results_array.transpose()
df_eval_results = pd.DataFrame(eval_results_array, data_id_list, model_name_dic.keys())
save_csv_file_path = os.path.join(save_results_dir, f'selection.csv')
df_eval_results.to_csv(save_csv_file_path)