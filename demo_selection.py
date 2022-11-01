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

print("===> Getting configuration")
parser = argparse.ArgumentParser(description="parameters for training")
parser.add_argument("config", type=str, help="configuration yaml file path")
args = parser.parse_args()
cfg_arg = Dict(yaml.safe_load(open(args.config)))

# model_name_list = ['volleyball-isa_bbox_GT_gaze_GT_act_GT']
model_name_list = ['volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_p_s_only']
for selected_model_name in model_name_list:
    print(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, selected_model_name, 'train*.yaml'))
    saved_yaml_file_path = glob.glob(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, selected_model_name, 'train*.yaml'))[0]
    cfg = Dict(yaml.safe_load(open(saved_yaml_file_path)))
    cfg.update(cfg_arg)

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

    model_saliency_weight_path = os.path.join(weight_saved_dir, "model_saliency_best.pth.tar")
    if os.path.exists(model_saliency_weight_path):
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

    print("===> Making directories to save results")
    save_results_dir = os.path.join('results', cfg.data.name, cfg_arg.exp_set.model_name)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    print("===> Starting demo processing")
    stop_iteration = 20
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
            head_vector = out_head['head_vector']
            batch['head_img_extract'] = out_head['head_img_extract']

            if cfg.exp_params.use_gt_gaze:
                batch['head_vector'] = batch['head_vector_gt']
            else:
                batch['head_vector'] = out_head['head_vector']

            # change position inputs
            if cfg.model_params.use_gaze:
                batch['input_gaze'] = head_vector.clone() 
            else:
                batch['input_gaze'] = head_vector.clone() * 0

            # scene feature extraction
            out_scene_feat = model_saliency(batch)
            batch = {**batch, **out_scene_feat}

            # joint attention estimation
            out_attention = model_attention(batch)
            out = {**out_head, **out_scene_feat, **out_attention, **batch}

        gt_box = out['gt_box'].to('cpu').detach()[0]
        img_path = out['rgb_path'][0]
        person_person_joint_attention_heatmap = out['person_person_joint_attention_heatmap'].to('cpu').detach()[0].numpy()
        person_scene_joint_attention_heatmap = out['person_scene_joint_attention_heatmap'].to('cpu').detach()[0].numpy()
        final_joint_attention_heatmap = out['final_joint_attention_heatmap'].to('cpu').detach()[0].numpy()

        # redefine image size
        img = Image.open(batch['rgb_path'][0])
        original_width, original_height = img.size
        cfg.exp_set.resize_height = original_height
        cfg.exp_set.resize_width = original_width

        # define data id
        data_type_id = ''
        data_id = data_id_generator(img_path, cfg)
        print(f'Iter:{iteration}, {data_id}, {data_type_id}')

        # save joint attention estimation as a superimposed image
        img = cv2.resize(img, (cfg.exp_set.resize_width, cfg.exp_set.resize_height))
        # person_person_joint_attention_heatmap = cv2.imread(os.path.join(save_image_dir_dic['person_person_jo_att'], data_type_id, f'{mode}_{data_id}_person_person_jo_att.png'), cv2.IMREAD_GRAYSCALE)
        # person_scene_joint_attention_heatmap = cv2.imread(os.path.join(save_image_dir_dic['person_scene_jo_att'], data_type_id, f'{mode}_{data_id}_person_scene_jo_att.png'), cv2.IMREAD_GRAYSCALE)
        # final_joint_attention_heatmap = cv2.imread(os.path.join(save_image_dir_dic['final_jo_att'], data_type_id, f'{mode}_{data_id}_final_jo_att.png'), cv2.IMREAD_GRAYSCALE)


        person_person_joint_attention_heatmap = cv2.resize(person_person_joint_attention_heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        person_scene_joint_attention_heatmap = cv2.resize(person_scene_joint_attention_heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        final_joint_attention_heatmap = cv2.resize(final_joint_attention_heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        person_person_joint_attention_heatmap = norm_heatmap(person_person_joint_attention_heatmap).astype(np.uint8)
        person_scene_joint_attention_heatmap = norm_heatmap(person_scene_joint_attention_heatmap).astype(np.uint8)
        final_joint_attention_heatmap = norm_heatmap(final_joint_attention_heatmap).astype(np.uint8)

        # get estimated joint attention coordinates
        pred_y_mid_p_p, pred_x_mid_p_p = np.unravel_index(np.argmax(person_person_joint_attention_heatmap), person_person_joint_attention_heatmap.shape)
        pred_y_mid_p_s, pred_x_mid_p_s = np.unravel_index(np.argmax(person_scene_joint_attention_heatmap), person_scene_joint_attention_heatmap.shape)
        pred_y_mid_final, pred_x_mid_final = np.unravel_index(np.argmax(final_joint_attention_heatmap), final_joint_attention_heatmap.shape)

        gt_x_min, gt_y_min, gt_x_max, gt_y_max = map(float, gt_box[0])
        gt_x_min, gt_x_max = map(lambda x:x*cfg.exp_set.resize_width, [gt_x_min, gt_x_max])
        gt_y_min, gt_y_max = map(lambda y:y*cfg.exp_set.resize_height, [gt_y_min, gt_y_max])
        gt_x_mid, gt_y_mid = (gt_x_min+gt_x_max)/2, (gt_y_min+gt_y_max)/2
        l2_dist_x = ((gt_x_mid-pred_x_mid)**2)**0.5
        l2_dist_y = ((gt_y_mid-pred_y_mid)**2)**0.5
        l2_dist_euc = (l2_dist_x**2+l2_dist_y**2)**0.5

        sys.exit()