# deep learning
import torch
import torch.nn as nn
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
import numpy as np
import warnings
warnings.filterwarnings("ignore") 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import json
from PIL import Image
import glob
import os

print("===> Getting configuration")
parser = argparse.ArgumentParser(description="parameters for training")
parser.add_argument("config", type=str, help="configuration yaml file path")
parser.add_argument("-model_name", type=str, help="model name")
args = parser.parse_args()
cfg_arg = Dict(yaml.safe_load(open(args.config)))
if args.model_name:
    cfg_arg.exp_set.model_name = args.model_name
print(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, cfg_arg.exp_set.model_name, 'train*.yaml'))
saved_yaml_file_path = glob.glob(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, cfg_arg.exp_set.model_name, 'train*.yaml'))[0]
cfg = Dict(yaml.safe_load(open(saved_yaml_file_path)))
cfg.update(cfg_arg)

print("===> Setting gpu numbers")
# update gpu number for roi_align
import os
os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg_arg.exp_set.gpu_start}'
cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish = 0, 0
gpus_list = range(cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish+1)
cuda = cfg.exp_set.gpu_mode

# original module
from dataset.dataset_selector import dataset_generator
from models.model_selector import model_generator

def data_type_id_generator(cfg):
    data_type_id = f'bbox_{cfg.exp_params.bbox_types}_gaze_{cfg.exp_params.gaze_types}_act_{cfg.exp_params.action_types}_blur_{cfg.exp_params.use_blured_img}'
    return data_type_id

# print("===> Getting configuration")
# parser = argparse.ArgumentParser(description="parameters for training")
# parser.add_argument("config", type=str, help="configuration yaml file path")
# args = parser.parse_args()
# cfg_arg = Dict(yaml.safe_load(open(args.config)))
# saved_yaml_file_path = glob.glob(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, cfg_arg.exp_set.model_name, 'train*.yaml'))[0]
# cfg = Dict(yaml.safe_load(open(saved_yaml_file_path)))
# cfg.update(cfg_arg)
# print(cfg)

# print("===> Building gpu configuration")
# cuda = cfg.exp_set.gpu_mode
# gpus_list = range(cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish+1)

print("===> Building model")
model_head, model_attention, model_saliency, model_fusion, cfg = model_generator(cfg)

print("===> Building seed configuration")
np.random.seed(cfg.exp_set.seed_num)
torch.manual_seed(cfg.exp_set.seed_num)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms=True

print("===> Loading trained model")
model_name = cfg.exp_set.model_name
weight_saved_dir = os.path.join(cfg.exp_set.save_folder,cfg.data.name, model_name)
model_head_weight_path = os.path.join(weight_saved_dir, "model_head_best.pth.tar")
model_head.load_state_dict(torch.load(model_head_weight_path,  map_location='cuda:'+str(gpus_list[0])))

if 'bbox_PRED_gaze_PRED_act_PRED' in model_name and cfg.model_params.model_type == 'ja_transformer_dual':
    model_saliency_weight_path = os.path.join(os.path.join(cfg.exp_set.save_folder,cfg.data.name, 'volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_PRED_gaze_PRED_act_PRED_p_s_only'), "model_saliency_best.pth.tar")
elif 'bbox_GT_gaze_GT_act_GT' in model_name and cfg.model_params.model_type == 'ja_transformer_dual':
    model_saliency_weight_path = os.path.join(os.path.join(cfg.exp_set.save_folder,cfg.data.name, 'volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_p_s_only'), "model_saliency_best.pth.tar")
else:
    model_saliency_weight_path = os.path.join(os.path.join(cfg.exp_set.save_folder,cfg.data.name, model_name), "model_saliency_best.pth.tar")

model_saliency_weight_path = os.path.join(os.path.join(cfg.exp_set.save_folder,cfg.data.name, model_name), "model_saliency_best.pth.tar")
model_saliency.load_state_dict(torch.load(model_saliency_weight_path,  map_location='cuda:'+str(gpus_list[0])))

model_attention_weight_path = os.path.join(weight_saved_dir, "model_gaussian_best.pth.tar")
model_attention.load_state_dict(torch.load(model_attention_weight_path,  map_location='cuda:'+str(gpus_list[0])))
# model_attention.load_state_dict(torch.load(model_attention_weight_path,  map_location='cuda:'+str(gpus_list[0])), strict=False)

model_fusion_weight_path = os.path.join(weight_saved_dir, "model_fusion_best.pth.tar")
# model_fusion.load_state_dict(torch.load(model_fusion_weight_path,  map_location='cuda:'+str(gpus_list[0])))
# model_fusion.load_state_dict(torch.load(model_fusion_weight_path,  map_location='cuda:'+str(gpus_list[0])), strict=False)
if os.path.exists(model_fusion_weight_path):
    model_fusion.load_state_dict(torch.load(model_fusion_weight_path,  map_location='cuda:'+str(gpus_list[0])))

if cuda:
    model_head = model_head.cuda(gpus_list[0])
    model_saliency = model_saliency.cuda(gpus_list[0])
    model_attention = model_attention.cuda(gpus_list[0])
    model_fusion = model_fusion.cuda(gpus_list[0])
    model_head.eval()
    model_saliency.eval()
    model_attention.eval()
    model_fusion.eval()

# view learned fusion coeficient
# fusion_weight = model_fusion.state_dict()['final_fusion_weight'].detach().cpu()
# m = nn.Softmax()
# fusion_weight = m(fusion_weight)
# print(fusion_weight)
# sys.exit()

print("===> Loading dataset")
mode = cfg.exp_set.mode
test_set = dataset_generator(cfg, mode)
test_data_loader = DataLoader(dataset=test_set,
                                batch_size=cfg.exp_set.batch_size,
                                shuffle=False,
                                num_workers=cfg.exp_set.num_workers,
                                pin_memory=True)
print('{} demo samples found'.format(len(test_set)))

print("===> Making directories to save results")
if cfg.exp_set.test_gt_gaze:
    model_name = model_name + f'_use_gt_gaze'

data_type_id = data_type_id_generator(cfg)
save_results_dir = os.path.join('results', cfg.data.name, model_name, 'eval_results', data_type_id)
if not os.path.exists(save_results_dir):
    os.makedirs(save_results_dir)

print("===> Starting eval processing")
l2_dist_array = np.zeros((len(test_data_loader), 9))
for iteration, batch in enumerate(test_data_loader):
    print(f'{iteration}/{len(test_data_loader)}')

    # init heatmaps
    _, _, num_people = batch['head_img'].shape[0:3]
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
                if torch.is_tensor(val):
                    batch[key] = Variable(val).cuda(gpus_list[0])

        if cfg.model_params.use_position:
            input_feature = batch['head_feature'].clone() 
        else:
            input_feature = batch['head_feature'].clone()
            input_feature[:, :, :, :2] = input_feature[:, :, :, :2] * 0
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
        batch = {**out_head, **out_attention, **batch}

        # fusion network
        out_fusion = model_fusion(batch)
        out = {**batch, **out_fusion}

    gt_box = out['gt_box'].to('cpu').detach()[0].numpy()
    person_person_attention_heatmap = out['person_person_attention_heatmap'].to('cpu').detach()
    person_person_joint_attention_heatmap = out['person_person_joint_attention_heatmap'].to('cpu').detach()
    person_scene_attention_heatmap = out['person_scene_attention_heatmap'].to('cpu').detach()
    person_scene_joint_attention_heatmap = out['person_scene_joint_attention_heatmap'].to('cpu').detach()
    final_joint_attention_heatmap = out['final_joint_attention_heatmap'].to('cpu').detach()

    # calc a center of gt bbox
    img = Image.open(batch['rgb_path'][0][0])
    original_width, original_height = img.size

    # set key frame index for evaluation
    if cfg.exp_params.use_frame_type == 'all':
        key_frame_idx = 4
    elif cfg.exp_params.use_frame_type == 'mid':
        key_frame_idx = 0
    else:
        assert False, f'Not implemented frame type: {cfg.exp_params.use_frame_type}'

    gt_box = gt_box[key_frame_idx, :, :]
    person_person_joint_attention_heatmap = person_person_joint_attention_heatmap[:, key_frame_idx, :, :]
    person_scene_joint_attention_heatmap = person_scene_joint_attention_heatmap[:, key_frame_idx, :, :]
    final_joint_attention_heatmap = final_joint_attention_heatmap[:, key_frame_idx, :, :]

    peak_x_min_gt, peak_y_min_gt, peak_x_max_gt, peak_y_max_gt = gt_box[0]
    peak_x_mid_gt, peak_y_mid_gt = (peak_x_min_gt+peak_x_max_gt)/2, (peak_y_min_gt+peak_y_max_gt)/2
    peak_x_mid_gt, peak_y_mid_gt = peak_x_mid_gt*original_width, peak_y_mid_gt*original_height
    peak_x_mid_gt, peak_y_mid_gt = map(int, [peak_x_mid_gt, peak_y_mid_gt])

    # calc a centers of pred bboxes
    person_person_joint_attention_heatmap = F.interpolate(person_person_joint_attention_heatmap, (original_height, original_width), mode='bilinear')
    person_person_joint_attention_heatmap = person_person_joint_attention_heatmap[0, 0, :, :]
    pred_y_mid_p_p, pred_x_mid_p_p = np.unravel_index(np.argmax(person_person_joint_attention_heatmap), person_person_joint_attention_heatmap.shape)

    person_scene_joint_attention_heatmap = F.interpolate(person_scene_joint_attention_heatmap, (original_height, original_width), mode='bilinear')
    person_scene_joint_attention_heatmap = person_scene_joint_attention_heatmap[0, 0, :, :]
    pred_y_mid_p_s, pred_x_mid_p_s = np.unravel_index(np.argmax(person_scene_joint_attention_heatmap), person_scene_joint_attention_heatmap.shape)

    final_joint_attention_heatmap = F.interpolate(final_joint_attention_heatmap, (original_height, original_width), mode='bilinear')
    final_joint_attention_heatmap = final_joint_attention_heatmap[0, 0, :, :]
    pred_y_mid_final, pred_x_mid_final = np.unravel_index(np.argmax(final_joint_attention_heatmap), final_joint_attention_heatmap.shape)

    # calc metrics
    l2_dist_x_p_p, l2_dist_y_p_p = np.power(np.power(pred_x_mid_p_p-peak_x_mid_gt, 2), 0.5), np.power(np.power(pred_y_mid_p_p-peak_y_mid_gt, 2), 0.5)
    l2_dist_euc_p_p = np.power(np.power(l2_dist_x_p_p, 2) + np.power(l2_dist_y_p_p, 2), 0.5)

    l2_dist_x_p_s, l2_dist_y_p_s = np.power(np.power(pred_x_mid_p_s-peak_x_mid_gt, 2), 0.5), np.power(np.power(pred_y_mid_p_s-peak_y_mid_gt, 2), 0.5)
    l2_dist_euc_p_s = np.power(np.power(l2_dist_x_p_s, 2) + np.power(l2_dist_y_p_s, 2), 0.5)

    l2_dist_x_final, l2_dist_y_final = np.power(np.power(pred_x_mid_final-peak_x_mid_gt, 2), 0.5), np.power(np.power(pred_y_mid_final-peak_y_mid_gt, 2), 0.5)
    l2_dist_euc_final = np.power(np.power(l2_dist_x_final, 2) + np.power(l2_dist_y_final, 2), 0.5)

    print(f'Dis={l2_dist_euc_final:.0f}, GT=({peak_x_mid_gt:.0f},{peak_y_mid_gt:.0f}), peak=({pred_x_mid_final:.0f},{pred_y_mid_final:.0f})')
    l2_dist_array[iteration, 0] = l2_dist_x_p_p
    l2_dist_array[iteration, 1] = l2_dist_y_p_p
    l2_dist_array[iteration, 2] = l2_dist_euc_p_p
    l2_dist_array[iteration, 3] = l2_dist_x_p_s
    l2_dist_array[iteration, 4] = l2_dist_y_p_s
    l2_dist_array[iteration, 5] = l2_dist_euc_p_s
    l2_dist_array[iteration, 6] = l2_dist_x_final
    l2_dist_array[iteration, 7] = l2_dist_y_final
    l2_dist_array[iteration, 8] = l2_dist_euc_final

    # stop_iter = 20
    # if iteration > stop_iter:
        # break

# save metrics in a dict
metrics_dict = {}

# save l2 dist
# l2_dist_array = l2_dist_array[:stop_iter, :]
l2_dist_mean = np.mean(l2_dist_array, axis=0)
l2_dist_list = [
                'l2_dist_x_p_p', 'l2_dist_y_p_p', 'l2_dist_euc_p_p',
                'l2_dist_x_p_s', 'l2_dist_y_p_s', 'l2_dist_euc_p_s',
                'l2_dist_x_final', 'l2_dist_y_final', 'l2_dist_euc_final',
                ]

for l2_dist_idx, l2_dist_type in enumerate(l2_dist_list):
    metrics_dict[l2_dist_type] = l2_dist_mean[l2_dist_idx]

# save l2 dist (Histgrad analysis)
for l2_dist_idx, l2_dist_type in enumerate(l2_dist_list):
    if not l2_dist_type in ['l2_dist_euc_p_p', 'l2_dist_euc_p_s', 'l2_dist_euc_final']:
        continue
    save_figure_path = os.path.join(save_results_dir, f'{l2_dist_type}_hist.png')
    plt.figure()
    plt.hist(l2_dist_array[:, l2_dist_idx], bins=10)
    plt.xlim(0, 800)
    plt.savefig(save_figure_path)

# save detection rate
det_rate_list = [f'Det p-p (Thr={det_thr})' for det_thr in range(0, 110, 10)]
for det_rate_idx, det_rate_type in enumerate(det_rate_list, 1):
    det_rate = l2_dist_array[:, 2]<(det_rate_idx*10)
    det_rate_mean = np.mean(det_rate) * 100
    metrics_dict[det_rate_type] = det_rate_mean
det_rate_list = [f'Det p-s (Thr={det_thr})' for det_thr in range(0, 110, 10)]
for det_rate_idx, det_rate_type in enumerate(det_rate_list, 1):
    det_rate = l2_dist_array[:, 5]<(det_rate_idx*10)
    det_rate_mean = np.mean(det_rate) * 100
    metrics_dict[det_rate_type] = det_rate_mean
det_rate_list = [f'Det final (Thr={det_thr})' for det_thr in range(0, 110, 10)]
for det_rate_idx, det_rate_type in enumerate(det_rate_list, 1):
    det_rate = l2_dist_array[:, 8]<(det_rate_idx*10)
    det_rate_mean = np.mean(det_rate) * 100
    metrics_dict[det_rate_type] = det_rate_mean

# save metrics into json files
save_results_path = os.path.join(save_results_dir, 'eval_results.json')
with open(save_results_path, 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print(metrics_dict)
for met_name, met_val in metrics_dict.items():
    print(met_name, met_val)