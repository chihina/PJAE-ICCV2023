# deep learning
from select import select
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
import sys
import json
from PIL import Image
import glob

from sklearn.cluster import MeanShift
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# original module
from dataset.dataset_selector import dataset_generator
from models.model_selector import model_generator

def data_type_id_generator(cfg):
    data_type_id = f'{cfg.exp_set.mode}_gt_gaze_{cfg.exp_params.test_gt_gaze}_head_conf_{cfg.exp_params.test_heads_conf}'
    return data_type_id

def each_data_type_id_generator(head_vector_gt, head_tensor, gt_box, cfg):

    dets_people_num = np.sum(np.sum(head_vector_gt, axis=-1) != 0)
    # define data id of dets people
    dets_people_num = np.sum(np.sum(head_vector_gt, axis=-1) != 0)
    if dets_people_num <= 3:
        dets_people_id = '0<peo<3'
    else:
        dets_people_id = '3<=peo'

    # define data id of gaze estimation
    head_vector_gt_cos = head_vector_gt[:dets_people_num, :]
    head_vector_pred_cos = head_tensor[:dets_people_num, :2]
    head_gt_pred_cos_sim = np.sum(head_vector_gt_cos * head_vector_pred_cos, axis=1)
    head_gt_pred_cos_sim_ave = np.sum(head_gt_pred_cos_sim) / dets_people_num
    if head_gt_pred_cos_sim_ave < 0.5:
        gaze_error_id = '0_0<gaze<0_5'
    else:
        gaze_error_id = '0_5_gaze<1_0'

    # define data id of joint attention size
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_box[0, :]
    gt_x_size, gt_y_size = gt_x_max-gt_x_min, gt_y_max-gt_y_min
    gt_size = ((gt_x_size**2)+(gt_y_size**2))**0.5
    gt_size_thresh = 0.2
    if gt_size < gt_size_thresh:
        gt_size_id = f'0_0<size<0_2'
    else:
        gt_size_id = '0_2<size'

    data_type_id = f'{dets_people_id}:{gaze_error_id}:{gt_size_id}'
    # data_type_id = f'{dets_people_id}:{gaze_error_id}'
    # data_type_id = ''

    return data_type_id

print("===> Getting configuration")
parser = argparse.ArgumentParser(description="parameters for training")
parser.add_argument("config", type=str, help="configuration yaml file path")
args = parser.parse_args()
cfg_arg = Dict(yaml.safe_load(open(args.config)))
saved_yaml_file_path = glob.glob(os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, cfg_arg.exp_set.model_name, 'train*.yaml'))[0]
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
model_name = cfg.exp_set.model_name
weight_saved_dir = os.path.join(cfg.exp_set.save_folder,cfg.data.name, model_name)
model_head_weight_path = os.path.join(weight_saved_dir, "model_head_best.pth.tar")
model_saliency_weight_path = os.path.join(weight_saved_dir, "model_saliency_best.pth.tar")
model_attention_weight_path = os.path.join(weight_saved_dir, "model_gaussian_best.pth.tar")
model_head.load_state_dict(torch.load(model_head_weight_path,  map_location='cuda:'+str(gpus_list[0])))
model_saliency.load_state_dict(torch.load(model_saliency_weight_path,  map_location='cuda:'+str(gpus_list[0])))
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
cfg.data.name = 'videocoatt_no_att'
test_set = dataset_generator(cfg, mode)
cfg.data.name = 'videocoatt'
test_data_loader = DataLoader(dataset=test_set,
                                batch_size=cfg.exp_set.batch_size,
                                shuffle=False,
                                num_workers=cfg.exp_set.num_workers,
                                pin_memory=True)
print('{} demo samples found'.format(len(test_set)))

print("===> Making directories to save results")
if cfg.exp_set.use_gt_gaze:
    model_name = model_name + f'_use_gt_gaze'

data_type_id = data_type_id_generator(cfg)
save_results_dir = os.path.join('results', cfg.data.name, model_name, 'eval_results', data_type_id)
if not os.path.exists(save_results_dir):
    os.makedirs(save_results_dir)

print("===> Starting eval processing")
l2_dist_list = []
pred_acc_list = []
each_data_type_id_dic = {}
for iteration, batch in enumerate(test_data_loader):
    print(f'{iteration}/{len(test_data_loader)}')

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

        out_attention = model_attention(batch)
        out = {**out_head, **out_attention, **batch}

    img_gt = out['img_gt'].to('cpu').detach()[0]
    head_feature = out['head_feature'].to('cpu').detach()[0].numpy()
    gt_box = out['gt_box'].to('cpu').detach()[0].numpy()
    att_inside_flag = out['att_inside_flag'].to('cpu').detach()[0]

    person_person_attention_heatmap = out['person_person_attention_heatmap'].to('cpu').detach()[0].numpy()
    person_person_joint_attention_heatmap = out['person_person_joint_attention_heatmap'].to('cpu').detach()[0, 0].numpy()
    person_scene_attention_heatmap = out['person_scene_attention_heatmap'].to('cpu').detach()[0].numpy()
    person_scene_joint_attention_heatmap = out['person_scene_joint_attention_heatmap'].to('cpu').detach()[0, 0].numpy()
    final_joint_attention_heatmap = out['final_joint_attention_heatmap'].to('cpu').detach()[0, 0].numpy()

    # generate each data id
    each_data_type_id = ''
    if not each_data_type_id in each_data_type_id_dic.keys():
        each_data_type_id_dic[each_data_type_id] = len(each_data_type_id_dic.keys())
    each_data_type_id_idx = each_data_type_id_dic[each_data_type_id]

    # get a padding number
    people_padding_mask = (np.sum(head_feature, axis=-1) != 0)
    people_padding_num = np.sum(people_padding_mask)
    if people_padding_num == 0:
        continue

    # calc centers of gt bbox
    gt_box_ja_list = []
    for person_idx in range(people_padding_num):
        peak_x_min_gt, peak_y_min_gt, peak_x_max_gt, peak_y_max_gt = gt_box[person_idx, :]
        peak_x_mid_gt, peak_y_mid_gt = (peak_x_min_gt+peak_x_max_gt)/2, (peak_y_min_gt+peak_y_max_gt)/2
        peak_x_mid_gt, peak_y_mid_gt = peak_x_mid_gt*cfg.exp_set.resize_width, peak_y_mid_gt*cfg.exp_set.resize_height
        peak_x_mid_gt, peak_y_mid_gt = map(int, [peak_x_mid_gt, peak_y_mid_gt])
        save_gt_peak = [peak_x_mid_gt, peak_y_mid_gt]
        if save_gt_peak not in gt_box_ja_list and (save_gt_peak != [0, 0]):
            gt_box_ja_list.append(save_gt_peak)
    gt_box_ja_array = np.array(gt_box_ja_list)

    co_att_flag_gt = np.sum(gt_box, axis=(0, 1)) != 0
    person_person_joint_attention_heatmap = cv2.resize(person_person_joint_attention_heatmap, (cfg.exp_set.resize_width, cfg.exp_set.resize_height))
    person_scene_joint_attention_heatmap = cv2.resize(person_scene_joint_attention_heatmap, (cfg.exp_set.resize_width, cfg.exp_set.resize_height))
    final_joint_attention_heatmap = cv2.resize(final_joint_attention_heatmap, (cfg.exp_set.resize_width, cfg.exp_set.resize_height))
    person_person_joint_attention_heatmap_peak_val = np.max(person_person_joint_attention_heatmap)
    person_scene_joint_attention_heatmap_peak_val = np.max(person_scene_joint_attention_heatmap)
    final_joint_attention_heatmap_peak_val = np.max(final_joint_attention_heatmap)

    co_att_flag_pred = final_joint_attention_heatmap_peak_val > 0.15
    pred_acc_list.append([co_att_flag_gt, co_att_flag_pred])
    if not co_att_flag_gt:
        continue

    pred_y_mid_p_p, pred_x_mid_p_p = np.unravel_index(np.argmax(person_person_joint_attention_heatmap), person_person_joint_attention_heatmap.shape)
    pred_y_mid_p_s, pred_x_mid_p_s = np.unravel_index(np.argmax(person_scene_joint_attention_heatmap), person_scene_joint_attention_heatmap.shape)
    pred_y_mid_final, pred_x_mid_final = np.unravel_index(np.argmax(final_joint_attention_heatmap), final_joint_attention_heatmap.shape)

    for gt_box_idx in range(gt_box_ja_array.shape[0]):
        peak_x_mid_gt, peak_y_mid_gt = gt_box_ja_array[gt_box_idx, :]

        l2_dist_x_p_p, l2_dist_y_p_p = np.power(np.power(pred_x_mid_p_p-peak_x_mid_gt, 2), 0.5), np.power(np.power(pred_y_mid_p_p-peak_y_mid_gt, 2), 0.5)
        l2_dist_euc_p_p = np.power(np.power(l2_dist_x_p_p, 2) + np.power(l2_dist_y_p_p, 2), 0.5)

        l2_dist_x_p_s, l2_dist_y_p_s = np.power(np.power(pred_x_mid_p_s-peak_x_mid_gt, 2), 0.5), np.power(np.power(pred_y_mid_p_s-peak_y_mid_gt, 2), 0.5)
        l2_dist_euc_p_s = np.power(np.power(l2_dist_x_p_s, 2) + np.power(l2_dist_y_p_s, 2), 0.5)

        l2_dist_x_final, l2_dist_y_final = np.power(np.power(pred_x_mid_final-peak_x_mid_gt, 2), 0.5), np.power(np.power(pred_y_mid_final-peak_y_mid_gt, 2), 0.5)
        l2_dist_euc_final = np.power(np.power(l2_dist_x_final, 2) + np.power(l2_dist_y_final, 2), 0.5)

        l2_dist_list_append = [l2_dist_x_p_p, l2_dist_y_p_p, l2_dist_euc_p_p,
                               l2_dist_x_p_s, l2_dist_y_p_s, l2_dist_euc_p_s,
                               l2_dist_x_final, l2_dist_y_final, l2_dist_euc_final,
                               each_data_type_id_idx,
                               ]
        l2_dist_list.append(l2_dist_list_append)
        print(f'Dist {l2_dist_euc_final:.0f}, ({pred_x_mid_final},{pred_y_mid_final}), GT:({peak_x_mid_gt},{peak_y_mid_gt})')

# save metrics in a dict
metrics_dict = {}

# save l2 dist
l2_dist_array = np.array(l2_dist_list)
l2_dist_mean = np.mean(l2_dist_array, axis=0)
l2_dist_list = ['l2_dist_x_p_p', 'l2_dist_y_p_p', 'l2_dist_euc_p_p',
                'l2_dist_x_p_s', 'l2_dist_y_p_s', 'l2_dist_euc_p_s',
                'l2_dist_x_final', 'l2_dist_y_final', 'l2_dist_euc_final',
                ]
for l2_dist_idx, l2_dist_type in enumerate(l2_dist_list):
    metrics_dict[l2_dist_type] = l2_dist_mean[l2_dist_idx]

# save l2 dist (Detailed analysis)
for each_data_id, each_data_id_idx in each_data_type_id_dic.items():
    l2_dist_array_each_data_id = l2_dist_array[l2_dist_array[:, -1] == each_data_id_idx]
    sample_ratio = l2_dist_array_each_data_id.shape[0]/l2_dist_array.shape[0]*100
    l2_dist_array_each_data_id_mean = np.mean(l2_dist_array_each_data_id, axis=0)
    metrics_dict[f'l2_dist_euc ({each_data_id}) ({sample_ratio:.0f}%)'] = l2_dist_array_each_data_id_mean[2]

# save l2 dist (Histgrad analysis)
for l2_dist_idx, l2_dist_type in enumerate(l2_dist_list):
    save_figure_path = os.path.join(save_results_dir, f'{l2_dist_type}_hist.png')
    plt.figure()
    plt.hist(l2_dist_array[:, l2_dist_idx])
    plt.xlim(0, 200)
    plt.savefig(save_figure_path)

# save prediction accuracy
pred_acc_array = np.array(pred_acc_list)
co_att_gt_array = pred_acc_array[:, 0]
co_att_pred_array = pred_acc_array[:, 1]
cm = confusion_matrix(co_att_gt_array, co_att_pred_array)
plt.figure()
sns.heatmap(cm, annot=True, cmap='Blues')
save_cm_path = os.path.join(save_results_dir, 'confusion_matrix.png')
plt.savefig(save_cm_path)
metrics_dict['accuracy'] = accuracy_score(co_att_gt_array, co_att_pred_array)
metrics_dict['precision'] = precision_score(co_att_gt_array, co_att_pred_array)
metrics_dict['recall'] = recall_score(co_att_gt_array, co_att_pred_array)
metrics_dict['f1'] = f1_score(co_att_gt_array, co_att_pred_array)
metrics_dict['auc'] = roc_auc_score(co_att_gt_array, co_att_pred_array)

# save detection rate
det_rate_list = [f'Det (Thr={det_thr})' for det_thr in range(0, 110, 10)]
for det_rate_idx, det_rate_type in enumerate(det_rate_list, 1):
    det_rate = l2_dist_array[:, 2]<(det_rate_idx*10)
    det_rate_mean = np.mean(det_rate) * 100
    metrics_dict[det_rate_type] = det_rate_mean

# save metrics into json files
save_results_path = os.path.join(save_results_dir, 'eval_results.json')
with open(save_results_path, 'w') as f:
    json.dump(metrics_dict, f, indent=4)

# print metrics into a command line
print(metrics_dict)
for met_name, met_val in metrics_dict.items():
    print(met_name, met_val)