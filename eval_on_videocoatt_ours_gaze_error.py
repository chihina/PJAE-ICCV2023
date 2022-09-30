# deep learning
from builtins import iter
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
cfg.data.name = 'videocoatt'
valid_set = dataset_generator(cfg, 'validate')
cfg.data.name = 'videocoatt'
test_set = dataset_generator(cfg, mode)
valid_data_loader = DataLoader(dataset=valid_set,
                                batch_size=cfg.exp_set.batch_size,
                                shuffle=False,
                                num_workers=cfg.exp_set.num_workers,
                                pin_memory=True)
print('{} demo samples found'.format(len(valid_set)))
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

stop_iteration = 100
# stop_iteration = 1000000
print("===> Starting test processing")
gaze_cos_sim_list = []
each_data_type_id_dic = {}
for iteration, batch in enumerate(test_data_loader):
    print(f'{iteration}/{len(test_data_loader)}')

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

        out = {**out_head, **batch}

    head_feature = out['head_feature'].to('cpu').detach()[0].numpy()
    gt_box = out['gt_box'].to('cpu').detach()[0].numpy()
    att_inside_flag = out['att_inside_flag'].to('cpu').detach()[0].numpy()
    head_vector_gt = out['head_vector_gt'].to('cpu').detach()[0].numpy()
    head_vector = out['head_vector'].to('cpu').detach()[0].numpy()

    # generate each data id
    # each_data_type_id = ''
    each_data_type_id = each_data_type_id_generator(head_vector_gt, head_vector, gt_box, cfg)
    if not each_data_type_id in each_data_type_id_dic.keys():
        each_data_type_id_dic[each_data_type_id] = len(each_data_type_id_dic.keys())
    each_data_type_id_idx = each_data_type_id_dic[each_data_type_id]

    # get a padding number
    people_padding_mask = (np.sum(head_feature, axis=-1) != 0)
    people_padding_num = np.sum(people_padding_mask)
    if people_padding_num == 0:
        continue
    
    gaze_cos_sim = np.sum(head_vector * head_vector_gt, axis=-1)
    gaze_cos_sim_masked = gaze_cos_sim * att_inside_flag
    gaze_cos_sim_mean = np.sum(gaze_cos_sim_masked) / np.sum(people_padding_mask)
    gaze_cos_sim_list.append([gaze_cos_sim_mean, each_data_type_id_idx])
    
    # if iteration > stop_iteration:
        # break

# save metrics in a dict
metrics_dict = {}

# save l2 dist
gaze_error_array = np.array(gaze_cos_sim_list)
gaze_error_mean = np.mean(gaze_error_array, axis=0)
gaze_error_type_list = ['gaze_error_euc']
for gaze_error_idx, gaze_error_type in enumerate(gaze_error_type_list):
    metrics_dict[gaze_error_type] = gaze_error_mean[gaze_error_idx]

# save metrics into json files
save_results_path = os.path.join(save_results_dir, 'eval_results_gaze_cos_sim.json')
with open(save_results_path, 'w') as f:
    json.dump(metrics_dict, f, indent=4)

# print metrics into a command line
print(metrics_dict)
for met_name, met_val in metrics_dict.items():
    print(met_name, met_val)