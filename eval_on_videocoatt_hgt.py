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
import seaborn as sns
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import glob
import json

# original module
from dataset.dataset_selector import dataset_generator
from models.model_selector import model_generator

# normalize heatmap
def norm_heatmap(img_heatmap):
    if np.min(img_heatmap) == np.max(img_heatmap):
        img_heatmap[:, :] = 0
    else: 
        img_heatmap = (img_heatmap - np.min(img_heatmap)) / (np.max(img_heatmap) - np.min(img_heatmap))
        img_heatmap *= 255

    return img_heatmap

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
model_attention_weight_path = os.path.join(weight_saved_dir, "model_gaussian_best.pth.tar")
model_head.load_state_dict(torch.load(model_head_weight_path,  map_location='cuda:'+str(gpus_list[0])))
model_attention.load_state_dict(torch.load(model_attention_weight_path,  map_location='cuda:'+str(gpus_list[0])))
if cuda:
    model_head = model_head.cuda(gpus_list[0])
    model_attention = model_attention.cuda(gpus_list[0])
    model_head.eval()
    model_attention.eval()

print("===> Loading dataset")
mode = cfg.exp_set.mode
cfg.data.name = 'videocoatt_no_att'
valid_set = dataset_generator(cfg, 'validate')
test_set = dataset_generator(cfg, mode)
cfg.data.name = 'videocoatt'
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
save_results_dir = os.path.join('results', cfg.data.name, model_name, 'eval_results')
if not os.path.exists(save_results_dir):
    os.makedirs(save_results_dir)

# stop_iteration = 30
stop_iteration = 10000000
print("===> Starting validation processing")
# get threshold for prediction accuracy
heatmap_peak_val_list = []
co_att_flag_gt_list = []
for iteration, batch in enumerate(valid_data_loader,1):
    print(f'{iteration}/{len(valid_data_loader)}')
    with torch.no_grad():
        # move data into gpu
        if cuda:
            for key, val in batch.items():
                if key != 'rgb_path':
                    batch[key] = Variable(val).cuda(gpus_list[0])

        # head pose estimation
        out_attention = model_attention(batch)
        out = {**out_attention, **batch}

    img_gt = out['img_gt'].to('cpu').detach()[0].numpy()
    gt_box = out['gt_box'].to('cpu').detach()[0].numpy()
    head_loc_pred = out['head_loc_pred'].to('cpu').detach()[0].numpy()
    gaze_heatmap_pred = out['gaze_heatmap_pred'].to('cpu').detach()[0].numpy()
    is_head_pred = out['is_head_pred'].to('cpu').detach()[0].numpy()
    watch_outside_pred = out['watch_outside_pred'].to('cpu').detach()[0].numpy()
    img_path = out['rgb_path'][0]

    # get final heatmap
    img = cv2.imread(img_path)
    original_height, original_width, _ = img.shape
    img_final = np.zeros((original_height, original_width))
    head_conf_thresh = 0.8
    watch_outside_conf_thresh = 0.9
    aggregate_head_cnt = 0
    for head_idx in range(is_head_pred.shape[0]):
        head_conf = is_head_pred[head_idx][-1]
        watch_outside_conf = watch_outside_pred[head_idx][-1]
        head_bbox = head_loc_pred[head_idx, :]
        gaze_map = gaze_heatmap_pred[head_idx, :]
        head_conf_flag = head_conf > head_conf_thresh
        watch_outside_flag = watch_outside_conf < watch_outside_conf_thresh
        if head_conf_flag and watch_outside_flag:
            view_size = int(gaze_map.shape[0]**0.5)
            gaze_map_view = gaze_map.reshape(view_size, view_size)
            gaze_map_resize = cv2.resize(gaze_map_view, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_final += gaze_map_resize
            aggregate_head_cnt += 1
    img_final /= aggregate_head_cnt

    # save peak values
    peak_val = np.max(img_final)
    heatmap_peak_val_list.append(peak_val)
    
    # save co att flag
    co_att_flag_gt = np.sum(gt_box, axis=(0, 1)) != 0
    co_att_flag_gt_list.append(co_att_flag_gt)

    if iteration > stop_iteration:
        break

heatmap_peak_val_array = np.array(heatmap_peak_val_list)
co_att_flag_gt = np.array(co_att_flag_gt_list)
valid_metrics_list = ['accuracy', 'precision', 'recall', 'f1']
valid_metrics_array = np.zeros((255, len(valid_metrics_list)), dtype=np.float32)
for thresh_cand in range(0, 255, 1):
    heatmap_thresh = thresh_cand / 255
    co_att_flag_pred = heatmap_peak_val_array > heatmap_thresh
    accuracy = accuracy_score(co_att_flag_gt, co_att_flag_pred)
    precision = precision_score(co_att_flag_gt, co_att_flag_pred)
    recall = recall_score(co_att_flag_gt, co_att_flag_pred)
    f1 = f1_score(co_att_flag_gt, co_att_flag_pred)
    valid_metrics_array[thresh_cand, 0] = accuracy
    valid_metrics_array[thresh_cand, 1] = precision
    valid_metrics_array[thresh_cand, 2] = recall
    valid_metrics_array[thresh_cand, 3] = f1
thresh_best_row = np.argmax(valid_metrics_array[:, 3])
thresh_best = np.argmax(valid_metrics_array[:, 3])/255

print("===> Starting test processing")
l2_dist_list = []
pred_acc_list = []
for iteration, batch in enumerate(test_data_loader,1):
    print(f'{iteration}/{len(test_data_loader)}')
    with torch.no_grad():            
        # move data into gpu
        if cuda:
            for key, val in batch.items():
                if key != 'rgb_path':
                    batch[key] = Variable(val).cuda(gpus_list[0])

        # head pose estimation
        out_attention = model_attention(batch)
        out = {**out_attention, **batch}

    img_gt = out['img_gt'].to('cpu').detach()[0].numpy()
    gt_box = out['gt_box'].to('cpu').detach()[0].numpy()
    head_loc_pred = out['head_loc_pred'].to('cpu').detach()[0].numpy()
    gaze_heatmap_pred = out['gaze_heatmap_pred'].to('cpu').detach()[0].numpy()
    is_head_pred = out['is_head_pred'].to('cpu').detach()[0].numpy()
    watch_outside_pred = out['watch_outside_pred'].to('cpu').detach()[0].numpy()
    img_path = out['rgb_path'][0]
    head_feature = out['head_feature'].to('cpu').detach()[0].numpy()

    # get final heatmap
    img = cv2.imread(img_path)
    original_height, original_width, _ = img.shape
    img_final = np.zeros((original_height, original_width))
    head_conf_thresh = 0.8
    watch_outside_conf_thresh = 0.9
    aggregate_head_cnt = 0
    for head_idx in range(is_head_pred.shape[0]):
        head_conf = is_head_pred[head_idx][-1]
        watch_outside_conf = watch_outside_pred[head_idx][-1]
        head_bbox = head_loc_pred[head_idx, :]
        gaze_map = gaze_heatmap_pred[head_idx, :]
        head_conf_flag = head_conf > head_conf_thresh
        watch_outside_flag = watch_outside_conf < watch_outside_conf_thresh
        if head_conf_flag and watch_outside_flag:
            view_size = int(gaze_map.shape[0]**0.5)
            gaze_map_view = gaze_map.reshape(view_size, view_size)
            gaze_map_resize = cv2.resize(gaze_map_view, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_final += gaze_map_resize
            aggregate_head_cnt += 1
    img_final /= aggregate_head_cnt

    # save peak values
    peak_val = np.max(img_final)
    
    # save co att flag
    co_att_flag_gt = np.sum(gt_box, axis=(0, 1)) != 0

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
        peak_x_mid_gt, peak_y_mid_gt = peak_x_mid_gt*original_width, peak_y_mid_gt*original_height
        peak_x_mid_gt, peak_y_mid_gt = map(int, [peak_x_mid_gt, peak_y_mid_gt])
        save_gt_peak = [peak_x_mid_gt, peak_y_mid_gt]
        if save_gt_peak not in gt_box_ja_list and (save_gt_peak != [0, 0]):
            gt_box_ja_list.append(save_gt_peak)
    gt_box_ja_array = np.array(gt_box_ja_list)

    co_att_flag_gt = np.sum(gt_box, axis=(0, 1)) != 0
    peak_val = np.max(img_final)
    co_att_flag_pred = peak_val > thresh_best
    pred_acc_list.append([co_att_flag_gt, co_att_flag_pred])
    if not co_att_flag_gt:
        continue

    peak_y_mid_pred, peak_x_mid_pred = np.unravel_index(np.argmax(img_final), img_final.shape)
    for gt_box_idx in range(gt_box_ja_array.shape[0]):
        peak_x_mid_gt, peak_y_mid_gt = gt_box_ja_array[gt_box_idx, :]
        l2_dist_x = np.linalg.norm(peak_x_mid_gt-peak_x_mid_pred)
        l2_dist_y = np.linalg.norm(peak_y_mid_gt-peak_y_mid_pred)
        l2_dist_euc = np.power(np.power(l2_dist_x, 2)+np.power(l2_dist_y, 2), 0.5)
        print(f'Dist {l2_dist_euc:.0f}, ({peak_x_mid_pred},{peak_y_mid_pred}), GT:({peak_x_mid_gt},{peak_y_mid_gt})')
        l2_dist_list.append([l2_dist_x, l2_dist_y, l2_dist_euc])

    if iteration > stop_iteration:
        break

# save metrics in a dict
metrics_dict = {}

# save l2 dist
l2_dist_array = np.array(l2_dist_list)
l2_dist_mean = np.mean(l2_dist_array, axis=0)
l2_dist_list = ['l2_dist_x', 'l2_dist_y', 'l2_dist_euc']
for l2_dist_idx, l2_dist_type in enumerate(l2_dist_list):
    metrics_dict[l2_dist_type] = l2_dist_mean[l2_dist_idx]

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
metrics_dict['hresh'] = thresh_best

# save detection rate
det_rate_list = [f'Det (Thr={det_thr})' for det_thr in range(0, 210, 10)]
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