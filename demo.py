# deep learning
from pyrfc3339 import generate
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image

# general module
import numpy as np
import argparse
import sys
import yaml
from addict import Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import cv2
import os
from collections import OrderedDict
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

# original module
from dataset.dataset_selector import dataset_generator
from models.model_selector import model_generator

# generate data id
def data_id_generator(img_path, cfg):
    if cfg.data.name == 'volleyball':
        video_num, seq_num, img_name = img_path.split('/')[-3:]
        img_num = img_name.split('.')[0]
        data_id = f'{video_num}_{seq_num}_{img_num}'

        return data_id
    else:
        return 'data_id_unknown'

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
saved_yaml_file_path = os.path.join(cfg_arg.exp_set.save_folder, cfg_arg.data.name, cfg_arg.exp_set.model_name, 'train.yaml')
cfg = Dict(yaml.safe_load(open(saved_yaml_file_path)))
cfg.update(cfg_arg)
print(cfg)

print("===> Building model")
model_head, model_attention, cfg = model_generator(cfg)

print("===> Building gpu configuration")
cuda = cfg.exp_set.gpu_mode
gpus_list = range(cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish+1)

print("===> Building seed configuration")
np.random.seed(cfg.exp_set.seed_num)
torch.manual_seed(cfg.exp_set.seed_num)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms=True

print("===> Load trained model")
model_name = cfg.exp_set.model_name
weight_saved_dir = os.path.join(cfg.exp_set.save_folder,cfg.data.name, model_name)
model_head_weight_path = os.path.join(weight_saved_dir, "model_head_best.pth.tar")
model_attention_weight_path = os.path.join(weight_saved_dir, "model_gaussian_best.pth.tar")
model_head.load_state_dict(torch.load(model_head_weight_path))
model_attention.load_state_dict(torch.load(model_attention_weight_path))
if cuda:
    model_head = model_head.cuda(gpus_list[0])
    model_attention = model_attention.cuda(gpus_list[0])
    model_head.eval()
    model_attention.eval()

print("===> Load dataset")
mode = cfg.exp_set.mode
test_set = dataset_generator(cfg, mode)
test_data_loader = DataLoader(dataset=test_set,
                                batch_size=cfg.exp_set.batch_size,
                                shuffle=False,
                                num_workers=cfg.exp_set.num_workers,
                                pin_memory=True)
print('{} demo samples found'.format(len(test_set)))

print("===> Make directories to save results")
if cfg.exp_set.test_gt_gaze:
    model_name = model_name + f'_use_gt_gaze'
result_save_dir = os.path.join('results', cfg.data.name, model_name)
save_image_dir = os.path.join(result_save_dir, 'demo_output')
save_image_dir_person = os.path.join(result_save_dir, 'demo_output_person')
save_image_dir_person_all_superimposed = os.path.join(result_save_dir, 'demo_output_person_all_superimposed')
save_image_dir_person_all = os.path.join(result_save_dir, 'demo_output_person_all')
save_image_dir_person_angle = os.path.join(result_save_dir, 'demo_output_person_angle')
save_image_dir_person_distance = os.path.join(result_save_dir, 'demo_output_person_distance')
save_image_dir_person_att_map = os.path.join(result_save_dir, 'demo_output_person_att_map')
save_image_dir_superimposed = os.path.join(result_save_dir, 'demo_output_superimposed')
save_image_dir_gt = os.path.join(result_save_dir, 'demo_output_gt')
save_image_dir_scene_feat = os.path.join(result_save_dir, 'demo_output_scene_feat')
save_image_dir_scene_feat_superimposed = os.path.join(result_save_dir, 'demo_output_scene_feat_superimposed')
save_image_dir_superimposed_concat = os.path.join(result_save_dir, 'demo_output_superimposed_concat')
save_image_dir_mha_weights = os.path.join(result_save_dir, 'demo_output_mha_weights')

if not os.path.exists(save_image_dir):
    os.makedirs(save_image_dir)
if not os.path.exists(save_image_dir_person):
    os.makedirs(save_image_dir_person)
if not os.path.exists(save_image_dir_person_all_superimposed):
    os.makedirs(save_image_dir_person_all_superimposed)
if not os.path.exists(save_image_dir_person_all):
    os.makedirs(save_image_dir_person_all)
if not os.path.exists(save_image_dir_person_angle):
    os.makedirs(save_image_dir_person_angle)
if not os.path.exists(save_image_dir_person_distance):
    os.makedirs(save_image_dir_person_distance)
if not os.path.exists(save_image_dir_person_att_map):
    os.makedirs(save_image_dir_person_att_map)
if not os.path.exists(save_image_dir_superimposed):
    os.makedirs(save_image_dir_superimposed)
if not os.path.exists(save_image_dir_gt):
    os.makedirs(save_image_dir_gt)
if not os.path.exists(save_image_dir_scene_feat):
    os.makedirs(save_image_dir_scene_feat)
if not os.path.exists(save_image_dir_scene_feat_superimposed):
    os.makedirs(save_image_dir_scene_feat_superimposed)
if not os.path.exists(save_image_dir_superimposed_concat):
    os.makedirs(save_image_dir_superimposed_concat)
if not os.path.exists(save_image_dir_mha_weights):
    os.makedirs(save_image_dir_mha_weights)

for iteration, batch in enumerate(test_data_loader,1):
    print(f'Iter:{iteration}')

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

        head_feature = batch['head_feature']
        if cfg.model_params.use_position:
            input_feature = head_feature.clone() 
        else:
            input_feature = head_feature.clone()
            input_feature[:, :, :2] = input_feature[:, :, :2] * 0
        batch['input_feature'] = input_feature

        # head pose estimation
        out_head = model_head(batch)
        head_vector = out_head['head_vector']
        head_enc_map = out_head['head_enc_map']
        batch['head_enc_map'] = head_enc_map

        if cfg.exp_params.use_gt_gaze:
            head_vector = batch['head_vector_gt']
        batch['head_vector'] = head_vector

        # change position inputs
        if cfg.model_params.use_gaze:
            input_gaze = head_vector.clone() 
        else:
            input_gaze = head_vector.clone() * 0
        batch['input_gaze'] = input_gaze

        out_attention = model_attention(batch)

        out = {**out_head, **out_attention, **batch}

    img_gt = out['img_gt'].to('cpu').detach()[0]
    img_pred = out['img_pred'].to('cpu').detach()[0]
    img_mid_pred = out['img_mid_pred'].to('cpu').detach()[0]
    img_mid_mean_pred = out['img_mid_mean_pred'].to('cpu').detach()[0]
    angle_dist = out['angle_dist'].to('cpu').detach()[0]
    distance_dist = out['distance_dist'].to('cpu').detach()[0]
    att_map = out['att_map'].to('cpu').detach()[0]
    saliency_img = out['saliency_img'].to('cpu').detach()[0]
    head_tensor = out['head_tensor'].to('cpu').detach()[0]
    head_vector_gt = out['head_vector_gt'].to('cpu').detach()[0]
    head_feature = out['head_feature'].to('cpu').detach()[0]
    person_atn = out['person_atn'].to('cpu').detach()[0]
    mha_weights = out['mha_weights'].to('cpu').detach()[0]
    gt_box = out['gt_box'].to('cpu').detach()[0]
    att_inside_flag = out['att_inside_flag'].to('cpu').detach()[0]
    img_path = out['rgb_path'][0]

    # define data id
    # data_type_id = search_data_type_id(head_vector_gt, head_tensor, gt_box, cfg)
    data_type_id = f''
    data_id = data_id_generator(img_path, cfg)

    if not os.path.exists(os.path.join(save_image_dir_gt, data_type_id)):
        os.makedirs(os.path.join(save_image_dir_gt, data_type_id))
    if not os.path.exists(os.path.join(save_image_dir, data_type_id)):
        os.makedirs(os.path.join(save_image_dir, data_type_id))
    if not os.path.exists(os.path.join(save_image_dir_person_all, data_type_id)):
        os.makedirs(os.path.join(save_image_dir_person_all, data_type_id))
    if not os.path.exists(os.path.join(save_image_dir_scene_feat, data_type_id)):
        os.makedirs(os.path.join(save_image_dir_scene_feat, data_type_id))
    save_image(img_gt, os.path.join(save_image_dir_gt, data_type_id, f'{mode}_{data_id}_gt.png'))
    save_image(img_pred, os.path.join(save_image_dir, data_type_id, f'{mode}_{data_id}_pred.png'))
    save_image(img_mid_mean_pred, os.path.join(save_image_dir_person_all, data_type_id, f'{mode}_{data_id}_pred.png'))
    save_image(saliency_img, os.path.join(save_image_dir_scene_feat, data_type_id, f'{mode}_{data_id}_pred.png'))

    if not os.path.exists(os.path.join(save_image_dir_person, data_type_id, f'{data_id}')):
        os.makedirs(os.path.join(save_image_dir_person, data_type_id, f'{data_id}'))
    if not os.path.exists(os.path.join(save_image_dir_person_angle, data_type_id, f'{data_id}')):
        os.makedirs(os.path.join(save_image_dir_person_angle, data_type_id, f'{data_id}'))
    if not os.path.exists(os.path.join(save_image_dir_person_distance, data_type_id, f'{data_id}')):
        os.makedirs(os.path.join(save_image_dir_person_distance, data_type_id, f'{data_id}'))
    if not os.path.exists(os.path.join(save_image_dir_person_att_map, data_type_id, f'{data_id}')):
        os.makedirs(os.path.join(save_image_dir_person_att_map, data_type_id, f'{data_id}'))

    # save attention maps
    people_num, rgb_people_trans_enc_num, rgb_feat_height, rgb_feat_width = att_map.shape
    att_map = att_map.view(rgb_people_trans_enc_num*people_num, 1, rgb_feat_height, rgb_feat_width)
    att_map = F.interpolate(att_map, (cfg.exp_set.resize_height, cfg.exp_set.resize_width), mode='nearest')
    att_map = att_map.view(people_num, rgb_people_trans_enc_num, 1, cfg.exp_set.resize_height, cfg.exp_set.resize_width)

    for person_idx in range(people_num):
        if att_inside_flag[person_idx]:
            for i in range(cfg.model_params.rgb_people_trans_enc_num):
                att_map_enc = att_map[person_idx, i, 0, :, :]
                save_image(att_map_enc, os.path.join(save_image_dir_person_att_map, data_type_id, f'{data_id}', f'{mode}_{data_id}_p{person_idx}_{i}_pred.png'))

    # save heatmaps of people
    for person_idx in range(people_num):
        img_mid_pred_person = img_mid_pred[person_idx]
        angle_dist_person = angle_dist[person_idx]
        distance_dist_person = distance_dist[person_idx]

        save_image(img_mid_pred_person, os.path.join(save_image_dir_person, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.png'))
        save_image(angle_dist_person, os.path.join(save_image_dir_person_angle, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.png'))
        save_image(distance_dist_person, os.path.join(save_image_dir_person_distance, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.png'))

    # plot multi head attention weights
    # plt.figure(figsize=(8, 6))
    # df_person = []
    # for person_idx in range(people_num):
    #     data_name = f'{person_idx}'
    #     df_person.append(data_name)

    # mha_weights = pd.DataFrame(data=mha_weights, index=df_person, columns=df_person)
    # sns.heatmap(mha_weights, cmap='jet')
    # plt.savefig(os.path.join(save_image_dir_mha_weights, f'{mode}_{data_id}_mha_weights.png'))
    # plt.close()

    # reag a rgb image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (cfg.exp_set.resize_width, cfg.exp_set.resize_height))

    # read gaze map
    img_heatmap_gaze = cv2.imread(os.path.join(save_image_dir_person_all, data_type_id, f'{mode}_{data_id}_pred.png'), cv2.IMREAD_GRAYSCALE)
    img_heatmap_gaze = cv2.resize(img_heatmap_gaze, (img.shape[1], img.shape[0]))
    img_heatmap_gaze = norm_heatmap(img_heatmap_gaze)

    # read scene feat map
    scene_feat_heatmap = cv2.imread(os.path.join(save_image_dir_scene_feat, data_type_id, f'{mode}_{data_id}_pred.png'), cv2.IMREAD_GRAYSCALE)
    scene_feat_heatmap = cv2.resize(scene_feat_heatmap, (img.shape[1], img.shape[0]))
    scene_feat_heatmap = norm_heatmap(scene_feat_heatmap)

    # read final joint attention map
    img_heatmap = cv2.imread(os.path.join(save_image_dir, data_type_id, f'{mode}_{data_id}_pred.png'), cv2.IMREAD_GRAYSCALE)
    img_heatmap = cv2.resize(img_heatmap, (img.shape[1], img.shape[0]))

    # norm img heatmaps
    img_heatmap_norm = norm_heatmap(img_heatmap)

    # overlay a heatmap to images
    img_heatmap_gaze = img_heatmap_gaze.astype(np.uint8)
    img_heatmap_gaze = cv2.applyColorMap(img_heatmap_gaze, cv2.COLORMAP_JET)
    superimposed_image_gaze = cv2.addWeighted(img, 0.5, img_heatmap_gaze, 0.5, 0)
    scene_feat_heatmap = scene_feat_heatmap.astype(np.uint8)
    scene_feat_heatmap = cv2.applyColorMap(scene_feat_heatmap, cv2.COLORMAP_JET)
    superimposed_image_scene_feat_heatmap = cv2.addWeighted(img, 0.5, scene_feat_heatmap, 0.5, 0)
    img_heatmap_norm = img_heatmap_norm.astype(np.uint8)
    img_heatmap_norm = cv2.applyColorMap(img_heatmap_norm, cv2.COLORMAP_JET)

    if cfg.model_params.dynamic_distance_type == 'gaussian':
        superimposed_image = cv2.addWeighted(img, 1.0, img_heatmap_norm, 0, 0)
    elif cfg.dynamic_distance_type == 'generator':
        superimposed_image = cv2.addWeighted(img, 0.5, img_heatmap_norm, 0.5, 0)
    else:
        superimposed_image = cv2.addWeighted(img, 0.5, img_heatmap_norm, 0.5, 0)

    # for box_idx in range(bboxes.shape[0]):
        # peak_x_min_pred, peak_y_min_pred, peak_x_max_pred, peak_y_max_pred = bboxes[box_idx]
        # peak_x_min_pred, peak_y_min_pred, peak_x_max_pred, peak_y_max_pred = map(int, [peak_x_min_pred, peak_y_min_pred, peak_x_max_pred, peak_y_max_pred])
        # cv2.rectangle(superimposed_image_gaze, (peak_x_min_pred, peak_y_min_pred), (peak_x_max_pred, peak_y_max_pred), (255, 0, 0), thickness=4)
        # cv2.rectangle(superimposed_image_scene_feat_heatmap, (peak_x_min_pred, peak_y_min_pred), (peak_x_max_pred, peak_y_max_pred), (255, 0, 0), thickness=4)
        # cv2.rectangle(superimposed_image, (peak_x_min_pred, peak_y_min_pred), (peak_x_max_pred, peak_y_max_pred), (255, 0, 0), thickness=4)
        # cv2.circle(superimposed_image_gaze, (peak_x_min_pred, peak_y_min_pred), 10, (0, 255 ,0), thickness=-1)
        # cv2.circle(superimposed_image_scene_feat_heatmap, (peak_x_min_pred, peak_y_min_pred), 10, (0, 255 ,0), thickness=-1)
        # cv2.circle(superimposed_image, (peak_x_min_pred, peak_y_min_pred), 10, (0, 255 ,0), thickness=-1)

    # calc distances for each co att box
    gt_box_num = torch.sum(att_inside_flag)
    print(f'GT bbox:{gt_box_num}, Max:{np.max(img_heatmap):.1f}, Min:{np.min(img_heatmap):.1f}')

    for gt_box_idx in range(gt_box_num):
        # calc a center of gt bbox
        peak_x_min_gt, peak_y_min_gt, peak_x_max_gt, peak_y_max_gt = gt_box[gt_box_idx]
        peak_x_mid_gt, peak_y_mid_gt = (peak_x_min_gt+peak_x_max_gt)//2, (peak_y_min_gt+peak_y_max_gt)//2

        # calc centers of pred bboxes
        if cfg.model_params.dynamic_distance_type == 'gaussian':
            peak_x_mid_pred_all, peak_y_mid_pred_all = head_tensor[:gt_box_num, 3], head_tensor[:gt_box_num, 4]
            peak_x_mid_pred, peak_y_mid_pred = torch.mean(peak_x_mid_pred_all), torch.mean(peak_y_mid_pred_all) 
            peak_x_mid_pred, peak_y_mid_pred = int(peak_x_mid_pred*cfg.exp_set.resize_width), int(peak_y_mid_pred*cfg.exp_set.resize_height)
        else:
            peak_y_mid_pred, peak_x_mid_pred = np.unravel_index(np.argmax(img_heatmap), img_heatmap.shape)

        peak_y_mid_pred, peak_x_mid_pred = np.unravel_index(np.argmax(img_heatmap), img_heatmap.shape)
        peak_x_min_pred, peak_x_max_pred = peak_x_mid_pred-20, peak_x_mid_pred+20
        peak_y_min_pred, peak_y_max_pred = peak_y_mid_pred-20, peak_y_mid_pred+20

        peak_x_diff, peak_y_diff = peak_x_mid_pred-peak_x_mid_gt, peak_y_mid_pred-peak_y_mid_gt
        diff_dis = np.power(np.power(peak_x_diff, 2) + np.power(peak_y_diff, 2), 0.5)
        cv2.circle(superimposed_image, (peak_x_mid_pred, peak_y_mid_pred), 10, (128, 0, 128), thickness=-1)

        print(f'[Bbox] Dis={diff_dis:.0f}, GT=({peak_x_mid_gt:.0f},{peak_y_mid_gt:.0f}), peak=({peak_x_mid_pred:.0f},{peak_y_mid_pred:.0f})')

        # transform float to int
        peak_x_min_gt, peak_y_min_gt, peak_x_max_gt, peak_y_max_gt  = map(int, [peak_x_min_gt, peak_y_min_gt, peak_x_max_gt, peak_y_max_gt])
        peak_x_mid_gt, peak_y_mid_gt = (peak_x_min_gt+peak_x_max_gt)//2, (peak_y_min_gt+peak_y_max_gt)//2        
        cv2.circle(superimposed_image, (peak_x_mid_gt, peak_y_mid_gt), 10, (0, 255, 0), thickness=-1)

        # plot gt box and pred box
        # cv2.rectangle(superimposed_image_gaze, (peak_x_min_gt, peak_y_min_gt), (peak_x_max_gt, peak_y_max_gt), (0, 255, 0), thickness=4)
        # cv2.rectangle(superimposed_image_gaze, (peak_x_min_pred, peak_y_min_pred), (peak_x_max_pred, peak_y_max_pred), (255, 0, 0), thickness=4)
        # cv2.rectangle(superimposed_image_scene_feat_heatmap, (peak_x_min_gt, peak_y_min_gt), (peak_x_max_gt, peak_y_max_gt), (0, 255, 0), thickness=4)
        # cv2.rectangle(superimposed_image_scene_feat_heatmap, (peak_x_min_pred, peak_y_min_pred), (peak_x_max_pred, peak_y_max_pred), (255, 0, 0), thickness=4)
        # cv2.rectangle(superimposed_image, (peak_x_min_gt, peak_y_min_gt), (peak_x_max_gt, peak_y_max_gt), (0, 255, 0), thickness=4)
        # cv2.rectangle(superimposed_image, (peak_x_min_pred, peak_y_min_pred), (peak_x_max_pred, peak_y_max_pred), (255, 0, 0), thickness=4)
        # cv2.circle(superimposed_image_gaze, (peak_x_min_gt, peak_y_min_gt), 10, (128, 0, 128), thickness=-1)
        # cv2.circle(superimposed_image_gaze, (peak_x_min_pred, peak_y_min_pred), 10, (0, 255 ,0), thickness=-1)
        # cv2.circle(superimposed_image_scene_feat_heatmap, (peak_x_min_gt, peak_y_min_gt), 10, (128, 0, 128), thickness=-1)
        # cv2.circle(superimposed_image_scene_feat_heatmap, (peak_x_min_pred, peak_y_min_pred), 10, (0, 255 ,0), thickness=-1)
        # cv2.circle(superimposed_image, (peak_x_min_gt, peak_y_min_gt), 10, (128, 0, 128), thickness=-1)
        # cv2.circle(superimposed_image, (peak_x_min_pred, peak_y_min_pred), 10, (0, 255 ,0), thickness=-1)

    for person_idx in range(people_num):
        if att_inside_flag[person_idx]:
            for i in range(cfg.model_params.rgb_people_trans_enc_num):
                att_map_enc = cv2.imread(os.path.join(save_image_dir_person_att_map, data_type_id, f'{data_id}', f'{mode}_{data_id}_p{person_idx}_{i}_pred.png'), cv2.IMREAD_GRAYSCALE)
                att_map_enc = att_map_enc.astype(np.uint8)
                att_map_enc = cv2.applyColorMap(cv2.resize(att_map_enc, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)
                superimposed_att_map_enc = cv2.addWeighted(img, 0.5, att_map_enc, 0.5, 0)
                cv2.imwrite(os.path.join(save_image_dir_person_att_map, data_type_id, f'{data_id}', f'{mode}_{data_id}_p{person_idx}_{i}_pred.png'), superimposed_att_map_enc)

            head_tensor_person = head_tensor[person_idx]
            head_feature_person = head_feature[person_idx]
            person_atn_weight = float(person_atn[person_idx])

            head_x, head_y = head_feature_person[0:2]
            head_x, head_y = int(head_x*cfg.exp_set.resize_width), int(head_y*cfg.exp_set.resize_height)

            head_vec_x, head_vec_y = head_tensor_person[0:2]
            scale_factor = 30
            pred_x = int(head_vec_x*scale_factor + head_x)
            pred_y = int(head_vec_y*scale_factor + head_y)

            # plot predict gaze direction (Red)
            atn_weight_set = (0, 0, 255)
            # arrow_set = (255, 255, 255)
            arrow_set = (0, 0, 0)

            superimposed_image= cv2.arrowedLine(superimposed_image, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
            superimposed_image_gaze= cv2.arrowedLine(superimposed_image_gaze, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)

            img_heatmap_person = cv2.imread(os.path.join(save_image_dir_person, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.png'), cv2.IMREAD_GRAYSCALE)
            img_heatmap_angle = cv2.imread(os.path.join(save_image_dir_person_angle, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.png'), cv2.IMREAD_GRAYSCALE)
            img_heatmap_distance = cv2.imread(os.path.join(save_image_dir_person_distance, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.png'), cv2.IMREAD_GRAYSCALE)

            img_heatmap_person = norm_heatmap(img_heatmap_person)

            img_heatmap_person = img_heatmap_person.astype(np.uint8)
            img_heatmap_angle = img_heatmap_angle.astype(np.uint8)
            img_heatmap_distance = img_heatmap_distance.astype(np.uint8)

            img_heatmap_person = cv2.applyColorMap(cv2.resize(img_heatmap_person, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)
            img_heatmap_angle = cv2.applyColorMap(cv2.resize(img_heatmap_angle, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)
            img_heatmap_distance = cv2.applyColorMap(cv2.resize(img_heatmap_distance, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)

            if cfg.dynamic_distance_type == 'gaussian':
                superimposed_image_person = cv2.addWeighted(img, 1.0, img_heatmap_person, 0, 0)
            elif cfg.dynamic_distance_type == 'generator':
                superimposed_image_person = cv2.addWeighted(img, 0.5, img_heatmap_person, 0.5, 0)
            else:
                superimposed_image_person = cv2.addWeighted(img, 0.5, img_heatmap_person, 0.5, 0)

            # superimposed_image_person = cv2.addWeighted(img, 0.5, img_heatmap_person, 0.5, 0)
            superimposed_image_angle = cv2.addWeighted(img, 0.5, img_heatmap_angle, 0.5, 0)
            superimposed_image_distance = cv2.addWeighted(img, 0.5, img_heatmap_distance, 0.5, 0)

            # cv2.rectangle(superimposed_image_person, (x_min_ball, y_min_ball), (x_max_ball, y_max_ball), (0, 255, 0), thickness=4)
            # cv2.rectangle(superimposed_image_angle, (x_min_ball, y_min_ball), (x_max_ball, y_max_ball), (0, 255, 0), thickness=4)
            # cv2.rectangle(superimposed_image_distance, (x_min_ball, y_min_ball), (x_max_ball, y_max_ball), (0, 255, 0), thickness=4)

            head_tensor_person = head_tensor[person_idx]
            head_feature_person = head_feature[person_idx]
            person_atn_weight = float(person_atn[person_idx])

            head_x, head_y = head_feature_person[0:2]
            head_x, head_y = int(head_x*cfg.exp_set.resize_width), int(head_y*cfg.exp_set.resize_height)

            head_vec_x, head_vec_y = head_tensor_person[0:2]
            scale_factor = 30
            pred_x = int(head_vec_x*scale_factor + head_x)
            pred_y = int(head_vec_y*scale_factor + head_y)

            # plot predict gaze direction (Red)
            atn_weight_set = (0, 0, 255)
            # arrow_set = (255, 255, 255)
            arrow_set = (0, 0, 0)

            superimposed_image_person = cv2.arrowedLine(superimposed_image_person, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
            superimposed_image_angle = cv2.arrowedLine(superimposed_image_angle, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
            superimposed_image_distance = cv2.arrowedLine(superimposed_image_distance, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
            superimposed_image= cv2.arrowedLine(superimposed_image, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
            superimposed_image_gaze= cv2.arrowedLine(superimposed_image_gaze, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
            peak_x_mid_pred, peak_y_mid_pred = int(peak_x_mid_pred_all[person_idx]*cfg.exp_set.resize_width), int(peak_y_mid_pred_all[person_idx]*cfg.exp_set.resize_height)
            cv2.circle(superimposed_image_person, (peak_x_mid_pred, peak_y_mid_pred), 10, (128, 0, 128), thickness=-1)

            # plot person weight
            # cv2.putText(superimposed_image_person, f'{person_atn_weight:.2f}', (30, 30), cv2.FONT_HERSHEY_PLAIN, 3, atn_weight_set, 3, cv2.LINE_AA)
            # cv2.putText(superimposed_image_angle, f'{person_atn_weight:.2f}', (30, 30), cv2.FONT_HERSHEY_PLAIN, 3, atn_weight_set, 3, cv2.LINE_AA)
            # cv2.putText(superimposed_image_distance, f'{person_atn_weight:.2f}', (30, 30), cv2.FONT_HERSHEY_PLAIN, 3, atn_weight_set, 3, cv2.LINE_AA)
            # cv2.putText(superimposed_image, f'{person_atn_weight:.2f}', (head_x-10, head_y-10), cv2.FONT_HERSHEY_PLAIN, 2, atn_weight_set, 2, cv2.LINE_AA)

            save_txt_angle = os.path.join(save_image_dir_person_angle, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.txt')
            save_txt_distance = os.path.join(save_image_dir_person_distance, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.txt')

            if head_tensor_person.shape[0] > 10:
                theta_mean, dis_mean_x, dis_mean_y, theta_sig, dis_sig_00, dis_sig_10, dis_sig_01, dis_sig_11 = head_tensor_person[3:11]
                gaze_x, gaze_y, gaze_var = head_tensor_person[:3]
            elif head_tensor_person.shape[0] == 7:
                theta_mean, theta_sig = 0, 0
                dis_mean_x, dis_mean_y, dis_sig_x, dis_sig_y = head_tensor_person[3:]
                gaze_x, gaze_y = head_tensor_person[:2]
            else:
                theta_mean, dis_mean_x, dis_mean_y, theta_sig, dis_sig_x, dis_sig_y = head_tensor_person[2:8]
                gaze_x, gaze_y = head_tensor_person[:2]

            # print(f'Person:{person_idx}')
            # with open(save_txt_angle, 'w') as f:
            #     f.write(f'Mean:{theta_mean}\n')
            #     f.write(f'Sigma:{theta_sig}\n')
            #     if head_tensor_person.shape[0] > 10:
            #         f.write(f'Gaze var:{gaze_var}\n')

            # with open(save_txt_distance, 'w') as f:
            #     f.write(f'Mean x:{dis_mean_x}\n')
            #     f.write(f'Mean y:{dis_mean_y}\n')
            #     f.write(f'Sigma x:{dis_sig_00}\n')
            #     f.write(f'Sigma y:{dis_sig_11}\n')
            #     if head_tensor_person.shape[0] > 10:
            #         f.write(f'Gaze var:{gaze_var}\n')

            #     cv2.putText(superimposed_image_person, f'{iar_label}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 3, atn_weight_set, 3, cv2.LINE_AA)
            #     cv2.putText(superimposed_image_angle, f'{iar_label}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 3, atn_weight_set, 3, cv2.LINE_AA)
            #     cv2.putText(superimposed_image_distance, f'{iar_label}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 3, atn_weight_set, 3, cv2.LINE_AA)
            #     cv2.putText(superimposed_image, f'{person_idx}:{iar_label}', (head_x-10, head_y-20), cv2.FONT_HERSHEY_PLAIN, 2, atn_weight_set, 2, cv2.LINE_AA)

            # save images
            cv2.imwrite(os.path.join(save_image_dir_person, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.png'), superimposed_image_person)
            cv2.imwrite(os.path.join(save_image_dir_person_angle, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.png'), superimposed_image_angle)
            cv2.imwrite(os.path.join(save_image_dir_person_distance, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_pred.png'), superimposed_image_distance)

    # save image
    if not os.path.exists(os.path.join(save_image_dir_person_all_superimposed, data_type_id)):
        os.makedirs(os.path.join(save_image_dir_person_all_superimposed, data_type_id))
    if not os.path.exists(os.path.join(save_image_dir_scene_feat_superimposed, data_type_id)):
        os.makedirs(os.path.join(save_image_dir_scene_feat_superimposed, data_type_id))
    if not os.path.exists(os.path.join(save_image_dir_superimposed, data_type_id)):
        os.makedirs(os.path.join(save_image_dir_superimposed, data_type_id))
    cv2.imwrite(os.path.join(save_image_dir_superimposed, data_type_id, f'{mode}_{data_id}_superimposed.png'), superimposed_image)
    cv2.imwrite(os.path.join(save_image_dir_person_all_superimposed, data_type_id, f'{mode}_{data_id}_superimposed.png'), superimposed_image_gaze)
    cv2.imwrite(os.path.join(save_image_dir_scene_feat_superimposed, data_type_id, f'{mode}_{data_id}_superimposed.png'), superimposed_image_scene_feat_heatmap)

    # concat all images
    fig = plt.figure(figsize=(6, 4))
    not_pad_num = torch.sum(att_inside_flag)
    columns, rows = max(3, not_pad_num), 4
    
    if cfg.model_params.people_feat_aggregation_type == 'max_pool':
        pass
    else:
        for person_idx in range(1, not_pad_num+1, 1):
            angle_img_path = os.path.join(save_image_dir_person_angle, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx-1}_pred.png')
            distance_img_path = os.path.join(save_image_dir_person_distance, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx-1}_pred.png')
            person_img_path = os.path.join(save_image_dir_person, data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx-1}_pred.png')
            fig.add_subplot(rows, columns, person_idx)
            plt.imshow(cv2.cvtColor(cv2.imread(angle_img_path), cv2.COLOR_BGR2RGB))
            plt.xticks(color="None")
            plt.yticks(color="None")
            fig.add_subplot(rows, columns, person_idx+columns)
            plt.imshow(cv2.cvtColor(cv2.imread(distance_img_path), cv2.COLOR_BGR2RGB))
            plt.xticks(color="None")
            plt.yticks(color="None")
            fig.add_subplot(rows, columns, person_idx+columns*2)
            plt.imshow(cv2.cvtColor(cv2.imread(person_img_path), cv2.COLOR_BGR2RGB))
            plt.xticks(color="None")
            plt.yticks(color="None")

    ## overall images
    person_all_img_path = os.path.join(save_image_dir_person_all_superimposed, data_type_id, f'{mode}_{data_id}_superimposed.png')
    scene_feat_img_path = os.path.join(save_image_dir_scene_feat_superimposed, data_type_id, f'{mode}_{data_id}_superimposed.png')
    joint_attention_img_path = os.path.join(save_image_dir_superimposed, data_type_id, f'{mode}_{data_id}_superimposed.png')
    fig.add_subplot(rows, columns, 1+columns*(rows-1))
    plt.imshow(cv2.cvtColor(cv2.imread(person_all_img_path), cv2.COLOR_BGR2RGB))
    plt.xticks(color="None")
    plt.yticks(color="None")
    fig.add_subplot(rows, columns, 2+columns*(rows-1))
    plt.imshow(cv2.cvtColor(cv2.imread(scene_feat_img_path), cv2.COLOR_BGR2RGB))
    plt.xticks(color="None")
    plt.yticks(color="None")
    fig.add_subplot(rows, columns, 3+columns*(rows-1))
    plt.imshow(cv2.cvtColor(cv2.imread(joint_attention_img_path), cv2.COLOR_BGR2RGB))
    plt.xticks(color="None")
    plt.yticks(color="None")
    if not os.path.exists(os.path.join(save_image_dir_superimposed_concat, data_type_id)):
        os.makedirs(os.path.join(save_image_dir_superimposed_concat, data_type_id))
    plt.savefig(os.path.join(save_image_dir_superimposed_concat, data_type_id, f'{mode}_{data_id}_superimposed.png'))