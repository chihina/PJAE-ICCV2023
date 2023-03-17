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

# original module
from dataset.dataset_selector import dataset_generator
from models.model_selector import model_generator

# generate data type
def data_type_id_generator(head_vector_gt, head_tensor, gt_box, cfg):
    data_type_id = ''

    if cfg.data.name == 'volleyball':
        data_type_id = f'bbox_{cfg.exp_params.bbox_types}_gaze_{cfg.exp_params.gaze_types}_act_{cfg.exp_params.action_types}'
    elif cfg.data.name == 'videocoatt':
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
        gt_x_size /= cfg.exp_set.resize_width
        gt_y_size /= cfg.exp_set.resize_height
        gt_size = ((gt_x_size**2)+(gt_y_size**2))**0.5
        if gt_size < 0.1:
            gt_size_id = '0_0<size<0_1'
        else:
            gt_size_id = '0_1<size'

        # data_type_id = f'{dets_people_id}:{gaze_error_id}:{gt_size_id}'
        # data_type_id = f'{dets_people_id}:{gaze_error_id}'
        data_type_id = ''

    return data_type_id

# generate data id
def data_id_generator(img_path, cfg):
    data_id = 'unknown'
    if cfg.data.name == 'volleyball':
        video_num, seq_num, img_name = img_path.split('/')[-3:]
        img_num = img_name.split('.')[0]
        data_id = f'{video_num}_{seq_num}_{img_num}'
    elif cfg.data.name == 'videocoatt':
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

    return data_id

# normalize heatmap
def norm_heatmap(img_heatmap):
    if np.min(img_heatmap) == np.max(img_heatmap):
        img_heatmap[:, :] = 0
    else: 
        img_heatmap = (img_heatmap - np.min(img_heatmap)) / (np.max(img_heatmap) - np.min(img_heatmap))
        img_heatmap *= 255

    return img_heatmap

def get_edge_boxes(im):
    model = os.path.join('saved_weights', 'videocoatt', 'model.yml.gz')
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(50)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    return boxes

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

print("===> Loading trained model")
model_name = cfg.exp_set.model_name
weight_saved_dir = os.path.join(cfg.exp_set.save_folder,cfg.data.name, model_name)
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
if cfg.exp_set.test_gt_gaze:
    model_name = model_name + f'_use_gt_gaze'

save_image_dir_dic = {}
save_image_dir_list = ['joint_attention', 'joint_attention_superimposed',
                       'attention_fan', 'attention_fan_pool', 'attention_fan_pool_superimposed',
                       'saliency_map', 'saliency_map_superimposed',
                       'gt_map'
                       ]
for dir_name in save_image_dir_list:
    save_image_dir_dic[dir_name] = os.path.join('results', cfg.data.name, model_name, dir_name)
    if not os.path.exists(save_image_dir_dic[dir_name]):
        os.makedirs(save_image_dir_dic[dir_name])

print("===> Starting demo processing")
stop_iteration = 20
if mode == 'test':
    stop_iteration = 200
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

        out_attention = model_attention(batch)
        # loss_set_head = model_head.calc_loss(batch, out_head)
        # loss_set_attention = model_attention.calc_loss(batch, out_attention, cfg)

        out = {**out_head, **out_attention, **batch}

    img_gt = out['img_gt'].to('cpu').detach()[0]

    img_pred = out['img_pred'].to('cpu').detach()[0]
    angle_dist = out['angle_dist'].to('cpu').detach()[0]
    angle_dist_pool = out['angle_dist_pool'].to('cpu').detach()[0]
    saliency_map = out['saliency_map'].to('cpu').detach()[0]
    gt_box = out['gt_box'].to('cpu').detach()[0]
    head_tensor = out['head_tensor'].to('cpu').detach()[0].numpy()
    head_feature = out['head_feature'].to('cpu').detach()[0].numpy()
    head_vector_gt = out['head_vector_gt'].to('cpu').detach()[0].numpy()
    att_inside_flag = out['att_inside_flag'].to('cpu').detach()[0]
    img_path = out['rgb_path'][0]

    # redefine image size
    img = cv2.imread(img_path)
    original_height, original_width, _ = img.shape

    # define data id
    data_type_id = data_type_id_generator(head_vector_gt, head_tensor, gt_box, cfg)
    data_id = data_id_generator(img_path, cfg)
    print(f'Iter:{iteration}, {data_id}, {data_type_id}')

    # expand directories
    single_image_dir = ['joint_attention', 'joint_attention_superimposed',
                        'attention_fan_pool', 'attention_fan_pool_superimposed',
                        'saliency_map', 'saliency_map_superimposed',
                        ]
    for dir_name in single_image_dir:
        if not os.path.exists(os.path.join(save_image_dir_dic[dir_name], data_type_id)):
            os.makedirs(os.path.join(save_image_dir_dic[dir_name], data_type_id))
    for dir_name in ['attention_fan', 'gt_map']:
        if not os.path.exists(os.path.join(save_image_dir_dic[dir_name], data_type_id, f'{data_id}')):
            os.makedirs(os.path.join(save_image_dir_dic[dir_name], data_type_id, f'{data_id}'))

    # save joint attention estimation
    save_image(img_pred, os.path.join(save_image_dir_dic['joint_attention'], data_type_id, f'{mode}_{data_id}_joint_attention.png'))
    save_image(saliency_map, os.path.join(save_image_dir_dic['saliency_map'], data_type_id, f'{mode}_{data_id}_joint_attention.png'))
    save_image(angle_dist_pool, os.path.join(save_image_dir_dic['attention_fan_pool'], data_type_id, f'{mode}_{data_id}_joint_attention.png'))

    # save joint attention estimation as a superimposed image
    img = cv2.resize(img, (original_width, original_height))
    
    img_heatmap = cv2.imread(os.path.join(save_image_dir_dic['joint_attention'], data_type_id, f'{mode}_{data_id}_joint_attention.png'), cv2.IMREAD_GRAYSCALE)
    attention_fan_pool = cv2.imread(os.path.join(save_image_dir_dic['attention_fan_pool'], data_type_id, f'{mode}_{data_id}_joint_attention.png'), cv2.IMREAD_GRAYSCALE)
    saliency_map = cv2.imread(os.path.join(save_image_dir_dic['saliency_map'], data_type_id, f'{mode}_{data_id}_joint_attention.png'), cv2.IMREAD_GRAYSCALE)

    img_heatmap = cv2.resize(img_heatmap, (img.shape[1], img.shape[0]))
    attention_fan_pool = cv2.resize(attention_fan_pool, (img.shape[1], img.shape[0]))
    saliency_map = cv2.resize(saliency_map, (img.shape[1], img.shape[0]))

    img_heatmap_norm = norm_heatmap(img_heatmap)
    attention_fan_pool = norm_heatmap(attention_fan_pool)
    saliency_map = norm_heatmap(saliency_map)

    img_heatmap_norm = img_heatmap_norm.astype(np.uint8)
    attention_fan_pool = attention_fan_pool.astype(np.uint8)
    saliency_map = saliency_map.astype(np.uint8)
    
    # get estimated joint attention coordinates
    y_mid_pred, x_mid_pred = np.unravel_index(np.argmax(img_heatmap_norm), img_heatmap_norm.shape)

    img_heatmap_norm = cv2.applyColorMap(img_heatmap_norm, cv2.COLORMAP_JET)
    attention_fan_pool = cv2.applyColorMap(attention_fan_pool, cv2.COLORMAP_JET)
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

    superimposed_image = cv2.addWeighted(img, 0.5, img_heatmap_norm, 0.5, 0)
    superimposed_fan = cv2.addWeighted(img, 0.5, attention_fan_pool, 0.5, 0)
    superimposed_sal = cv2.addWeighted(img, 0.5, saliency_map, 0.5, 0)

    for person_idx in range(angle_dist.shape[0]):
        save_image(angle_dist[person_idx], os.path.join(save_image_dir_dic['attention_fan'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_fan.png'))
        save_image(img_gt[person_idx], os.path.join(save_image_dir_dic['gt_map'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_gt.png'))

    # get region proposals
    # edge_boxes = get_edge_boxes(img)
    # edge_boxes_score = np.zeros((len(edge_boxes[0]), 4+1))
    # for idx in range(len(edge_boxes[0])):
    #     (x, y, w, h), conf = edge_boxes[0][idx], edge_boxes[1][idx]
    #     edge_boxes_score[idx, :4] = np.array([x, y, x+w, y+h])
    #     atn_score = np.mean(img_heatmap[y:y+h, x:x+w])
    #     edge_boxes_score[idx, 4] = atn_score
    # score_max_idx = np.argmax(edge_boxes_score[:, 4])
    # pred_bbox = edge_boxes_score[score_max_idx, :4]
    # x_min_pred, y_min_pred, x_max_pred, y_max_pred = map(int, pred_bbox)
    # x_mid_pred, y_mid_pred = (x_min_pred+x_max_pred)//2, (y_min_pred+y_max_pred)//2

    # calc distances for each co att box
    gt_box = gt_box.numpy()
    gt_box_num = np.sum(np.sum(gt_box, axis=1)!=0)
    print(gt_box_num)
    for gt_box_idx in range(gt_box_num):
        x_min_gt, y_min_gt, x_max_gt, y_max_gt = map(float, gt_box[gt_box_idx])
        x_min_gt, x_max_gt = map(lambda x: int(x*original_width), [x_min_gt, x_max_gt])
        y_min_gt, y_max_gt = map(lambda x: int(x*original_height), [y_min_gt, y_max_gt])
        x_mid_gt, y_mid_gt = (x_min_gt+x_max_gt)//2, (y_min_gt+y_max_gt)//2
        break
    l2_dist = ((x_mid_gt-x_mid_pred)**2+(y_mid_gt-y_mid_pred)**2)**0.5
    # print(l2_dist)

    # plot estimated and groung-truth joint attentions
    cv2.circle(superimposed_image, (x_mid_pred, y_mid_pred), 10, (0, 165, 255), thickness=-1)
    cv2.circle(superimposed_image, (int(x_mid_gt), int(y_mid_gt)), 10, (0, 255, 0), thickness=-1)
    # cv2.circle(superimposed_image, (x_mid_pred, y_mid_pred), 10, (128, 0, 128), thickness=-1)
    # cv2.circle(superimposed_image, (x_mid_gt, y_mid_gt), 10, (0, 255, 0), thickness=-1)
    # cv2.rectangle(superimposed_image, (x_min_pred, y_min_pred), (x_max_pred, y_max_pred), (128, 0, 128), thickness=1)
    # cv2.rectangle(superimposed_image, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 255, 0), thickness=1)

    cv2.imwrite(os.path.join(save_image_dir_dic['joint_attention_superimposed'], data_type_id, f'{mode}_{data_id}_superimposed.png'), superimposed_image)
    cv2.imwrite(os.path.join(save_image_dir_dic['attention_fan_pool_superimposed'], data_type_id, f'{mode}_{data_id}_superimposed.png'), superimposed_fan)
    cv2.imwrite(os.path.join(save_image_dir_dic['saliency_map_superimposed'], data_type_id, f'{mode}_{data_id}_superimposed.png'), superimposed_sal)