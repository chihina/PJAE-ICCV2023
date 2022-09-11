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
save_image_dir_list = ['attention', 'attention_superimposed', 'gt_map']
for dir_name in save_image_dir_list:
    save_image_dir_dic[dir_name] = os.path.join('results', cfg.data.name, model_name, dir_name)
    if not os.path.exists(save_image_dir_dic[dir_name]):
        os.makedirs(save_image_dir_dic[dir_name])

print("===> Starting demo processing")
stop_iteration = 20
if mode == 'test':
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

        out_attention = model_attention(batch)
        # loss_set_head = model_head.calc_loss(batch, out_head)
        # loss_set_attention = model_attention.calc_loss(batch, out_attention, cfg)

        out = {**out_head, **out_attention, **batch}

    img_gt = out['img_gt'].to('cpu').detach()[0]
    gt_box = out['gt_box'].to('cpu').detach()[0]
    head_loc_pred = out['head_loc_pred'].to('cpu').detach()[0]
    gaze_heatmap_pred = out['gaze_heatmap_pred'].to('cpu').detach()[0]
    is_head_pred = out['is_head_pred'].to('cpu').detach()[0]
    watch_outside_pred = out['watch_outside_pred'].to('cpu').detach()[0]
    img_path = out['rgb_path'][0]

    # redefine image size
    img = cv2.imread(img_path)
    original_height, original_width, _ = img.shape

    # define data id
    data_type_id = ''
    data_id = data_id_generator(img_path, cfg)
    print(f'Iter:{iteration}, {data_id}, {data_type_id}')

    # expand directories
    # for dir_name in ['joint_attention']:
        # if not os.path.exists(os.path.join(save_image_dir_dic[dir_name], data_type_id)):
            # os.makedirs(os.path.join(save_image_dir_dic[dir_name], data_type_id))
    # save_image(img_pred, os.path.join(save_image_dir_dic['joint_attention'], data_type_id, f'{mode}_{data_id}_joint_attention.png'))

    for dir_name in ['attention', 'gt_map']:
        if not os.path.exists(os.path.join(save_image_dir_dic[dir_name], data_type_id, f'{data_id}')):
            os.makedirs(os.path.join(save_image_dir_dic[dir_name], data_type_id, f'{data_id}'))

    img = cv2.resize(img, (original_width, original_height))
    head_query_num = is_head_pred.shape[0]
    head_conf_thresh = 0.7
    for head_idx in range(head_query_num):
        head_conf = is_head_pred[head_idx][0]
        watch_outside_conf = watch_outside_pred[head_idx][0]
        head_bbox = head_loc_pred[head_idx, :]
        gaze_map = gaze_heatmap_pred[head_idx, :]
        
        if head_conf > head_conf_thresh:
            print(f'Idx:{head_idx}, Head conf:{head_conf:.2f}, Watch conf:{watch_outside_conf:.2f}')
            print(head_bbox)

            gaze_map_view = gaze_map.view(20, 30)
            save_image(gaze_map_view, os.path.join(save_image_dir_dic['attention'], data_type_id, f'{mode}_{data_id}_attention_{head_idx}.png'))
            gaze_map = cv2.imread(os.path.join(save_image_dir_dic['attention'], data_type_id, f'{mode}_{data_id}_attention_{head_idx}.png'), cv2.IMREAD_GRAYSCALE)
            gaze_map = cv2.resize(gaze_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            gaze_map_norm = norm_heatmap(gaze_map)
            gaze_map_norm = gaze_map_norm.astype(np.uint8)
            gaze_map_norm = cv2.applyColorMap(gaze_map_norm, cv2.COLORMAP_JET)
            gaze_map_norm_superimposed = cv2.addWeighted(img, 0.5, gaze_map_norm, 0.5, 0)
            cv2.imwrite(os.path.join(save_image_dir_dic['attention_superimposed'], data_type_id, f'{mode}_{data_id}_superimposed_{head_idx}.png'), gaze_map_norm_superimposed)

    sys.exit()
    # save joint attention estimation as a superimposed image
    # for person_idx in range(img_gt.shape[0]):
        # save_image(img_gt[person_idx], os.path.join(save_image_dir_dic['gt_map'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_gt.png'))

    # calc distances for each co att box
    # gt_box = gt_box.numpy()
    # gt_box_num = np.sum(np.sum(gt_box, axis=1)!=0)
    # for gt_box_idx in range(gt_box_num):
    #     x_min_gt, y_min_gt, x_max_gt, y_max_gt = map(float, gt_box[gt_box_idx])
    #     x_min_gt, x_max_gt = map(lambda x: int(x*original_width), [x_min_gt, x_max_gt])
    #     y_min_gt, y_max_gt = map(lambda x: int(x*original_height), [y_min_gt, y_max_gt])
    #     x_mid_gt, y_mid_gt = (x_min_gt+x_max_gt)//2, (y_min_gt+y_max_gt)//2

    # cv2.circle(superimposed_image, (x_mid_pred, y_mid_pred), 10, (128, 0, 128), thickness=-1)
    # cv2.circle(superimposed_image, (x_mid_gt, y_mid_gt), 10, (0, 255, 0), thickness=-1)
    # cv2.rectangle(superimposed_image, (x_min_pred, y_min_pred), (x_max_pred, y_max_pred), (128, 0, 128), thickness=1)
    # cv2.rectangle(superimposed_image, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 255, 0), thickness=1)
    # cv2.imwrite(os.path.join(save_image_dir_dic['joint_attention_superimposed'], data_type_id, f'{mode}_{data_id}_superimposed.png'), superimposed_image)