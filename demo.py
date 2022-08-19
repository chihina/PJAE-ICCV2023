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
save_image_dir_list = ['joint_attention', 'joint_attention_superimposed', 'attention', 'attention_fan', 'attention_dist',
                    'gt_map', 'people_people_att', 'people_rgb_att']
for dir_name in save_image_dir_list:
    save_image_dir_dic[dir_name] = os.path.join('results', cfg.data.name, model_name, dir_name)
    if not os.path.exists(save_image_dir_dic[dir_name]):
        os.makedirs(save_image_dir_dic[dir_name])

print("===> Starting demo processing")
stop_iteration = 20
if mode == 'test':
    stop_iteration = 50
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
    img_pred = out['img_pred'].to('cpu').detach()[0]
    img_mid_pred = out['img_mid_pred'].to('cpu').detach()[0]
    angle_dist = out['angle_dist'].to('cpu').detach()[0]
    distance_dist = out['distance_dist'].to('cpu').detach()[0]
    saliency_img = out['saliency_img'].to('cpu').detach()[0]
    head_tensor = out['head_tensor'].to('cpu').detach()[0].numpy()
    head_vector_gt = out['head_vector_gt'].to('cpu').detach()[0].numpy()
    head_feature = out['head_feature'].to('cpu').detach()[0]
    trans_att_people_people = out['trans_att_people_people'].to('cpu').detach()[0].numpy()
    gt_box = out['gt_box'].to('cpu').detach()[0]
    att_inside_flag = out['att_inside_flag'].to('cpu').detach()[0]
    img_path = out['rgb_path'][0]

    if cfg.model_params.model_type == 'ja_transformer':
        trans_att_people_rgb = out['trans_att_people_rgb'].to('cpu').detach()[0]

    # redefine image size
    img = cv2.imread(img_path)
    original_height, original_width, _ = img.shape
    cfg.exp_set.resize_height = original_height
    cfg.exp_set.resize_width = original_width

    # define data id
    data_type_id = data_type_id_generator(head_vector_gt, head_tensor, gt_box, cfg)
    data_id = data_id_generator(img_path, cfg)
    print(f'Iter:{iteration}, {data_id}, {data_type_id}')

    # expand directories
    for dir_name in ['joint_attention', 'joint_attention_superimposed', 'people_people_att']:
        if not os.path.exists(os.path.join(save_image_dir_dic[dir_name], data_type_id)):
            os.makedirs(os.path.join(save_image_dir_dic[dir_name], data_type_id))
    for dir_name in ['attention', 'attention_fan', 'attention_dist', 'people_people_att', 'people_rgb_att', 'gt_map']:
        if not os.path.exists(os.path.join(save_image_dir_dic[dir_name], data_type_id, f'{data_id}')):
            os.makedirs(os.path.join(save_image_dir_dic[dir_name], data_type_id, f'{data_id}'))

    # save joint attention estimation
    save_image(img_pred, os.path.join(save_image_dir_dic['joint_attention'], data_type_id, f'{mode}_{data_id}_joint_attention.png'))

    key_no_padding_num = torch.sum((torch.sum(head_feature, dim=-1) != 0)).numpy()
    if cfg.model_params.model_type == 'ja_transformer':
    # save attention of transformers (people and rgb attention)
        people_num, rgb_people_trans_enc_num, rgb_feat_height, rgb_feat_width = trans_att_people_rgb.shape
        trans_att_people_rgb = trans_att_people_rgb.view(rgb_people_trans_enc_num*people_num, 1, rgb_feat_height, rgb_feat_width)
        trans_att_people_rgb = F.interpolate(trans_att_people_rgb, (cfg.exp_set.resize_height, cfg.exp_set.resize_width), mode='nearest')
        trans_att_people_rgb = trans_att_people_rgb.view(people_num, rgb_people_trans_enc_num, 1, cfg.exp_set.resize_height, cfg.exp_set.resize_width)

        for person_idx in range(key_no_padding_num):
            for i in range(cfg.model_params.rgb_people_trans_enc_num):
                trans_att_people_rgb_save = trans_att_people_rgb[person_idx, i, 0, :, :]
                trans_att_people_rgb_save_max, trans_att_people_rgb_save_min = torch.max(trans_att_people_rgb_save), torch.min(trans_att_people_rgb_save)
                trans_att_people_rgb_save = (trans_att_people_rgb_save-trans_att_people_rgb_save_min)/(trans_att_people_rgb_save_max-trans_att_people_rgb_save_min)
                save_image(trans_att_people_rgb_save, os.path.join(save_image_dir_dic['people_rgb_att'], data_type_id, f'{data_id}', f'{mode}_{data_id}_p{person_idx}_enc{i}_people_rgb_attention.png'))

    # save attention of transformers (people and people attention)
    df_person = [person_idx for person_idx in range(key_no_padding_num)]
    for i in range(cfg.model_params.people_people_trans_enc_num):
        plt.figure(figsize=(8, 6))
        trans_att_people_people_enc = pd.DataFrame(data=trans_att_people_people[i, :key_no_padding_num, :key_no_padding_num], index=df_person, columns=df_person)
        sns.heatmap(trans_att_people_people_enc, cmap='jet')
        plt.savefig(os.path.join(save_image_dir_dic['people_people_att'], data_type_id, f'{data_id}', f'{mode}_{data_id}_enc{i}_people_people_att.png'))
        plt.close()

    # save attention of each person
    for person_idx in range(key_no_padding_num):
        save_image(img_mid_pred[person_idx], os.path.join(save_image_dir_dic['attention'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_attention.png'))
        save_image(angle_dist[person_idx], os.path.join(save_image_dir_dic['attention_fan'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_fan.png'))
        save_image(distance_dist[person_idx], os.path.join(save_image_dir_dic['attention_dist'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_dist.png'))
        save_image(img_gt[person_idx], os.path.join(save_image_dir_dic['gt_map'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_gt.png'))

    # save joint attention estimation as a superimposed image
    img = cv2.resize(img, (cfg.exp_set.resize_width, cfg.exp_set.resize_height))
    img_heatmap = cv2.imread(os.path.join(save_image_dir_dic['joint_attention'], data_type_id, f'{mode}_{data_id}_joint_attention.png'), cv2.IMREAD_GRAYSCALE)
    img_heatmap = cv2.resize(img_heatmap, (img.shape[1], img.shape[0]))
    img_heatmap_norm = norm_heatmap(img_heatmap)
    img_heatmap_norm = img_heatmap_norm.astype(np.uint8)
    img_heatmap_norm = cv2.applyColorMap(img_heatmap_norm, cv2.COLORMAP_JET)

    if cfg.model_params.dynamic_distance_type == 'gaussian':
        superimposed_image = cv2.addWeighted(img, 1.0, img_heatmap_norm, 0, 0)
    else:
        superimposed_image = cv2.addWeighted(img, 0.5, img_heatmap_norm, 0.5, 0)

    # calculate metrics for each attetntion estimation
    for person_idx in range(key_no_padding_num):
        # calc a centers of pred bboxes
        if cfg.model_params.dynamic_distance_type == 'gaussian':
            peak_x_mid_pred, peak_y_mid_pred = head_tensor[person_idx, 3], head_tensor[person_idx, 4]
            peak_x_mid_pred, peak_y_mid_pred = peak_x_mid_pred*cfg.exp_set.resize_width, peak_y_mid_pred*cfg.exp_set.resize_height
            peak_x_mid_pred, peak_y_mid_pred = map(int, [peak_x_mid_pred, peak_y_mid_pred])
        else:
            img_heatmap_person = cv2.imread(os.path.join(save_image_dir_dic['attention'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_attention.png'), cv2.IMREAD_GRAYSCALE)
            img_heatmap_person = cv2.resize(img_heatmap_person, (img.shape[1], img.shape[0]))
            peak_y_mid_pred, peak_x_mid_pred = np.unravel_index(np.argmax(img_heatmap_person), img_heatmap_person.shape)

        if att_inside_flag[person_idx]:
            # calc a center of gt bbox
            peak_x_min_gt, peak_y_min_gt, peak_x_max_gt, peak_y_max_gt = gt_box[person_idx]
            peak_x_mid_gt, peak_y_mid_gt = (peak_x_min_gt+peak_x_max_gt)/2, (peak_y_min_gt+peak_y_max_gt)/2
            peak_x_mid_gt, peak_y_mid_gt = peak_x_mid_gt*cfg.exp_set.resize_width, peak_y_mid_gt*cfg.exp_set.resize_height
            peak_x_mid_gt, peak_y_mid_gt = map(int, [peak_x_mid_gt, peak_y_mid_gt])

            # calc metrics
            peak_x_diff, peak_y_diff = peak_x_mid_pred-peak_x_mid_gt, peak_y_mid_pred-peak_y_mid_gt
            l2_dist = np.power(np.power(peak_x_diff, 2) + np.power(peak_y_diff, 2), 0.5)
            print(f'Person_id:{person_idx}, Dis={l2_dist:.0f}, GT=({peak_x_mid_gt:.0f},{peak_y_mid_gt:.0f}), peak=({peak_x_mid_pred:.0f},{peak_y_mid_pred:.0f})')

        # get head position and gaze direction
        head_tensor_person = head_tensor[person_idx]
        head_vec_x, head_vec_y = head_tensor_person[0:2]
        head_feature_person = head_feature[person_idx]
        head_x, head_y = head_feature_person[0:2]
        head_x, head_y = int(head_x*cfg.exp_set.resize_width), int(head_y*cfg.exp_set.resize_height)
        scale_factor = 30
        pred_x = int(head_vec_x*scale_factor + head_x)
        pred_y = int(head_vec_y*scale_factor + head_y)

        # save attention estimation as a superimposed image
        img_heatmap_person = cv2.imread(os.path.join(save_image_dir_dic['attention'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_attention.png'), cv2.IMREAD_GRAYSCALE)
        img_heatmap_person = norm_heatmap(img_heatmap_person)
        img_heatmap_person = img_heatmap_person.astype(np.uint8)
        img_heatmap_person = cv2.applyColorMap(cv2.resize(img_heatmap_person, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)
        
        img_heatmap_angle = cv2.imread(os.path.join(save_image_dir_dic['attention_fan'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_fan.png'), cv2.IMREAD_GRAYSCALE)
        img_heatmap_angle = img_heatmap_angle.astype(np.uint8)
        img_heatmap_angle = cv2.applyColorMap(cv2.resize(img_heatmap_angle, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)

        img_heatmap_distance = cv2.imread(os.path.join(save_image_dir_dic['attention_dist'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_dist.png'), cv2.IMREAD_GRAYSCALE)
        img_heatmap_distance = img_heatmap_distance.astype(np.uint8)
        img_heatmap_distance = cv2.applyColorMap(cv2.resize(img_heatmap_distance, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)

        if cfg.model_params.dynamic_distance_type == 'gaussian':
            superimposed_image_person = cv2.addWeighted(img, 1.0, img_heatmap_person, 0, 0)
            # superimposed_image_angle = cv2.addWeighted(img, 1.0, img_heatmap_angle, 0, 0)
            # superimposed_image_distance = cv2.addWeighted(img, 1.0, img_heatmap_distance, 0, 0)
        else:
            superimposed_image_person = cv2.addWeighted(img, 0.5, img_heatmap_person, 0.5, 0)
            # superimposed_image_angle = cv2.addWeighted(img, 0.5, img_heatmap_angle, 0.5, 0)
            # superimposed_image_distance = cv2.addWeighted(img, 0.5, img_heatmap_distance, 0.5, 0)

        superimposed_image_angle = cv2.addWeighted(img, 0.5, img_heatmap_angle, 0.5, 0)
        superimposed_image_distance = cv2.addWeighted(img, 0.5, img_heatmap_distance, 0.5, 0)

        # head pose arrow color
        arrow_set = (255, 255, 255)
        # arrow_set = (0, 0, 0)

        if cfg.model_params.model_type == 'ja_transformer':
            # save attention of people and rgb
            for i in range(cfg.model_params.rgb_people_trans_enc_num):            
                people_rgb_att_one = cv2.imread(os.path.join(save_image_dir_dic['people_rgb_att'], data_type_id, f'{data_id}', f'{mode}_{data_id}_p{person_idx}_enc{i}_people_rgb_attention.png'), cv2.IMREAD_GRAYSCALE)
                people_rgb_att_one = people_rgb_att_one.astype(np.uint8)
                people_rgb_att_one = cv2.applyColorMap(cv2.resize(people_rgb_att_one, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)
                superimposed_people_rgb_att_one = cv2.addWeighted(img, 0.5, people_rgb_att_one, 0.5, 0)
                superimposed_people_rgb_att_one = cv2.arrowedLine(superimposed_people_rgb_att_one, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
                cv2.circle(superimposed_people_rgb_att_one, (peak_x_mid_pred, peak_y_mid_pred), 10, (128, 0, 128), thickness=-1)
                if att_inside_flag[person_idx]:
                    cv2.circle(superimposed_people_rgb_att_one, (peak_x_mid_gt, peak_y_mid_gt), 10, (0, 255, 0), thickness=-1)
                cv2.imwrite(os.path.join(save_image_dir_dic['people_rgb_att'], data_type_id, f'{data_id}', f'{mode}_{data_id}_p{person_idx}_enc{i}_people_rgb_attention.png'), superimposed_people_rgb_att_one)

        superimposed_image= cv2.arrowedLine(superimposed_image, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
        superimposed_image_person = cv2.arrowedLine(superimposed_image_person, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
        superimposed_image_angle = cv2.arrowedLine(superimposed_image_angle, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)
        superimposed_image_distance = cv2.arrowedLine(superimposed_image_distance, (head_x, head_y), (pred_x, pred_y), arrow_set, thickness=3)

        # plot ground-truth and predicted points
        cv2.circle(superimposed_image, (peak_x_mid_pred, peak_y_mid_pred), 10, (128, 0, 128), thickness=-1)
        cv2.circle(superimposed_image_person, (peak_x_mid_pred, peak_y_mid_pred), 10, (128, 0, 128), thickness=-1)
        cv2.circle(superimposed_image_angle, (peak_x_mid_pred, peak_y_mid_pred), 10, (128, 0, 128), thickness=-1)
        cv2.circle(superimposed_image_distance, (peak_x_mid_pred, peak_y_mid_pred), 10, (128, 0, 128), thickness=-1)

        if att_inside_flag[person_idx]:
            cv2.circle(superimposed_image, (peak_x_mid_gt, peak_y_mid_gt), 10, (0, 255, 0), thickness=-1)
            cv2.circle(superimposed_image_person, (peak_x_mid_gt, peak_y_mid_gt), 10, (0, 255, 0), thickness=-1)
            cv2.circle(superimposed_image_angle, (peak_x_mid_gt, peak_y_mid_gt), 10, (0, 255, 0), thickness=-1)
            cv2.circle(superimposed_image_distance, (peak_x_mid_gt, peak_y_mid_gt), 10, (0, 255, 0), thickness=-1)

        # save attention of each person
        cv2.imwrite(os.path.join(save_image_dir_dic['attention'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_attention.png'), superimposed_image_person)
        cv2.imwrite(os.path.join(save_image_dir_dic['attention_fan'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_fan.png'), superimposed_image_angle)
        cv2.imwrite(os.path.join(save_image_dir_dic['attention_dist'], data_type_id, f'{data_id}', f'{mode}_{data_id}_{person_idx}_dist.png'), superimposed_image_distance)

    # save joint attention estimation as a superimposed image
    cv2.imwrite(os.path.join(save_image_dir_dic['joint_attention_superimposed'], data_type_id, f'{mode}_{data_id}_superimposed.png'), superimposed_image)