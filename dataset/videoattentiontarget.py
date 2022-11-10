import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
import sys
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import glob
import pandas as pd

class VideoAttentionTargetDataset(Dataset):
    def __init__(self, cfg, mode, transform=None, pre_transform=None):

        # dataset variables
        self.dataset_dir = cfg.data.dataset_dir
        self.wandb_name = cfg.data.wandb_name
        self.mode = mode

        # data parameters
        self.resize_width = cfg.exp_set.resize_width
        self.resize_height = cfg.exp_set.resize_height
        self.resize_head_width = cfg.exp_set.resize_head_width
        self.resize_head_height = cfg.exp_set.resize_head_height

        # gt parameters
        self.bbox_types = cfg.exp_params.bbox_types
        self.bbox_iou_thresh = cfg.exp_params.bbox_iou_thresh
        self.gaussian_sigma = cfg.exp_params.gaussian_sigma

        # experiment parameters
        self.feature_list = []
        self.head_bbox_list = []
        self.gt_bbox = []
        self.gt_bbox_id = []

        self.rgb_path_list = []
        self.saliency_path_list = []
        self.att_inside_list = []

        # make dataset
        self.generate_dataset_list()

        # set maximum number of people
        self.max_num_people = 0
        self.set_max_num_people()

        self.transforms_head = transforms.Compose(
            [
                transforms.Resize((self.resize_head_height, self.resize_head_width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transforms_rgb = transforms.Compose(
            [
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transforms_gt = transforms.Compose(
            [
                transforms.Resize((self.resize_height, self.resize_width)),
            ]
        )

        self.transforms_saliency = transforms.Compose(
            [
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.rgb_path_list)

    def __getitem__(self, idx):

        # get image path
        img_file_path = self.rgb_path_list[idx]
        saliency_file_path = self.saliency_path_list[idx]

        # open images
        img = Image.open(img_file_path)
        img_width, img_height = img.size

        # generate gt heatmaps
        bboxes_origin = np.copy(self.gt_bbox[idx])
        gt_img = self.load_gt_imgs_ball(img_width, img_height, bboxes_origin, self.gaussian_sigma)
        gt_img = torch.tensor(gt_img).float()
        att_inside_flag = np.zeros((self.max_num_people))
        att_inside_flag[:bboxes_origin.shape[0]] = np.copy(self.att_inside_list[idx])[:, 0]

        # normalize gt bbox
        bboxes = np.zeros((self.max_num_people, 4))
        bboxes[:bboxes_origin.shape[0], :] = np.copy(self.gt_bbox[idx])
        bboxes[:, ::2] /= img_width
        bboxes[:, 1::2] /= img_height

        # initialize tensors
        head_img = torch.zeros(self.max_num_people, 3, self.resize_head_height, self.resize_head_width)
        head_feature_tensor = torch.zeros(self.max_num_people, 2)
        head_bbox_tensor = torch.zeros(self.max_num_people, 4)
        head_vector_gt_tensor = torch.zeros(self.max_num_people, 2)

        # get features of heads
        head_num = self.feature_list[idx].shape[0]
        for head_idx in range(head_num):
            head_x, head_y = map(int, self.feature_list[idx][head_idx])
            head_x_min, head_y_min, head_x_max, head_y_max = map(int, self.head_bbox_list[idx][head_idx])
            x_mid = (bboxes_origin[head_idx, 0]+bboxes_origin[head_idx, 2])/2
            y_mid = (bboxes_origin[head_idx, 1]+bboxes_origin[head_idx, 3])/2
            head_ball_vec_x, head_ball_vec_y = x_mid-head_x, y_mid-head_y
            head_ball_vec_norm = ((head_ball_vec_x**2+head_ball_vec_y**2) ** 0.5) + 1e-5
            head_ball_vec_x, head_ball_vec_y = head_ball_vec_x/head_ball_vec_norm, head_ball_vec_y/head_ball_vec_norm
            croped_head = img.crop((head_x_min, head_y_min, head_x_max, head_y_max))

            # transform tensor 
            if self.transforms_head:
                croped_head = self.transforms_head(croped_head)

            head_img[head_idx, :, :, :] = croped_head
            head_feature_tensor[head_idx, :2] = torch.tensor(self.feature_list[idx][head_idx])
            head_vector_gt_tensor[head_idx, :2] = torch.tensor([head_ball_vec_x, head_ball_vec_y])
            head_bbox_tensor[head_idx, :4] = torch.tensor([head_x_min, head_y_min, head_x_max, head_y_max])

        # normalize head position
        head_feature_tensor[:, 0] /= img_width
        head_feature_tensor[:, 1] /= img_height
        head_bbox_tensor[:, ::2] /= img_width
        head_bbox_tensor[:, 1::2] /= img_height

        # transform tensor 
        if self.transforms_rgb:
            rgb_tensor = self.transforms_rgb(Image.open(img_file_path))
        if self.transforms_gt:
            gt_img = self.transforms_gt(gt_img)
        if self.transforms_saliency:
            saliency_tensor = self.transforms_saliency(Image.open(saliency_file_path))

        # generate gt box id for joint attention estimation
        gt_box_id_original = torch.tensor(self.gt_bbox_id[idx])
        gt_box_id_original_num = gt_box_id_original.shape[0]
        gt_box_id_original_max = torch.max(gt_box_id_original)
        gt_box_id_expand_num = self.max_num_people - gt_box_id_original_num
        gt_box_id_expand = torch.tensor([(gt_box_id_original_max+i+1) for i in range(gt_box_id_expand_num)]).view(-1, 1)
        gt_box_id = torch.cat([gt_box_id_original, gt_box_id_expand], dim=0).long()

        data = {}
        data['head_img'] = head_img
        data['head_feature'] = head_feature_tensor
        data['head_vector_gt'] = head_vector_gt_tensor
        data['img_gt'] = gt_img
        data['gt_box'] = bboxes
        data['gt_box_id'] = gt_box_id
        data['rgb_img'] = rgb_tensor
        data['saliency_img'] = saliency_tensor
        data['head_bbox'] = head_bbox_tensor
        data['att_inside_flag'] = att_inside_flag
        data['rgb_path'] = img_file_path

        return data

    # set maximum number of people
    def set_max_num_people(self):
        for data_idx in range(len(self.feature_list)):
            data_people_num = len(self.feature_list[data_idx])
            self.max_num_people = max(self.max_num_people, data_people_num)

    def get_detected_heads(self, img_dir_name, seq_name, img_width, img_height):
        det_heads_dic = {}
        det_file_path_list = glob.glob(os.path.join(self.dataset_dir, 'det_heads', img_dir_name, seq_name, '*'))
        for det_file_path in det_file_path_list:
            det_file_name = det_file_path.split('/')[-1].split('.')[0]
            data_id = f'{img_dir_name}_{seq_name}_{det_file_name}'
            det_heads_dic[data_id] = []
            with open(det_file_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                det_type, bbox = line.split()[0], line.split()[1:]
                if det_type == '1':
                    x_mid, y_mid, width, height = map(float, bbox)
                    x_min, y_min, x_max, y_max = x_mid-width/2, y_mid-height/2, x_mid+width/2, y_mid+height/2
                    x_min, x_max = map(lambda x:int(x*img_width), [x_min, x_max])
                    y_min, y_max = map(lambda x:int(x*img_height), [y_min, y_max])
                    det_heads_dic[data_id].append([x_min, y_min, x_max, y_max])

        return det_heads_dic

    def iou_np(self, a, b):

        a_area = (a[2] - a[0] + 1) \
                * (a[3] - a[1] + 1)
        b_area = (b[:,2] - b[:,0] + 1) \
                * (b[:,3] - b[:,1] + 1)
        
        abx_mn = np.maximum(a[0], b[:,0]) # xmin
        aby_mn = np.maximum(a[1], b[:,1]) # ymin
        abx_mx = np.minimum(a[2], b[:,2]) # xmax
        aby_mx = np.minimum(a[3], b[:,3]) # ymax
        w = np.maximum(0, abx_mx - abx_mn + 1)
        h = np.maximum(0, aby_mx - aby_mn + 1)
        intersect = w*h
        
        iou = intersect / (a_area + b_area - intersect)
        return iou

    # read graph information
    def generate_dataset_list(self):
        for img_dir_name in sorted(os.listdir(os.path.join(self.dataset_dir, 'annotations', self.mode))):
            for seq_name in sorted(os.listdir(os.path.join(self.dataset_dir, 'annotations', self.mode, img_dir_name))):
                seq_dic = {}
                
                if self.bbox_types == 'PRED':
                    rgb_img_file_path = glob.glob(os.path.join(self.dataset_dir, 'images', img_dir_name, seq_name, '*'))[0]
                    img_width, img_height = Image.open(rgb_img_file_path).size
                    det_heads_dic = self.get_detected_heads(img_dir_name, seq_name, img_width, img_height)
                
                for person_ann_path in sorted(os.listdir(os.path.join(self.dataset_dir, 'annotations', self.mode, img_dir_name, seq_name))):
                    with open(os.path.join(self.dataset_dir, 'annotations', self.mode, img_dir_name, seq_name, person_ann_path)) as f:
                        lines = f.readlines()
                    for line_idx, line in enumerate(lines):
                        
                        # follow the cvpr2022 paper
                        if line_idx % 5 != 0:
                            continue
                        # if line_idx % 30 != 0:
                        #     continue

                        line = line.strip().split(',')
                        img_name = line[0]
                        img_id = img_name.split('.')[0]
                        img_path = os.path.join(self.dataset_dir, 'images', img_dir_name, seq_name, img_name)

                        head_bbox = list(map(int, line[1:5]))
                        att_point = list(map(int, line[5:7]))

                        if att_point[0] == -1 and att_point[1] == -1:
                            att_inside = False
                        else:
                            att_inside = True

                        if not img_path in seq_dic.keys():
                            seq_dic[img_path] = {}
                            seq_dic[img_path]['head_bbox'] = []
                            seq_dic[img_path]['att_point'] = []
                            seq_dic[img_path]['att_inside'] = []

                        # detected heads filtering using gt heads
                        head_use_flag = True
                        if self.bbox_types == 'PRED':
                            det_heads = det_heads_dic[f'{img_dir_name}_{seq_name}_{img_id}']
                            det_heads_bbox = np.array(det_heads)
                            gt_head_bbox = np.array(head_bbox)

                            iou_det_heads = self.iou_np(gt_head_bbox, det_heads_bbox)
                            if np.max(iou_det_heads) < self.bbox_iou_thresh:
                                head_use_flag = False

                        if head_use_flag:                        
                            seq_dic[img_path]['head_bbox'].append(head_bbox)
                            seq_dic[img_path]['att_point'].append(att_point)
                            seq_dic[img_path]['att_inside'].append(att_inside)

                for img_path, img_item in seq_dic.items():

                    # get head features
                    head_bbox = np.array(img_item['head_bbox']).reshape(-1, 4)
                    use_head_feature = np.zeros((head_bbox.shape[0], 2))
                    if head_bbox.shape[0] != 0:
                        use_head_feature[:, 0] = (head_bbox[:, 0] + head_bbox[:, 2]) / 2
                        use_head_feature[:, 1] = (head_bbox[:, 1] + head_bbox[:, 3]) / 2

                    # get attention points
                    gt_point = np.array(img_item['att_point']).reshape(-1, 2)
                    gt_bbox = np.zeros((gt_point.shape[0], 4))
                    box_size = 20
                    gt_bbox[:, 0:2], gt_bbox[:, 2:4] = gt_point - box_size, gt_point + box_size
                    att_inside = np.array(img_item['att_inside']).reshape(-1, 1)
                    gt_bbox_idx = np.arange(gt_bbox.shape[0]).reshape(-1, 1)

                    if np.sum(att_inside) == 0:
                        continue

                    # append information of one image
                    self.rgb_path_list.append(img_path)
                    self.saliency_path_list.append(img_path)
                    self.head_bbox_list.append(head_bbox)
                    self.feature_list.append(use_head_feature)
                    self.gt_bbox.append(gt_bbox)
                    self.gt_bbox_id.append(gt_bbox_idx)
                    self.att_inside_list.append(att_inside)

    # generage gt imgs for probability heatmap
    def load_gt_imgs_ball(self, img_width, img_height, bbox, gamma):
        gt_gaussian = np.zeros((self.max_num_people, img_height, img_width))
        gt_box_num = bbox.shape[0]
        # generate a heatmap of each attention
        for co_idx in range(gt_box_num):
            x_min, y_min, x_max, y_max = map(int, bbox[co_idx])
            x_center, y_center = int((x_max + x_min)//2), int((y_max + y_min)//2)
            gt_gaussian_one = self.generate_2d_gaussian(img_height, img_width, (x_center, y_center), gamma)
            gt_gaussian[co_idx, :, :] =  gt_gaussian_one

        return gt_gaussian

    # generator gt gaussian
    def generate_2d_gaussian(self, height, width, peak_xy, sigma=1):
        peak_x, peak_y = peak_xy
        x = np.arange(0, width, 1)
        y = np.arange(0, height, 1)
        X,Y = np.meshgrid(x,y)
        heatmap = np.exp(- ((X-peak_x) ** 2 + (Y-peak_y) ** 2) / (2 * sigma ** 2))

        return heatmap.reshape(1, heatmap.shape[0], heatmap.shape[1])