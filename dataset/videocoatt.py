import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
import sys
from PIL import Image
from tqdm import tqdm
import time
import matplotlib
matplotlib.use('Agg')
import random

class VideoCoAttDataset(Dataset):
    def __init__(self, cfg, mode, transform=None, pre_transform=None):

        # data
        self.dataset_dir = cfg.data.dataset_dir
        self.saliency_dataset_dir = cfg.data.saliency_dataset_dir
        self.wandb_name = cfg.exp_set.wandb_name
        self.mode = mode

        # exp_params    
        self.train_det_heads = cfg.exp_params.train_det_heads
        self.train_heads_conf = cfg.exp_params.train_heads_conf
        if self.train_det_heads:
            print('Train the model by detetected heads')
        self.gaussian_sigma = cfg.exp_params.gaussian_sigma

        # test settings
        self.test_heads_type = cfg.exp_params.test_heads_type
        self.det_heads_model = cfg.exp_params.det_heads_model
        self.test_heads_conf = cfg.exp_params.test_heads_conf
        self.test_heads_size = cfg.exp_params.test_heads_size

        # exp set
        self.resize_width = cfg.exp_set.resize_width
        self.resize_height = cfg.exp_set.resize_height
        self.resize_head_width = cfg.exp_set.resize_head_width
        self.resize_head_height = cfg.exp_set.resize_head_height

        self.feature_list = []
        self.head_bbox_list = []
        self.gt_bbox = []
        self.rgb_path_list = []
        self.saliency_path_list = []

        # make dataset
        if self.mode in ['train', 'validate']:
            self.generate_train_dataset_list()
            self.set_random_sample()
        elif self.mode == 'test':
            self.generate_test_dataset_list()
        else:
            print('please employ correct mode')
            sys.exit()

        # set maximum number of people
        self.max_num_people = 0
        self.set_max_num_people()

        self.transforms = transforms.Compose(
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
        # get path of images_nk
        img_file_path = self.rgb_path_list[idx]
        saliency_file_path = self.saliency_path_list[idx]

        # read a rgb image
        img = Image.open(img_file_path)
        img_width, img_height = img.size

        # generate gt heatmaps
        bboxes_ori = self.gt_bbox[idx]
        x_min, y_min, x_max, y_max = map(int, bboxes_ori[0])
        x_mid, y_mid = (x_min+x_max)/2, (y_min+y_max)/2
        gt_img = self.load_gt_imgs(img_width, img_height, bboxes_ori, self.gaussian_sigma)

        # zero padding for mini-batch training
        bboxes = np.zeros((self.max_num_people, 4))
        bboxes[:len(bboxes_ori), :] = bboxes_ori

        # initialize tensors
        head_img = torch.zeros(self.max_num_people, 3, self.resize_head_height, self.resize_head_width)
        head_vector_gt_tensor = torch.zeros(self.max_num_people, 2)
        head_feature_tensor = torch.zeros(self.max_num_people, 2)
        head_bbox_tensor = torch.zeros(self.max_num_people, 4)

        # crop heads in images_nk
        for head_idx in range(len(self.feature_list[idx])):
            head_x, head_y = map(int, self.feature_list[idx][head_idx])
            head_x_min, head_y_min, head_x_max, head_y_max = map(int, self.head_bbox_list[idx][head_idx])
            head_ball_vec_x, head_ball_vec_y = x_mid-head_x, y_mid-head_y
            head_ball_vec_norm = ((head_ball_vec_x**2+head_ball_vec_y**2) ** 0.5) + 1e-5
            head_ball_vec_x, head_ball_vec_y = head_ball_vec_x/head_ball_vec_norm, head_ball_vec_y/head_ball_vec_norm
            croped_head = img.crop((head_x_min, head_y_min, head_x_max, head_y_max))

            # transform tensor 
            if self.transforms:
                croped_head = self.transforms(croped_head)

            head_img[head_idx, :, :, :] = croped_head
            head_feature_tensor[head_idx, :2] = torch.tensor([head_x, head_y])
            head_vector_gt_tensor[head_idx, :2] = torch.tensor([head_ball_vec_x, head_ball_vec_y])
            head_bbox_tensor[head_idx, :4] = torch.tensor(list(map(int, [head_x_min, head_y_min, head_x_max, head_y_max])))

        head_feature_tensor[:, 0] /= img_width
        head_feature_tensor[:, 1] /= img_height
        gt_img = torch.tensor(gt_img).float()

        # transform tensor 
        if self.transforms_rgb:
            rgb_tensor = self.transforms_rgb(Image.open(img_file_path))
        if self.transforms_gt:
            gt_img = self.transforms_gt(gt_img)[0, :, :]
        if self.transforms_saliency:
            saliency_tensor = self.transforms_saliency(Image.open(saliency_file_path))

        # pack one data into a dict
        batch = {}
        batch['head_img'] = head_img
        batch['head_feature'] = head_feature_tensor
        batch['head_vector_gt'] = head_vector_gt_tensor
        batch['img_gt'] = gt_img
        batch['gt_box'] = bboxes
        batch['rgb_img'] = rgb_tensor
        batch['saliency_img'] = saliency_tensor
        batch['att_inside_flag'] = torch.sum(torch.tensor(bboxes), dim=-1)!=0

        return batch

    # set maximum number of people
    def set_max_num_people(self):
        for data_idx in range(len(self.feature_list)):
            data_people_num = len(self.feature_list[data_idx])
            self.max_num_people = max(self.max_num_people, data_people_num)

    # read graph information
    def generate_train_dataset_list(self):
        video_cnt = 0
        for video_num in tqdm(sorted(os.listdir(os.path.join(self.dataset_dir, 'images_nk', self.mode)))):
            video_cnt += 1
            seq_cnt = 0
            annotation_path = os.path.join(self.dataset_dir, 'annotations', self.mode, f'{video_num}.txt')
            ann_dic = self.read_annotation_file(annotation_path)

            for file_name in sorted(os.listdir(os.path.join(self.dataset_dir, 'images_nk', self.mode, str(video_num)))):

                # get rgb file path
                frame_id = file_name.split('.')[0].split('_')[0]
                rgb_img_file_path = os.path.join(self.dataset_dir, 'images_nk', self.mode, str(video_num), file_name)

                # get saliency file path
                saliency_file_path = os.path.join(self.saliency_dataset_dir, 'images', self.mode, str(video_num), file_name)

                # following authors setting
                if (int(frame_id) - 1) % 10 != 0:
                    continue

                # if annotation is not exists
                if int(frame_id) not in ann_dic.keys():
                    continue
                else:
                    # # get annotation data
                    ann_info = ann_dic[int(frame_id)]

                    # get yolo detection results
                    if self.train_det_heads:
                        det_file_path = os.path.join(self.dataset_dir, self.det_heads_model, self.mode, str(video_num), file_name.replace('jpg', 'txt'))
                        det_bbox_conf = self.read_det_file(det_file_path, self.mode)
                        det_bbox, det_conf = det_bbox_conf[:, :-1], det_bbox_conf[:, -1]
                        use_head_boxes_dets = np.array(det_bbox)
                        use_head_boxes_gt = np.array(sum(ann_info['head_bbox_list'], []), dtype=np.int32).reshape(-1, 4)
                        if use_head_boxes_dets.shape[0] == 0:
                            use_head_boxes = np.concatenate([use_head_boxes_gt], 0)
                        else:
                            use_head_boxes = np.concatenate([use_head_boxes_dets, use_head_boxes_gt], 0)

                        use_head_boxes, _, _ = self.nms_fast(use_head_boxes, np.ones(use_head_boxes.shape[0]), np.ones(use_head_boxes.shape[0]), 0.1)

                    else:
                        use_head_boxes = np.array(sum(ann_info['head_bbox_list'], []), dtype=np.int32).reshape(-1, 4)

                    for co_att_idx in range(len(ann_info['co_att_bbox_list'])):
                        co_att_bbox = np.array(ann_info['co_att_bbox_list'][co_att_idx], dtype=np.int32).reshape(1, 4)
                        use_head_boxes = np.array(ann_info['head_bbox_list'][co_att_idx], dtype=np.int32).reshape(-1, 4)

                        # calculate each person head position
                        use_head_feature = np.zeros((use_head_boxes.shape[0], 2))
                        if use_head_boxes.shape[0] != 0:
                            use_head_feature[:, 0] = (use_head_boxes[:, 0] + use_head_boxes[:, 2]) / 2
                            use_head_feature[:, 1] = (use_head_boxes[:, 1] + use_head_boxes[:, 3]) / 2

                        self.gt_bbox.append(co_att_bbox)
                        self.rgb_path_list.append(rgb_img_file_path)
                        self.saliency_path_list.append(saliency_file_path)
                        self.head_bbox_list.append(use_head_boxes)
                        self.feature_list.append(use_head_feature)
                        seq_cnt += 1

                        if self.wandb_name == 'demo' and seq_cnt > 1:
                            break

                # if self.mode == 'validate' and seq_cnt > 5:
                    # break
                # if 'debug' in self.wandb_name and seq_cnt > 10:
                    # break
                if self.wandb_name == 'demo' and seq_cnt > 1:
                    break

            # if self.mode == 'validate' and video_cnt > 5:
                # break
            # if self.wandb_name == 'debug' and video_cnt > 5:
                # break
            if self.wandb_name == 'demo' and video_cnt > 20:
                break

    # read test information
    def generate_test_dataset_list(self):
        video_cnt = 0
        for video_num in tqdm(sorted(os.listdir(os.path.join(self.dataset_dir, 'images_nk', self.mode)))):
            video_cnt += 1
            seq_cnt = 0

            annotation_path = os.path.join(self.dataset_dir, 'annotations', self.mode, f'{video_num}.txt')
            ann_dic = self.read_annotation_file(annotation_path)

            for file_name in sorted(os.listdir(os.path.join(self.dataset_dir, 'images_nk', self.mode, str(video_num)))):
                # get rgb file path
                frame_id = file_name.split('.')[0].split('_')[0]
                rgb_img_file_path = os.path.join(self.dataset_dir, 'images_nk', self.mode, str(video_num), file_name)

                # get saliency file path
                saliency_file_path = os.path.join(self.saliency_dataset_dir, 'images', self.mode, str(video_num), file_name)

                # following authors setting
                if (int(frame_id) - 1) % 10 == 0:
                    pass
                else:
                    continue

                # if no co-att box in the frame
                if int(frame_id) not in ann_dic.keys():
                    continue
                else:
                    # # get annotation data
                    ann_info = ann_dic[int(frame_id)]
                    co_att_bbox = np.array(ann_info['co_att_bbox_list'])

                # get yolo detection results
                if self.test_heads_type == 'det':
                    det_file_path = os.path.join(self.dataset_dir, self.det_heads_model, self.mode, str(video_num), file_name.replace('jpg', 'txt'))
                    det_bbox_conf = self.read_det_file(det_file_path, self.mode)
                    det_bbox, det_conf = det_bbox_conf[:, :-1], det_bbox_conf[:, -1]
                    use_head_boxes = np.array(det_bbox)
                elif self.test_heads_type == 'gt':
                    use_head_boxes = np.array(sum(ann_info['head_bbox_list'], []), dtype=np.int32).reshape(-1, 4)
                else:
                    print('Please employ correct heads type')

                # calculate each person head position
                use_head_feature = np.zeros((use_head_boxes.shape[0], 2))
                if use_head_boxes.shape[0] != 0:
                    use_head_feature[:, 0] = (use_head_boxes[:, 0] + use_head_boxes[:, 2]) / 2
                    use_head_feature[:, 1] = (use_head_boxes[:, 1] + use_head_boxes[:, 3]) / 2

                self.gt_bbox.append(co_att_bbox)
                self.rgb_path_list.append(rgb_img_file_path)
                self.saliency_path_list.append(saliency_file_path)
                self.head_bbox_list.append(use_head_boxes)
                self.feature_list.append(use_head_feature)

                seq_cnt += 1

                # if 'test' in self.wandb_name and seq_cnt > 5:
                    # break
                if self.wandb_name == 'demo' and seq_cnt > 1:
                    break
                # if 'debug' in self.wandb_name and seq_cnt > 1:
                #     break

            # if 'test' in self.wandb_name and video_cnt > 5:
                # break
            if self.wandb_name == 'demo' and video_cnt > 40:
                break

    def set_random_sample(self):
        iteration_num = len(self.gt_bbox)
        self.gt_bbox_ymin_array = np.zeros((iteration_num))
        for gt_bbox_idx in range(iteration_num):
            self.gt_bbox_ymin_array[gt_bbox_idx] = np.min(self.gt_bbox[gt_bbox_idx][:, 1])

    def get_random_sample(self):
        gt_ymin_array = self.gt_bbox_ymin_array
        y_cls_num = 3
        y_cls = random.randint(0, y_cls_num-1)
        y_range_min = y_cls * (self.resize_height//y_cls_num)
        y_range_max = (y_cls+1) * (self.resize_height//y_cls_num)
        gt_ymin_array_split = np.where((gt_ymin_array > y_range_min) & (gt_ymin_array < y_range_max))[0]
        idx = np.random.choice(gt_ymin_array_split.shape[0], 1, replace=False)[0]
        return idx

    # generage gt imgs for probability heatmap
    def load_gt_imgs(self, img_width, img_height, bbox, gamma):
        gt_gaussian = np.zeros((1, img_height, img_width), dtype=np.float32)
        co_bbox_num = bbox.shape[0]
        for co_idx in range(co_bbox_num):
            x_min, y_min, x_max, y_max = map(int, bbox[co_idx])
            x_center, y_center = int((x_max + x_min)//2), int((y_max + y_min)//2)
            gt_gaussian += self.generate_2d_gaussian(img_height, img_width, (x_center, y_center), gamma)
    
        return gt_gaussian

    # get a ball location
    def read_annotation_file(self, txt_file_path: str) -> list:
        
        ann_dic = {}
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            co_att_id, frame_id, bbox_info = line.split()[0], line.split()[1], line.split()[2:]
            co_att_bbox, head_bbox = bbox_info[0:4], bbox_info[4:]
            co_att_bbox = list(map(int, co_att_bbox))

            if not int(frame_id) in ann_dic.keys():
                ann_dic[int(frame_id)] = {}
                ann_dic[int(frame_id)]['co_att_id_list'] = []
                ann_dic[int(frame_id)]['co_att_bbox_list'] = []
                ann_dic[int(frame_id)]['head_bbox_list'] = []

            ann_dic[int(frame_id)][f'co_att_id_list'].append(co_att_id)
            ann_dic[int(frame_id)][f'co_att_bbox_list'].append(co_att_bbox)
            ann_dic[int(frame_id)][f'head_bbox_list'].append(head_bbox)

        return ann_dic

    # read yolo detection results
    def read_det_file(self, txt_file_path: str, mode: str) -> list:
        
        # if detection result are not found
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as f:
                lines = f.readlines()
        else:
            return np.zeros((0, 4))

        bbox_arrray = np.array(list(map(lambda x: x.split(), lines)), dtype=np.float32)

        if mode == 'test':
            heads_conf_thresh = self.test_heads_conf
        else:
            heads_conf_thresh = self.train_heads_conf

        bbox_arrray = bbox_arrray[bbox_arrray[:, -1] > heads_conf_thresh]
        bbox_arrray = bbox_arrray.astype(np.int32)

        return bbox_arrray

    def str_time(self):
        self.time_str = time.time()
        self.time_idx = 0

    def print_time(self):
        self.time_idx += 1
        print(f'{self.time_idx}, {time.time()-self.time_str}')

    # generator gt gaussian
    def generate_2d_gaussian(self, height, width, peak_xy, sigma=1):
        peak_x, peak_y = peak_xy
        x = np.arange(0, width, 1)
        y = np.arange(0, height, 1)
        X,Y = np.meshgrid(x,y)
        heatmap = np.exp(- ((X-peak_x) ** 2 + (Y-peak_y) ** 2) / (2 * sigma ** 2))

        return heatmap.reshape(1, heatmap.shape[0], heatmap.shape[1])

    def nms_fast(self, bboxes, scores, classes, iou_threshold=0.5):
        areas = (bboxes[:,2] - bboxes[:,0] + 1) \
                * (bboxes[:,3] - bboxes[:,1] + 1)
        
        sort_index = np.argsort(scores)
        
        i = -1 # 未処理の矩形のindex
        while(len(sort_index) >= 2 - i):
            max_scr_ind = sort_index[i]
            ind_list = sort_index[:i]
            iou = self.iou_np(bboxes[max_scr_ind], bboxes[ind_list], \
                        areas[max_scr_ind], areas[ind_list])
            
            del_index = np.where(iou >= iou_threshold)
            sort_index = np.delete(sort_index, del_index)
            i -= 1 # 未処理の矩形のindexを1減らす
        
        bboxes = bboxes[sort_index]
        scores = scores[sort_index]
        classes = classes[sort_index]
        
        return bboxes, scores, classes

    def iou_np(self, a, b, a_area, b_area):
 
        abx_mn = np.maximum(a[0], b[:,0]) # xmin
        aby_mn = np.maximum(a[1], b[:,1]) # ymin
        abx_mx = np.minimum(a[2], b[:,2]) # xmax
        aby_mx = np.minimum(a[3], b[:,3]) # ymax
        w = np.maximum(0, abx_mx - abx_mn + 1)
        h = np.maximum(0, aby_mx - aby_mn + 1)
        intersect = w*h
        
        iou_np = intersect / (a_area + b_area - intersect)
        return iou_np
