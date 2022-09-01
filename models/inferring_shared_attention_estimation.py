import torch
import torch.nn as nn
import torchvision.models as models

class InferringSharedAttentionEstimator(nn.Module):
    def __init__(self, cfg):
        super(InferringSharedAttentionEstimator, self).__init__()

        ## set useful variables
        self.epsilon = 1e-7
        self.pi = 3.1415

        ## set data
        self.dataset_name = cfg.data.name

        ## exp settings
        self.resize_width = cfg.exp_set.resize_width
        self.resize_height = cfg.exp_set.resize_height

        self.gpu_list = range(cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish+1)
        self.device = torch.device(f"cuda:{self.gpu_list[0]}")
        self.wandb_name = cfg.exp_set.wandb_name
        self.batch_size = cfg.exp_set.batch_size

        # define loss function
        self.loss = cfg.exp_params.loss
        if self.loss == 'mse':
            print('Use MSE loss function')
            self.loss_func_joint_attention = nn.MSELoss()
        elif self.loss == 'bce':
            print('Use BCE loss function')
            self.loss_func_joint_attention = nn.BCELoss()
        elif self.loss == 'l1':
            print('Use l1 loss function')
            self.loss_func_joint_attention = nn.L1Loss()
        self.use_e_att_loss = cfg.exp_params.use_e_att_loss

        self.conv_in_channels = 1+1
        self.spatial_detection_module = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        input_feature = inp['input_feature']
        input_gaze = inp['input_gaze']
        head_vector = inp['head_vector']
        head_feature = inp['head_feature']
        xy_axis_map = inp['xy_axis_map']
        head_xy_map = inp['head_xy_map']
        gaze_xy_map = inp['gaze_xy_map']
        saliency_img = inp['saliency_img']
        rgb_img = inp['rgb_img']
        head_img_extract = inp['head_img_extract']
        att_inside_flag = inp['att_inside_flag']
        
        # get usuful variable
        batch_size, people_num, _, img_height, img_width = xy_axis_map.shape

        # generate head xy map
        head_xy_map = head_xy_map * head_feature[:, :, :2, None, None]

        # generate gaze xy map
        gaze_xy_map = gaze_xy_map * head_vector[:, :, :, None, None]

        # generate gaze cone map
        xy_axis_map_dif_head = xy_axis_map - head_xy_map
        x_axis_map_dif_head_mul_gaze = xy_axis_map_dif_head * gaze_xy_map
        xy_dot_product = torch.sum(x_axis_map_dif_head_mul_gaze, dim=2)
        xy_dot_product = xy_dot_product / (torch.norm(xy_axis_map_dif_head, dim=2) + self.epsilon)
        xy_dot_product = xy_dot_product / (torch.norm(gaze_xy_map, dim=2) + self.epsilon)

        # calculate theta and distance map
        theta_x_y = torch.acos(torch.clamp(xy_dot_product, -1+self.epsilon, 1-self.epsilon))

        # generate sigma of gaussian
        # multiply zero to padding maps
        self.gaussian_sigma = 0.5
        angle_dist = torch.exp(-torch.pow(theta_x_y, 2)/(2*self.gaussian_sigma**2)) / self.gaussian_sigma
        angle_dist = angle_dist * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]

        # sum all gaze maps (divide people num excluding padding people)
        no_pad_idx = torch.sum((torch.sum(head_feature, dim=2) != 0), dim=1)[:, None, None, None]
        angle_dist_sum_pooling = torch.sum(angle_dist, dim=1)[:, None, :, :] / no_pad_idx

        # cat angle img and saliency img
        # spatial detection module
        angle_saliency_img = torch.cat([angle_dist_sum_pooling, saliency_img], dim=1)
        angle_saliency_img = self.spatial_detection_module(angle_saliency_img)

        # return final img
        final_img = angle_saliency_img
        final_img = final_img[:, 0, :, :]

        # pack return values
        data = {}
        data['img_pred'] = final_img
        data['angle_dist'] = angle_dist

        return data

    def calc_loss(self, inp, out, cfg):
        # unpack data
        img_gt = inp['img_gt']
        att_inside_flag = inp['att_inside_flag']
        img_pred = out['img_pred']

        # switch loss coeficient
        if self.use_e_att_loss:
            loss_map_coef = 1
        else:
            loss_map_coef = 0

        # calculate final map loss
        img_gt_att = torch.sum(img_gt, dim=1)/torch.sum(att_inside_flag, dim=-1)[:, None, None]
        loss_map = self.loss_func_joint_attention(img_pred.float(), img_gt_att.float())
        loss_map = loss_map_coef * loss_map

        loss_set = {}
        loss_set['loss_map'] = loss_map

        return loss_set