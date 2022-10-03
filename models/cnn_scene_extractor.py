from errno import ESHUTDOWN
import  numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import sys
import os
import timm

class SceneFeatureCNN(nn.Module):
    def __init__(self, cfg):
        super(SceneFeatureCNN, self).__init__()

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

        ## model params
        # position
        self.use_position = cfg.model_params.use_position

        # gaze
        self.use_gaze = cfg.model_params.use_gaze

        # action
        self.use_action = cfg.model_params.use_action

        # head embedding type
        self.head_embedding_type = cfg.model_params.head_embedding_type

        # whole image
        self.use_img = cfg.model_params.use_img

        # feature extractor
        self.people_feat_dim = cfg.model_params.people_feat_dim
        self.rgb_feat_dim = cfg.model_params.rgb_feat_dim
        self.rgb_cnn_extractor_type = cfg.model_params.rgb_cnn_extractor_type
        self.rgb_cnn_extractor_stage_idx = cfg.model_params.rgb_cnn_extractor_stage_idx
        self.p_s_estimator_cnn_pretrain = cfg.model_params.p_s_estimator_cnn_pretrain
        self.use_p_s_estimator_att_inside = cfg.model_params.use_p_s_estimator_att_inside

        # define loss function
        self.loss = cfg.exp_params.loss

        # head feature embedding
        embeding_param_num = 0
        if self.use_position:
            embeding_param_num += 2        
        if self.use_action:
            embeding_param_num += 9
        if self.use_gaze:
            embeding_param_num += 2
        if self.head_embedding_type == 'liner':
            self.head_info_feat_embeding = nn.Sequential(
                nn.Linear(embeding_param_num, self.people_feat_dim),
            )
        elif self.head_embedding_type == 'mlp':
            self.head_info_feat_embeding = nn.Sequential(
                nn.Linear(embeding_param_num, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
            )
        
        if 'resnet' in self.rgb_cnn_extractor_type:
            if self.p_s_estimator_cnn_pretrain:
                self.rgb_feat_extractor = models.resnet50(pretrained=False)
                num_ftrs = self.rgb_feat_extractor.fc.in_features
                num_classes = 365
                self.rgb_feat_extractor.fc = nn.Linear(num_ftrs, num_classes)
                weight_path = os.path.join('saved_weights', 'videoattentiontarget', 'pretrained_models', 'resnet50_places365.pt')
                weight_params = torch.load(weight_path)
                self.rgb_feat_extractor.load_state_dict(weight_params)
                self.rgb_feat_extractor = nn.Sequential(*list(self.rgb_feat_extractor.children())[:-2])
            else:
                self.rgb_feat_extractor = timm.create_model(self.rgb_cnn_extractor_type, features_only=True, pretrained=True)

            self.rgb_cnn_extractor_stage_idx = self.rgb_cnn_extractor_stage_idx
            if self.rgb_cnn_extractor_type == 'resnet50':
                feat_dim_list = [64, 256, 512, 1024, 2048]
            elif self.rgb_cnn_extractor_type == 'resnet18':
                feat_dim_list = [64, 64, 128, 256, 512]
            down_scale_list = [2, 4, 8, 16, 32]
            feat_dim = feat_dim_list[self.rgb_cnn_extractor_stage_idx]
            down_scale_ratio = down_scale_list[self.rgb_cnn_extractor_stage_idx]
            self.one_by_one_conv = nn.Sequential(
                                       nn.Conv2d(in_channels=feat_dim, out_channels=self.rgb_feat_dim, kernel_size=1),
                                       nn.ReLU(),
                                       )
        else:
            print('Please use correct rgb cnn extractor type')

        self.head_att_height = self.resize_height//down_scale_ratio
        self.head_att_width = self.resize_width//down_scale_ratio
        self.head_att_channel = self.rgb_feat_dim
        self.head_att_map_estimator = nn.Sequential(
            nn.Linear(self.people_feat_dim, self.people_feat_dim),
            nn.ReLU(),
            nn.Linear(self.people_feat_dim, self.people_feat_dim),
            nn.ReLU(),
            nn.Linear(self.people_feat_dim, 1),
        )

        # person scene heatmap estimator
        if self.loss == 'mse':
            final_activation_layer = nn.Identity()
        elif self.loss == 'bce':
            final_activation_layer = nn.Sigmoid()

        self.person_scene_heatmap_estimator = nn.Sequential(
            nn.ConvTranspose2d(self.rgb_feat_dim, self.rgb_feat_dim//2, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(self.rgb_feat_dim//2, self.rgb_feat_dim//4, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(self.rgb_feat_dim//4, self.rgb_feat_dim//8, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(self.rgb_feat_dim//8, self.rgb_feat_dim//16, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(self.rgb_feat_dim//16, 1, 4, 2, 1, bias=False),
            final_activation_layer,
        )

        if self.use_p_s_estimator_att_inside:
            self.loss_func_att_inside = nn.BCELoss(reduction='mean')
            self.person_att_inside_estimator = nn.Sequential(
            nn.Linear(self.rgb_feat_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
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
        torch.autograd.set_detect_anomaly(True)
        self.batch_size, people_num, _, _, _ = xy_axis_map.shape

        # rgb feature extraction
        if self.p_s_estimator_cnn_pretrain:
            rgb_feat = self.rgb_feat_extractor(rgb_img)
            rgb_feat = self.one_by_one_conv(rgb_feat)
        else:
            rgb_feat_set = self.rgb_feat_extractor(rgb_img)
            rgb_feat = rgb_feat_set[self.rgb_cnn_extractor_stage_idx]
            rgb_feat = self.one_by_one_conv(rgb_feat)
        
        rgb_feat_channel, rgb_feat_height, rgb_feat_width = rgb_feat.shape[-3:]
        rgb_feat_view = rgb_feat.view(self.batch_size, 1, rgb_feat_channel, rgb_feat_height, rgb_feat_width)
        rgb_feat_expand = rgb_feat_view.expand(self.batch_size, people_num, rgb_feat_channel, rgb_feat_height, rgb_feat_width)

        # head infomation embedding
        head_position = torch.cat([input_feature[:, :, :2]], dim=-1)
        head_action = torch.cat([input_feature[:, :, 2:]], dim=-1)
        head_gaze = torch.cat([input_gaze[:, :, :2]], dim=-1)
        if self.use_position and self.use_gaze and self.use_action:
            head_info_params = torch.cat([head_position, head_gaze, head_action], dim=-1)
        elif self.use_position and self.use_gaze:
            head_info_params = torch.cat([head_position, head_gaze], dim=-1)
        elif self.use_position and self.use_action:
            head_info_params = torch.cat([head_position, head_action], dim=-1)
        elif self.use_gaze and self.use_action:
            head_info_params = torch.cat([head_gaze, head_action], dim=-1)
        elif self.use_position:
            head_info_params = torch.cat([head_position], dim=-1)
        elif self.use_gaze:
            head_info_params = torch.cat([head_gaze], dim=-1)
        elif self.use_action:
            head_info_params = torch.cat([head_action], dim=-1)
        else:
            print('no person information')
            sys.exit()

        # head info embedding
        head_info_params = self.head_info_feat_embeding(head_info_params)

        # head att estimator
        head_att_params = self.head_att_map_estimator(head_info_params)        
        head_xy_map = head_xy_map * head_feature[:, :, :2, None, None]
        gaze_xy_map = gaze_xy_map * head_vector[:, :, :2, None, None]
        xy_axis_map_dif_head = xy_axis_map - head_xy_map
        x_axis_map_dif_head_mul_gaze = xy_axis_map_dif_head * gaze_xy_map
        xy_dot_product = torch.sum(x_axis_map_dif_head_mul_gaze, dim=2)
        xy_dot_product = xy_dot_product / (torch.norm(xy_axis_map_dif_head, dim=2) + self.epsilon)
        xy_dot_product = xy_dot_product / (torch.norm(gaze_xy_map, dim=2) + self.epsilon)
        theta_x_y = torch.acos(torch.clamp(xy_dot_product, -1+self.epsilon, 1-self.epsilon))
        theta_x_y = theta_x_y / self.pi
        theta_sigma = torch.exp(head_att_params[:, :, 0, None, None])
        head_att_map = torch.exp(-torch.pow(theta_x_y, 2)/(2*theta_sigma))
        head_att_map = F.interpolate(head_att_map, (rgb_feat_height, rgb_feat_width), mode='bilinear')
        head_att_map = head_att_map.view(self.batch_size, people_num, 1, rgb_feat_height, rgb_feat_width)

        # head att concat
        rgb_feat_head_att = rgb_feat_expand * head_att_map
        rgb_feat_head_att = rgb_feat_head_att.view(self.batch_size*people_num, -1, rgb_feat_height, rgb_feat_width)

        # attention map estimation
        person_scene_attention_heatmap = self.person_scene_heatmap_estimator(rgb_feat_head_att)
        person_scene_attention_heatmap = person_scene_attention_heatmap.view(self.batch_size, people_num, self.resize_height, self.resize_width)

        # attention inside estimation
        if self.use_p_s_estimator_att_inside:
            rgb_feat_head_att_gap = torch.mean(rgb_feat_head_att, dim=(-1, -2))
            estimated_att_inside = self.person_att_inside_estimator(rgb_feat_head_att_gap)
            estimated_att_inside = estimated_att_inside.view(self.batch_size, people_num)
            # estimated_att_inside_inv = 1-estimated_att_inside
            # estimated_att_inside_inv = estimated_att_inside_inv.view(self.batch_size, people_num, 1, 1)
            # person_scene_attention_heatmap = person_scene_attention_heatmap - estimated_att_inside_inv
            # person_scene_attention_heatmap = torch.clamp(input=person_scene_attention_heatmap, min=-0, max=1)

        # pack return values
        data = {}
        data['person_scene_attention_heatmap'] = person_scene_attention_heatmap
        data['ang_att_map'] = head_att_map
        if self.use_p_s_estimator_att_inside:
            data['estimated_att_inside'] = estimated_att_inside

        return data

    def calc_loss(self, inp, out, cfg):
        # unpack data
        att_inside_flag = inp['att_inside_flag']
        head_feature = inp['head_feature']
        no_padding_flag = torch.sum(head_feature == 0, dim=-1) == 0
        if self.use_p_s_estimator_att_inside:
            estimated_att_inside = inp['estimated_att_inside']

        loss_set = {}
        if self.use_p_s_estimator_att_inside:
            no_padding_flag_mask = no_padding_flag.flatten()
            estimated_att_inside_filt = estimated_att_inside[no_padding_flag]
            att_inside_flag_filt = att_inside_flag[no_padding_flag]
            estimated_att_inside = estimated_att_inside * no_padding_flag
            att_inside_flag = att_inside_flag * no_padding_flag
            loss_att_inside = self.loss_func_att_inside(estimated_att_inside_filt.float(), att_inside_flag_filt.float())
            loss_att_inside = loss_att_inside * 1e-2
            loss_set['loss_att_inside'] = loss_att_inside
            print(estimated_att_inside[0, no_padding_flag[0, :]])
            print(att_inside_flag[0, no_padding_flag[0, :]])
    
        return loss_set