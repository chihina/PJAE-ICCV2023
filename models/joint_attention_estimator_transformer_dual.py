import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import  numpy as np

class JointAttentionEstimatorTransformerDual(nn.Module):
    def __init__(self, cfg):
        super(JointAttentionEstimatorTransformerDual, self).__init__()

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

        # transformer
        self.people_feat_dim = cfg.model_params.people_feat_dim
        self.use_people_people_trans = cfg.model_params.use_people_people_trans
        self.people_people_trans_enc_num = cfg.model_params.people_people_trans_enc_num
        self.mha_num_heads_people_people = cfg.model_params.mha_num_heads_people_people

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
        self.use_middle_people_loss = cfg.exp_params.use_middle_people_loss
        self.use_final_loss = cfg.exp_params.use_final_loss

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
        elif self.head_embedding_type == 'ident':
            self.head_info_feat_embeding = nn.Sequential(
                nn.Linear(embeding_param_num, self.people_feat_dim),
            )
        else:
            print('please use correct head embedding type')
            sys.exit()

        self.ja_embedding = nn.Parameter(torch.zeros(1, 1, self.people_feat_dim))
        if self.use_people_people_trans:
            print('Use tranformer encoder for people relation')
            self.people_people_self_attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.people_feat_dim, num_heads=self.mha_num_heads_people_people, batch_first=True) for _ in range(self.people_people_trans_enc_num)])
            self.people_people_fc = nn.ModuleList(
                                        [nn.Sequential(
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.people_feat_dim, self.people_feat_dim),
                                        )
                                        for _ in range(self.people_people_trans_enc_num)
                                        ])
            self.trans_layer_norm_people_people = nn.LayerNorm(normalized_shape=self.people_feat_dim)

        self.hm_height, self.hm_width = 64, 64
        self.hm_height_middle, self.hm_width_middle = 16, 16
        if self.loss == 'mse':
            final_activation_layer = nn.Identity()
        elif self.loss == 'bce':
            final_activation_layer = nn.Sigmoid()
        self.heatmap_estimator_person_person = nn.Sequential(
            nn.Linear(self.people_feat_dim, self.people_feat_dim),
            nn.ReLU(),
            nn.Linear(self.people_feat_dim, self.hm_height_middle*self.hm_width_middle),
            final_activation_layer,
        )
        self.heatmap_estimator_final = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1),
            final_activation_layer,
        )

    def forward(self, inp):

        input_feature = inp['input_feature']
        input_gaze = inp['input_gaze']
        head_vector = inp['head_vector']
        head_feature = inp['head_feature']
        xy_axis_map = inp['xy_axis_map']
        head_xy_map = inp['head_xy_map']
        gaze_xy_map = inp['gaze_xy_map']

        torch.autograd.set_detect_anomaly(True)
        
        # get usuful variable
        self.batch_size, people_num, _, _, _ = xy_axis_map.shape

        # position and action info handing
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

        # person feature embedding
        head_info_params_emb = self.head_info_feat_embeding(head_info_params)
        ja_embedding = self.ja_embedding
        ja_embedding = ja_embedding.expand(self.batch_size, 1, self.people_feat_dim)
        head_info_params_emb = torch.cat([head_info_params_emb, ja_embedding], dim=1)

        # people relation encoding
        if self.use_people_people_trans:
            key_padding_mask_people_people = (torch.sum(head_feature, dim=-1) == 0).bool()
            key_padding_mask_joint_attention = torch.zeros(self.batch_size, 1, device=head_feature.device).bool()
            key_padding_mask_people_people = torch.cat([key_padding_mask_people_people, key_padding_mask_joint_attention], dim=1)

            for i in range(self.people_people_trans_enc_num):
                head_info_params_feat, people_people_trans_weights = self.people_people_self_attention[i](head_info_params_emb, head_info_params_emb, head_info_params_emb, key_padding_mask=key_padding_mask_people_people)
                head_info_params_feat_res = head_info_params_feat + head_info_params_emb
                head_info_params_feat_feed = self.people_people_fc[i](head_info_params_feat_res)
                head_info_params_feat_feed_res = head_info_params_feat_res + head_info_params_feat_feed
                head_info_params_feat_feed_res = self.trans_layer_norm_people_people(head_info_params_feat_feed_res)
                head_info_params_emb = head_info_params_feat_feed_res

                trans_att_people_people_i = people_people_trans_weights.view(self.batch_size, 1, people_num+1, people_num+1)
                if i == 0:
                    trans_att_people_people = trans_att_people_people_i
                else:
                    trans_att_people_people = torch.cat([trans_att_people_people, trans_att_people_people_i], dim=1)
        else:
            trans_att_people_people = torch.zeros(self.batch_size, self.people_people_trans_enc_num, people_num, people_num)

        ja_embedding_relation = head_info_params_emb[:, -1, :]
        hm_person_to_person = self.heatmap_estimator_person_person(ja_embedding_relation)
        hm_person_to_person = hm_person_to_person.view(self.batch_size, 1, self.hm_height_middle, self.hm_width_middle)
        hm_person_to_person = F.interpolate(hm_person_to_person, (self.hm_height, self.hm_width), mode='bilinear')

        hm_person_to_scene = inp['encoded_heatmap_davt']
        hm_person_to_scene = hm_person_to_scene.view(self.batch_size, people_num, 1, self.hm_height, self.hm_width)
        hm_person_to_scene = hm_person_to_scene * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None, None]
        hm_person_to_scene_mean = torch.sum(hm_person_to_scene, dim=1)
        no_pad_idx_cnt = torch.sum((torch.sum(head_feature, dim=2) != 0), dim=1)
        hm_person_to_scene_mean = hm_person_to_scene_mean / no_pad_idx_cnt[:, None, None, None]

        # final joint attention estimation
        hm_concat = torch.cat([hm_person_to_person, hm_person_to_scene_mean], dim=1)
        hm_final = self.heatmap_estimator_final(hm_concat)

        # generate head xy map
        head_xy_map = head_xy_map * head_feature[:, :, :2, None, None]

        # generate gaze xy map
        gaze_xy_map = gaze_xy_map * head_vector[:, :, :2, None, None]

        # generate gaze cone map
        xy_axis_map_dif_head = xy_axis_map - head_xy_map
        x_axis_map_dif_head_mul_gaze = xy_axis_map_dif_head * gaze_xy_map
        xy_dot_product = torch.sum(x_axis_map_dif_head_mul_gaze, dim=2)
        xy_dot_product = xy_dot_product / (torch.norm(xy_axis_map_dif_head, dim=2) + self.epsilon)
        xy_dot_product = xy_dot_product / (torch.norm(gaze_xy_map, dim=2) + self.epsilon)

        # calculate theta and distance map
        theta_x_y = torch.acos(torch.clamp(xy_dot_product, -1+self.epsilon, 1-self.epsilon))
        distance_x_y = torch.norm(xy_axis_map_dif_head, dim=2)

        # normalize theta and distance
        theta_x_y = theta_x_y / self.pi
        distance_x_y = distance_x_y / (2**0.5)

        angle_dist = torch.exp(-torch.pow(theta_x_y, 2)/(2* 0.5 ** 2))
        angle_dist = angle_dist * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]
        distance_dist = torch.exp(-torch.pow(theta_x_y, 2)/(2* 0.5 ** 2))
        distance_dist = distance_dist * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]
        trans_att_people_rgb = torch.zeros(self.batch_size, people_num, 1, self.hm_height, self.hm_height)
        trans_att_people_rgb = torch.zeros(self.batch_size, people_num, 1, self.hm_height, self.hm_height)

        hm_final = hm_final[:, 0, :, :]
        hm_person_to_person = hm_person_to_person[:, 0, :, :]
        hm_person_to_scene = hm_person_to_scene[:, :, 0, :, :]
        hm_person_to_scene_mean = hm_person_to_scene_mean[:, 0, :, :]

        # pack return values
        data = {}
        data['hm_final'] = hm_final
        data['hm_person_to_person'] = hm_person_to_person
        data['hm_person_to_scene'] = hm_person_to_scene
        data['hm_person_to_scene_mean'] = hm_person_to_scene_mean
        data['angle_dist'] = angle_dist
        data['distance_dist'] = distance_dist
        data['trans_att_people_rgb'] = trans_att_people_rgb
        data['trans_att_people_people'] = trans_att_people_people
        data['head_info_params'] = head_info_params

        return data

    def calc_loss(self, inp, out, cfg):
        # unpack data (input)
        img_gt = inp['img_gt']
        gt_box = inp['gt_box']
        gt_box_id = inp['gt_box_id']
        att_inside_flag = inp['att_inside_flag']

        # unpack data (output)
        hm_final = out['hm_final']
        hm_person_to_person = out['hm_person_to_person']
        hm_person_to_scene = out['hm_person_to_scene']
        hm_person_to_scene_mean = out['hm_person_to_scene_mean']

        # switch loss coeficient
        if self.use_middle_people_loss:
            loss_middle_people_coef = 1
        else:
            loss_middle_people_coef = 0

        if self.use_final_loss:
            loss_final_coef = 1
        else:
            loss_final_coef = 0

        # generate gt map
        img_gt_all = torch.sum(img_gt, dim=1)
        img_gt_all_thresh = torch.ones(1, device=img_gt.device)
        img_gt_all = torch.where(img_gt_all>img_gt_all_thresh, img_gt_all_thresh, img_gt_all)

        # calculate middle people loss
        hm_person_to_person_resize = F.interpolate(hm_person_to_person[:, None, :, :], (self.resize_height, self.resize_width), mode='bilinear')
        hm_person_to_person_resize = hm_person_to_person_resize[:, 0, :, :]
        loss_middle_people = self.loss_func_joint_attention(hm_person_to_person_resize.float(), img_gt_all.float())
        loss_middle_people = loss_middle_people_coef * loss_middle_people

        # calculate final loss
        hm_final_resize = F.interpolate(hm_final[:, None, :, :], (self.resize_height, self.resize_width), mode='bilinear')
        hm_final_resize = hm_final_resize[:, 0, :, :]
        loss_final = self.loss_func_joint_attention(hm_final_resize.float(), img_gt_all.float())
        loss_final = loss_final_coef * loss_final

        # print('hm_final', hm_final.shape)
        # print('hm_person_to_person', hm_person_to_person.shape)
        # print('hm_person_to_scene', hm_person_to_scene.shape)
        # print('hm_person_to_scene_mean', hm_person_to_scene_mean.shape)
        # sys.exit()

        # pack loss
        loss_set = {}
        loss_set['loss_middle_people'] = loss_middle_people
        loss_set['loss_final'] = loss_final

        return loss_set