from os import lseek
from random import expovariate
from tokenize import triple_quoted
import  numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import sys
import timm
import math
from einops.layers.torch import Rearrange

class SceneFeatureTransformer(nn.Module):
    def __init__(self, cfg):
        super(SceneFeatureTransformer, self).__init__()

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
        self.rgb_feat_dim = self.people_feat_dim
        self.rgb_cnn_extractor_type = cfg.model_params.rgb_cnn_extractor_type
        self.rgb_cnn_extractor_stage_idx = cfg.model_params.rgb_cnn_extractor_stage_idx

        # rgb-person transformer
        self.rgb_people_trans_dim = self.rgb_feat_dim
        self.rgb_people_trans_enc_num = cfg.model_params.rgb_people_trans_enc_num
        self.mha_num_heads_rgb_people = cfg.model_params.mha_num_heads_rgb_people

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

        if self.rgb_cnn_extractor_type == 'rgb_patch':
            down_scale_ratio = cfg.model_params.rgb_patch_size
            patch_height = down_scale_ratio
            patch_width = down_scale_ratio
            patch_dim = 3 * patch_height * patch_width
            self.rgb_feat_extractor = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, self.rgb_feat_dim),
            )
        elif 'resnet' in self.rgb_cnn_extractor_type:
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
        
        self.hm_height = self.resize_height//down_scale_ratio
        self.hm_width = self.resize_width//down_scale_ratio

        # rgb person transformer
        self.pe_generator_rgb = PositionalEmbeddingGenerator(self.hm_height, self.hm_width, self.rgb_feat_dim)
        self.rgb_people_self_attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.rgb_people_trans_dim, kdim=self.rgb_people_trans_dim, vdim=self.rgb_people_trans_dim, num_heads=self.mha_num_heads_rgb_people, batch_first=True) for _ in range(self.rgb_people_trans_enc_num)])
        self.rgb_people_fc = nn.ModuleList(
                                    [nn.Sequential(
                                    nn.Linear(self.rgb_people_trans_dim, self.rgb_people_trans_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.rgb_people_trans_dim, self.rgb_people_trans_dim),
                                    )
                                    for _ in range(self.rgb_people_trans_enc_num)
                                    ])
        self.trans_layer_norm_people_rgb = nn.LayerNorm(normalized_shape=self.rgb_people_trans_dim)

        # person scene heatmap estimator
        if self.loss == 'mse':
            final_activation_layer = nn.Identity()
        elif self.loss == 'bce':
            final_activation_layer = nn.Sigmoid()
        self.person_scene_heatmap_estimator = nn.Sequential(
                nn.Linear(self.people_feat_dim, self.people_feat_dim),
                nn.ReLU(),
                nn.Linear(self.people_feat_dim, 1),
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
        saliency_img = inp['saliency_img']
        rgb_img = inp['rgb_img']
        head_img_extract = inp['head_img_extract']
        att_inside_flag = inp['att_inside_flag']

        # get usuful variable
        torch.autograd.set_detect_anomaly(True)
        self.batch_size, people_num, _, _, _ = xy_axis_map.shape

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
        head_info_params = self.head_info_feat_embeding(head_info_params)

        # rgb feature extraction
        if self.rgb_cnn_extractor_type == 'rgb_patch':
            rgb_feat_patch = self.rgb_feat_extractor(rgb_img)
            rgb_feat_channel, rgb_feat_height, rgb_feat_width = rgb_feat_patch.shape[-1], self.hm_height, self.hm_width
        elif 'resnet' in self.rgb_cnn_extractor_type:
            rgb_feat_set = self.rgb_feat_extractor(rgb_img)
            rgb_feat = rgb_feat_set[self.rgb_cnn_extractor_stage_idx]
            rgb_feat = self.one_by_one_conv(rgb_feat)
            rgb_feat_channel, rgb_feat_height, rgb_feat_width = rgb_feat.shape[-3:]
            rgb_feat_patch = rgb_feat.view(self.batch_size, rgb_feat_channel, -1)
            rgb_feat_patch = torch.transpose(rgb_feat_patch, 1, 2)
        else:
            sys.exit()
        rgb_feat_patch_view = rgb_feat_patch.view(self.batch_size, 1, -1, self.rgb_feat_dim)
        rgb_feat_patch_expand = rgb_feat_patch_view.expand(self.batch_size, people_num, rgb_feat_patch.shape[1], self.rgb_feat_dim)
        head_info_params_view = head_info_params.view(self.batch_size, people_num, 1, self.people_feat_dim)
        head_info_params_expand = head_info_params_view.expand(self.batch_size, people_num, rgb_feat_patch.shape[1], self.people_feat_dim)
        rgb_pos_embedding = self.pe_generator_rgb.pos_embedding
        rgb_pos_embedding_view = rgb_pos_embedding.view(1, 1, -1, rgb_feat_channel)
        rgb_feat_patch_pos_expand = rgb_feat_patch_expand + rgb_pos_embedding_view
        rgb_people_feat_all = torch.cat([rgb_feat_patch_expand, head_info_params_view], dim=-2)
        rgb_people_feat_all_pos = torch.cat([rgb_feat_patch_pos_expand, head_info_params_view], dim=-2)
        rgb_people_feat_all = rgb_people_feat_all.view(self.batch_size*people_num, -1, self.rgb_feat_dim)
        rgb_people_feat_all_pos = rgb_people_feat_all_pos.view(self.batch_size*people_num, -1, self.rgb_feat_dim)

        # rgb person transformer
        for i in range(self.rgb_people_trans_enc_num):
            rgb_people_feat, rgb_people_trans_weights = self.rgb_people_self_attention[i](rgb_people_feat_all_pos, rgb_people_feat_all_pos, rgb_people_feat_all)
            rgb_people_feat_res = rgb_people_feat + rgb_people_feat_all
            rgb_people_feat_feed = self.rgb_people_fc[i](rgb_people_feat_res)
            rgb_people_feat_feed_res = rgb_people_feat_res + rgb_people_feat_feed
            rgb_people_feat_feed_res = self.trans_layer_norm_people_rgb(rgb_people_feat_feed_res)
            rgb_people_feat_all = rgb_people_feat_feed_res
            rgb_people_trans_weights_people_rgb = rgb_people_trans_weights[:, (rgb_feat_height*rgb_feat_width):, :(rgb_feat_height*rgb_feat_width)]
            trans_att_people_rgb_i = rgb_people_trans_weights_people_rgb.view(self.batch_size, people_num, 1, rgb_feat_height, rgb_feat_width)
            if i == 0:
                trans_att_people_rgb = trans_att_people_rgb_i
            else:
                trans_att_people_rgb = torch.cat([trans_att_people_rgb, trans_att_people_rgb_i], dim=2)
        
        rgb_people_feat_all = rgb_people_feat_all[:, :(rgb_feat_height*rgb_feat_width), :]
        person_scene_attention_heatmap = self.person_scene_heatmap_estimator(rgb_people_feat_all)
        person_scene_attention_heatmap = person_scene_attention_heatmap.view(self.batch_size, people_num, self.hm_height, self.hm_width)

        # pack return values
        data = {}
        data['person_scene_attention_heatmap'] = person_scene_attention_heatmap

        return data

    def calc_loss(self, inp, out, cfg):
        # unpack data
        img_gt = inp['img_gt']
        gt_box = inp['gt_box']
        gt_box_id = inp['gt_box_id']
        att_inside_flag = inp['att_inside_flag']
        img_pred = out['img_pred']
        img_mid_pred = out['img_mid_pred']
        img_mid_mean_pred = out['img_mid_mean_pred']
        head_tensor = out['head_tensor']
        rgb_people_feat_all = out['rgb_people_feat_all']
        head_feature = inp['head_feature']

        # switch loss coeficient
        if self.use_e_map_loss:
            loss_map_gaze_coef = 1
        else:
            loss_map_gaze_coef = 0

        loss_set = {}
        loss_set['loss_map_gaze_each'] = loss_map_gaze_each

        return loss_set

class PositionalEmbeddingGenerator(nn.Module):
    def __init__(self, h, w, dim):
        super().__init__()

        self.num_patches = w * h
        self._make_position_embedding(w, h, dim)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                        scale=2 * math.pi):
            h, w = self.pe_h, self.pe_w
            area = torch.ones(1, h, w)  # [b, h, w]
            y_embed = area.cumsum(1, dtype=torch.float32)
            x_embed = area.cumsum(2, dtype=torch.float32)

            one_direction_feats = d_model // 2

            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

            dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
            dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            pos = pos.flatten(2).permute(0, 2, 1)
            return pos