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

class JointAttentionEstimatorCNN(nn.Module):
    def __init__(self, cfg):
        super(JointAttentionEstimatorCNN, self).__init__()

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
        self.use_position_enc_person = cfg.model_params.use_position_enc_person
        self.use_position_enc_type = cfg.model_params.use_position_enc_type

        # gaze
        self.use_gaze = cfg.model_params.use_gaze
        self.gaze_type = cfg.model_params.gaze_type

        # action
        self.use_action = cfg.model_params.use_action

        # Whole image
        self.use_img = cfg.model_params.use_img

        # gaze map
        self.use_angle_dist_rgb_type = cfg.model_params.use_angle_dist_rgb_type
        self.use_dynamic_angle = cfg.model_params.use_dynamic_angle
        self.use_dynamic_distance = cfg.model_params.use_dynamic_distance
        self.dynamic_distance_type = cfg.model_params.dynamic_distance_type

        # transformer
        self.use_people_people_trans = cfg.model_params.use_people_people_trans
        self.people_people_trans_enc_num = cfg.model_params.people_people_trans_enc_num
        self.mha_num_heads_people_people = cfg.model_params.mha_num_heads_people_people

        self.rgb_cnn_extractor_type = cfg.model_params.rgb_cnn_extractor_type
        self.rgb_cnn_extractor_stage_idx = cfg.model_params.rgb_cnn_extractor_stage_idx
        self.rgb_embeding_dim = cfg.model_params.rgb_embeding_dim
        self.people_feat_dim = cfg.model_params.people_feat_dim

        # others
        self.angle_distance_fusion = cfg.model_params.angle_distance_fusion

        # define loss function
        if cfg.exp_params.loss == 'mse':
            print('Use MSE loss function')
            self.loss_func_joint_attention = nn.MSELoss()
        elif cfg.exp_params.loss == 'bce':
            print('Use BCE loss function')
            self.loss_func_joint_attention = nn.BCELoss()
        self.use_e_map_loss = cfg.exp_params.use_e_map_loss
        self.use_e_att_loss = cfg.exp_params.use_e_att_loss
        self.use_each_e_map_loss = cfg.exp_params.use_each_e_map_loss
        self.use_regression_loss = cfg.exp_params.use_regression_loss
        self.use_attraction_loss = cfg.exp_params.use_attraction_loss
        self.use_repulsion_loss = cfg.exp_params.use_repulsion_loss

        # set people feat dim based on model params
        self.people_feat_dim = self.rgb_embeding_dim

        embeding_param_num = 0
        if self.use_position:
            if self.use_position_enc_person:
                self.person_positional_encoding = PositionalEncoding2D_RGB(self.people_feat_dim)
            else:
                embeding_param_num += 2
        
        if self.use_action:
            embeding_param_num += 9

        if self.gaze_type == 'vector':
            if self.use_gaze:
                embeding_param_num += 2
        elif self.gaze_type == 'feature':
            gaze_feat_emb_dim = 64
            if self.use_gaze:
                embeding_param_num += gaze_feat_emb_dim
            self.gaze_feat_embeding = nn.Sequential(
                nn.Linear(512, gaze_feat_emb_dim),
                nn.ReLU(inplace=False),
            )

        self.head_info_feat_embeding = nn.Sequential(
            nn.Linear(embeding_param_num, self.people_feat_dim),
            nn.ReLU(),
            nn.Linear(self.people_feat_dim, self.people_feat_dim),
            nn.ReLU(),
            nn.Linear(self.people_feat_dim, self.people_feat_dim),
        )

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
            # self.trans_layer_norm_people_people = nn.LayerNorm(normalized_shape=self.people_feat_dim)

        if self.rgb_cnn_extractor_type == 'normal':
            feat_ext_rgb = models.resnet50(pretrained=True)
            modules = list(feat_ext_rgb.children())[:-2]
            self.rgb_feat_extractor = nn.Sequential(
                                       *modules,
                                       nn.Conv2d(in_channels=2048, out_channels=self.rgb_embeding_dim, kernel_size=1),
                                       nn.ReLU(),
                                       )
        elif self.rgb_cnn_extractor_type == 'patch':
            down_scale_ratio = 8
            patch_height = down_scale_ratio
            patch_width = down_scale_ratio
            patch_dim = 3 * patch_height * patch_width
            self.rgb_feat_extractor = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, 64),
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
                                       nn.Conv2d(in_channels=feat_dim, out_channels=self.rgb_embeding_dim, kernel_size=1),
                                       nn.ReLU(),
                                       )
        elif 'hrnet' in self.rgb_cnn_extractor_type:
            self.rgb_feat_extractor = timm.create_model(self.rgb_cnn_extractor_type, features_only=True, pretrained=True)
            self.rgb_cnn_extractor_stage_idx = self.rgb_cnn_extractor_stage_idx
            if self.rgb_cnn_extractor_type == 'hrnet_w18_small':
                feat_dim_list = [64, 128, 256, 512, 1024]
            elif self.rgb_cnn_extractor_type == 'hrnet_w32':
                feat_dim_list = [64, 128, 256, 512, 1024]
            down_scale_list = [2, 4, 8, 16, 32]
            feat_dim = feat_dim_list[self.rgb_cnn_extractor_stage_idx]
            down_scale_ratio = down_scale_list[self.rgb_cnn_extractor_stage_idx]
            self.one_by_one_conv = nn.Sequential(
                                    nn.Conv2d(in_channels=feat_dim, out_channels=self.rgb_embeding_dim, kernel_size=1),
                                    nn.ReLU(),
                                    )
        elif self.rgb_cnn_extractor_type == 'convnext':
            self.rgb_feat_extractor = timm.create_model('convnext_base', features_only=True, pretrained=True)
            self.rgb_cnn_extractor_stage_idx = self.rgb_cnn_extractor_stage_idx
            feat_dim = 128 * 2 ** (self.rgb_cnn_extractor_stage_idx)
            self.one_by_one_conv = nn.Sequential(
                                       nn.Conv2d(in_channels=feat_dim, out_channels=self.rgb_embeding_dim, kernel_size=1),
                                       nn.ReLU(),
                                       )
        elif self.rgb_cnn_extractor_type == 'saliency':
            down_scale_ratio = 8
            patch_height = down_scale_ratio
            patch_width = down_scale_ratio
            patch_dim = 1 * patch_height * patch_width
            self.rgb_feat_extractor = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, 64),
            )
        elif self.rgb_cnn_extractor_type == 'no_use':
            pass
        else:
            print('Please use correct rgb cnn extractor type')

        self.down_height = self.resize_height//down_scale_ratio
        self.down_width = self.resize_width//down_scale_ratio

        nz = self.rgb_embeding_dim
        self.attention_generator = nn.Sequential(
            nn.ConvTranspose2d(nz, nz//2, 4, 2, 1, bias=False),
            # nn.GroupNorm(1, nz//2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nz//2, nz//4, 4, 2, 1, bias=False),
            # nn.GroupNorm(1, nz//4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nz//4, nz//4, 4, 2, 1, bias=False),
            # nn.GroupNorm(1, nz//4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nz//4, nz//4, 4, 2, 1, bias=False),
            # nn.GroupNorm(1, nz//4),
            nn.ReLU(True),
            nn.Conv2d(nz//4, 1, kernel_size=(1, 1)),
        )

        if self.use_dynamic_angle:
            print(f'Use dynamic angle')
        if self.use_dynamic_distance:
            print(f'Use dynamic distance')
            if self.angle_distance_fusion == 'mult':
                print(f'Use mult fusion (angle and distance)')
            elif self.angle_distance_fusion == 'max':
                print(f'Use max fusion (angle and distance)')
            elif self.angle_distance_fusion == 'mean':
                print(f'Use mean fusion (angle and distance)')
            else:
                print(f'Use correct fusion method')
                sys.exit()

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

        torch.autograd.set_detect_anomaly(True)
        
        # get usuful variable
        self.batch_size, people_num, _, _, _ = xy_axis_map.shape

        # position and action info handing
        head_position = torch.cat([input_feature[:, :, :2]], dim=-1)
        head_action = torch.cat([input_feature[:, :, 2:]], dim=-1)

        # gaze info handing
        if self.gaze_type == 'vector':
            head_gaze = torch.cat([input_gaze[:, :, :2]], dim=-1)
        elif self.gaze_type == 'feature':
            head_img_extract = head_img_extract.view(self.batch_size*people_num, -1)
            head_gaze = self.gaze_feat_embeding(head_img_extract)
            head_gaze = head_gaze.view(self.batch_size, people_num, -1)

        if self.use_position and self.use_gaze and self.use_action:
            if self.use_position_enc_person:
                head_info_params = torch.cat([head_gaze, head_action], dim=-1)
            else:
                head_info_params = torch.cat([head_position, head_gaze, head_action], dim=-1)
        elif self.use_position and self.use_gaze:
            if self.use_position_enc_person:
                head_info_params = torch.cat([head_gaze], dim=-1)
            else:
                head_info_params = torch.cat([head_position, head_gaze], dim=-1)
        elif self.use_position and self.use_action:
            if self.use_position_enc_person:
                head_info_params = torch.cat([head_action], dim=-1)
            else:
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

        # position encoding of person
        head_info_params = self.head_info_feat_embeding(head_info_params)
        if self.use_position and self.use_position_enc_person:
            head_position_encoding = self.person_positional_encoding(head_position)
            head_info_params = head_info_params + head_position_encoding

        # people relation encoding
        if self.use_people_people_trans:
            key_padding_mask_people_people = (torch.sum(head_feature, dim=-1) == 0)
            for i in range(self.people_people_trans_enc_num):
                head_info_params_feat, people_people_trans_weights = self.people_people_self_attention[i](head_info_params, head_info_params, head_info_params, key_padding_mask=key_padding_mask_people_people)
                head_info_params_feat_res = head_info_params_feat + head_info_params
                head_info_params_feat_feed = self.people_people_fc[i](head_info_params_feat_res)
                head_info_params_feat_feed_res = head_info_params_feat_res + head_info_params_feat_feed
                # head_info_params_feat_feed_res = self.trans_layer_norm_people_people(head_info_params_feat_feed_res)
                head_info_params_feat_feed_res = self.trans_layer_norm_people_rgb(head_info_params_feat_feed_res)
                head_info_params = head_info_params_feat_feed_res

                trans_att_people_people_i = people_people_trans_weights.view(self.batch_size, 1, people_num, people_num)
                if i == 0:
                    trans_att_people_people = trans_att_people_people_i
                else:
                    trans_att_people_people = torch.cat([trans_att_people_people, trans_att_people_people_i], dim=1)
        else:
            trans_att_people_people = torch.zeros(self.batch_size, self.people_people_trans_enc_num, people_num, people_num)

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

        # extract rgb feature
        if self.rgb_cnn_extractor_type == 'saliency':
            rgb_feat_patch = self.rgb_feat_extractor(saliency_img)
        elif self.rgb_cnn_extractor_type == 'convnext':
            rgb_feat_set = self.rgb_feat_extractor(rgb_img)
            rgb_feat = rgb_feat_set[self.rgb_cnn_extractor_stage_idx]
            rgb_feat = self.one_by_one_conv(rgb_feat)
        elif 'resnet' in self.rgb_cnn_extractor_type:
            rgb_feat_set = self.rgb_feat_extractor(rgb_img)
            rgb_feat = rgb_feat_set[self.rgb_cnn_extractor_stage_idx]
            rgb_feat = self.one_by_one_conv(rgb_feat)
        elif 'hrnet' in self.rgb_cnn_extractor_type:
            rgb_feat_set = self.rgb_feat_extractor(rgb_img)
            rgb_feat = rgb_feat_set[self.rgb_cnn_extractor_stage_idx]
            rgb_feat = self.one_by_one_conv(rgb_feat)
        elif self.rgb_cnn_extractor_type == 'patch':
            rgb_feat_patch = self.rgb_feat_extractor(rgb_img)
        elif self.rgb_cnn_extractor_type == 'no_use':
            pass
        else:
            rgb_feat = self.rgb_feat_extractor(rgb_img)

        rgb_feat_channel = rgb_feat.shape[1]
        rgb_feat_view = rgb_feat.view(self.batch_size, 1, rgb_feat_channel, self.down_height, self.down_width)
        rgb_feat_expand = rgb_feat_view.expand(self.batch_size, people_num, rgb_feat_channel, self.down_height, self.down_width)

        head_info_params_view = head_info_params.view(self.batch_size, people_num, rgb_feat_channel, 1, 1)
        head_info_params_expand = head_info_params_view.expand(self.batch_size, people_num, rgb_feat_channel, self.down_height, self.down_width)

        rgb_and_head_feat = head_info_params_expand * rgb_feat_expand
        rgb_and_head_feat = rgb_and_head_feat.view(self.batch_size*people_num, rgb_feat_channel, self.down_height, self.down_width)

        distance_dist = self.attention_generator(rgb_and_head_feat)
        distance_dist = distance_dist.view(self.batch_size, people_num, self.resize_height, self.resize_width)
        distance_dist = distance_dist * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]

        angle_dist = torch.exp(-torch.pow(theta_x_y, 2)/(2* 0.5 ** 2))
        angle_dist = angle_dist * (torch.sum(head_feature, dim=2) != 0)[:, :, None, None]

        # calc saliency map per person
        if self.use_dynamic_angle and self.use_dynamic_distance:
            # choose fusion type
            if self.angle_distance_fusion == 'mult':
                x_mid = (angle_dist * distance_dist)
            elif self.angle_distance_fusion == 'max':
                x_mid = torch.max(angle_dist, distance_dist)
            elif self.angle_distance_fusion == 'mean':
                x_mid = (angle_dist + distance_dist) * 0.5
        elif self.use_dynamic_angle:
            x_mid = angle_dist
        elif self.use_dynamic_distance:
            x_mid = distance_dist
        else:
            print('please employ correct distributions fusion')
            sys.exit()

        # people aggregation
        no_pad_idx = torch.sum((torch.sum(head_feature, dim=2) != 0), dim=1)[:, None, None, None]
        no_pad_idx = torch.where(no_pad_idx==0, no_pad_idx+1, no_pad_idx)
        x_mid_mean = torch.sum(x_mid, dim=1)[:, None, :, :] / no_pad_idx
        x_final = torch.sum(x_mid, dim=1)[:, None, :, :] / no_pad_idx
        x_mid_mean = x_mid_mean[:, 0, :, :]
        x_final = x_final[:, 0, :, :]

        # concat two features to return
        head_tensor = torch.cat([head_vector], dim=2)

        # pack return values
        data = {}
        data['img_pred'] = x_final
        data['img_mid_pred'] = x_mid
        data['img_mid_mean_pred'] = x_mid_mean
        data['head_tensor'] = head_tensor
        data['angle_dist'] = angle_dist
        data['distance_dist'] = distance_dist
        # data['trans_att_people_rgb'] = trans_att_people_rgb
        data['trans_att_people_people'] = trans_att_people_people
        # data['rgb_people_feat_all'] = rgb_people_feat_all
        data['head_info_params'] = head_info_params

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

        # triplet loss
        rgb_people_feat_all = out['head_info_params']
        head_info_params = out['head_info_params']
        distance_dist = out['distance_dist']
        head_feature = inp['head_feature']

        # switch loss coeficient
        if self.use_e_map_loss:
            loss_map_gaze_coef = 1
        else:
            loss_map_gaze_coef = 0

        if self.use_e_att_loss:
            loss_map_coef = 1
        else:
            loss_map_coef = 0

        if self.use_each_e_map_loss:
            loss_map_gaze_each_coef = 1
        else:
            loss_map_gaze_each_coef = 0

        batch_size, people_num, img_height, img_width = img_mid_pred.shape

        # calculate mid gaze map loss
        img_mid_mean_gt = torch.sum(img_gt, dim=1)/torch.sum(att_inside_flag, dim=-1)[:, None, None]
        loss_map_gaze = self.loss_func_joint_attention(img_mid_mean_pred.float(), img_mid_mean_gt.float())
        loss_map_gaze = loss_map_gaze_coef * loss_map_gaze

        # calculate final map loss
        img_gt_att = torch.sum(img_gt, dim=1)/torch.sum(att_inside_flag, dim=-1)[:, None, None]
        loss_map = self.loss_func_joint_attention(img_pred.float(), img_gt_att.float())
        loss_map = loss_map_coef * loss_map

        # calculate each map loss
        img_mid_pred = img_mid_pred * att_inside_flag[:, :, None, None]
        img_mid_gt = img_gt * att_inside_flag[:, :, None, None]
        loss_map_gaze_each = self.loss_func_joint_attention(img_mid_pred.float(), img_mid_gt.float())
        loss_map_gaze_each = loss_map_gaze_each_coef * loss_map_gaze_each

        loss_set = {}
        loss_set['loss_map_gaze'] = loss_map_gaze
        loss_set['loss_map'] = loss_map
        loss_set['loss_map_gaze_each'] = loss_map_gaze_each

        return loss_set

class PositionalEncoding2D_RGB(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding2D_RGB, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        batch_size, person_num, _ = tensor.shape
        pos_x, pos_y = tensor[:, :, 0], tensor[:, :, 1]
        sin_inp_x = torch.einsum("ki,j->kij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("ki,j->kij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((batch_size, person_num, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y
        return emb

class PositionalEmbeddingGenerator(nn.Module):
    def __init__(self, h, w, dim, pos_embedding_type):
        super().__init__()

        self.pos_embedding_type = pos_embedding_type
        self.num_patches = w * h

        self._make_position_embedding(w, h, dim, pos_embedding_type)

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