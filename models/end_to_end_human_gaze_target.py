from os import lseek
from random import expovariate
from tokenize import triple_quoted
import  numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.nn import functional as F
import sys
import timm
import math

from models.detr_utils import build_matcher, SetCriterion

class EndToEndHumanGazeTargetTransformer(nn.Module):
    def __init__(self, cfg):
        super(EndToEndHumanGazeTargetTransformer, self).__init__()

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

        ## network
        # feature extractor
        self.rgb_embeding_dim = 256
        self.rgb_feature_extractor = timm.create_model('resnet50', features_only=True, pretrained=True)
        self.rgb_feature_one_by_one_conv = nn.Sequential(
                                            nn.Conv2d(in_channels=2048, out_channels=self.rgb_embeding_dim, kernel_size=1),
                                            nn.ReLU(),
                                            )

        self.down_scale_ratio = 4
        self.down_height = self.resize_height//self.down_scale_ratio
        self.down_width = self.resize_width//self.down_scale_ratio
        self.pe_generator_rgb = PositionalEmbeddingGenerator(self.down_height, self.down_width, self.rgb_embeding_dim, 'sine')

        # DETR
        self.detr = DETR()
        detr_state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
            map_location=self.device, check_hash=True)
        self.detr.load_state_dict(detr_state_dict)

        # instance prediction
        self.trans_dec_dim = self.rgb_embeding_dim
        self.head_location_mlp = nn.Sequential(
                nn.Linear(self.trans_dec_dim, self.trans_dec_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.trans_dec_dim, self.trans_dec_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.trans_dec_dim, 4),
                nn.Sigmoid(),
            )

        self.gaze_heatmap_mlp = nn.Sequential(
                nn.Linear(self.trans_dec_dim, self.trans_dec_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.trans_dec_dim, self.trans_dec_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.trans_dec_dim, self.trans_dec_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.trans_dec_dim, self.down_height*self.down_width),
                nn.Sigmoid(),
            )

        self.is_head_mlp = nn.Sequential(
                nn.Linear(self.trans_dec_dim, self.trans_dec_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.trans_dec_dim, 2),
                nn.Softmax(dim=-1),
            )

        self.watch_outside_mlp = nn.Sequential(
                nn.Linear(self.trans_dec_dim, self.trans_dec_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.trans_dec_dim, 2),
                nn.Softmax(dim=-1),
            )

        num_classes = 1
        matcher = build_matcher()
        weight_dict = {'loss_ce': 1, 'loss_bbox': 1}
        weight_dict['loss_giou'] = 1
        losses = ['boxes', 'is_head', 'watch_outside', 'gaze_map']
        eos_coef = 0.1
        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                eos_coef=eos_coef, losses=losses)
        self.criterion.to(self.device)

        self.gaze_map_resizer = transforms.Compose(
            [
                transforms.Resize((self.down_height, self.down_width)),
            ]
        )

    def forward(self, inp):
        # get usuful variable
        rgb_img = inp['rgb_img']
        self.batch_size, _, _, _ = rgb_img.shape

        # detr module
        decoded_feat = self.detr(rgb_img)

        # instance estimation
        head_loc_pred = self.head_location_mlp(decoded_feat)
        gaze_heatmap_pred = self.gaze_heatmap_mlp(decoded_feat)
        is_head_pred = self.is_head_mlp(decoded_feat)
        watch_outside_pred = self.watch_outside_mlp(decoded_feat)

        # pack return values
        data = {}
        data['head_loc_pred'] = head_loc_pred
        data['gaze_heatmap_pred'] = gaze_heatmap_pred
        data['is_head_pred'] = is_head_pred
        data['watch_outside_pred'] = watch_outside_pred

        return data

    def calc_loss(self, inp, out, cfg):
        # unpack data
        img_gt = inp['img_gt']
        head_bbox = inp['head_bbox']
        head_feature = inp['head_feature']
        att_inside_flag = inp['att_inside_flag']
        head_loc_pred = out['head_loc_pred']
        gaze_heatmap_pred = out['gaze_heatmap_pred']
        is_head_pred = out['is_head_pred']
        watch_outside_pred = out['watch_outside_pred']

        _, people_num, _, _ = img_gt.shape

        # pack outputs
        outputs = {}
        outputs['head_loc_pred'] = head_loc_pred
        outputs['gaze_heatmap_pred'] = gaze_heatmap_pred
        outputs['is_head_pred'] = is_head_pred
        outputs['watch_outside_pred'] = watch_outside_pred

        # get ground-truth
        head_loc_gt_no_pad = head_bbox
        gaze_heatmap_gt = img_gt
        gaze_heatmap_gt = self.gaze_map_resizer(gaze_heatmap_gt)
        gaze_heatmap_gt = gaze_heatmap_gt.view(self.batch_size, people_num, self.down_height*self.down_width)
        is_head_gt= (torch.sum(head_feature, dim=-1) != 0)
        is_head_gt = is_head_gt.view(self.batch_size, people_num, 1)
        watch_outside = att_inside_flag != 1
        watch_outside = watch_outside.view(self.batch_size, people_num, 1)
        head_loc_gt = torch.cat([head_loc_gt_no_pad], dim=1)
        gaze_heatmap_gt = torch.cat([gaze_heatmap_gt], dim=1)
        is_head_gt = torch.cat([is_head_gt], dim=1)
        watch_outside_gt = torch.cat([watch_outside], dim=1)

        # pack targets
        targets = {}
        targets['head_loc_gt'] = head_loc_gt
        targets['gaze_heatmap_gt'] = gaze_heatmap_gt
        targets['is_head_gt'] = is_head_gt
        targets['watch_outside_gt'] = watch_outside_gt

        loss_set = self.criterion(outputs, targets)

        return loss_set

class DETR(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes=91, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = models.resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        B, C, H, W = h.shape
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        pos_expand = pos.expand(W*H, B, C)
        h_trans = h.flatten(2).permute(2, 0, 1)
        query_pos_expand = self.query_pos.unsqueeze(1).expand(100, B, C)

        # propagate through the transformer
        h_out = self.transformer(pos_expand + 0.1 * h_trans, query_pos_expand).transpose(0, 1)
        h_out = h_out[:, :20, :]

        return h_out

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