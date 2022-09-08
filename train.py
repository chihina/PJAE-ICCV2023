# deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# general module
import argparse
import sys
import os
import shutil
import yaml
import numpy as np
from addict import Dict
import wandb
from tqdm import tqdm
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore") 

# original module
from dataset.dataset_selector import dataset_generator
from models.model_selector import model_generator

parser = argparse.ArgumentParser(description="parameters for training")
parser.add_argument("config", type=str, help="configuration yaml file path")
args = parser.parse_args()
cfg = Dict(yaml.safe_load(open(args.config)))
print(cfg)

def process_epoch(epoch, data_set, mode):
    data_length = len(data_set)
    epoch_loss_dic = {}

    if mode == 'train':
        model_head.train()
        model_attention.train()
        model_saliency.train()
        if cfg.exp_params.freeze_head_pose_estimator:
            model_head.eval()
        if cfg.exp_params.freeze_saliency_extractor:
            model_saliency.eval()
    else:
        model_head.eval()
        model_attention.eval()
        model_saliency.eval()

    for iteration, batch in enumerate(data_set, 1):
        # init graph
        if not cfg.exp_params.freeze_head_pose_estimator:
            optimizer_head.zero_grad()
        if not cfg.exp_params.freeze_saliency_extractor:
            optimizer_saliency.zero_grad()
        optimizer_attention.zero_grad()

        # init heatmaps
        cfg.exp_set.batch_size, num_people = batch['head_img'].shape[0:2]
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

        # move data into gpu
        if cuda:
            for key, val in batch.items():
                if key != 'rgb_path':
                    batch[key] = Variable(val).cuda(gpus_list[0])

        head_feature = batch['head_feature']
        if cfg.model_params.use_position:
            input_feature = head_feature.clone() 
        else:
            input_feature = head_feature.clone()
            input_feature[:, :, :2] = input_feature[:, :, :2] * 0
        batch['input_feature'] = input_feature

        # head pose estimation
        out_head = model_head(batch)
        batch['head_img_extract'] = out_head['head_img_extract']

        if cfg.exp_params.use_gt_gaze:
            batch['head_vector'] = batch['head_vector_gt']
        else:
            batch['head_vector'] = out_head['head_vector']

        # change position inputs
        if cfg.model_params.use_gaze:
            batch['input_gaze'] = batch['head_vector'].clone() 
        else:
            batch['input_gaze'] = batch['head_vector'].clone() * 0

        # scene feature extraction
        out_scene_feat = model_saliency(batch)
        batch = {**batch, **out_scene_feat}

        # joint attention estimation
        out_attention = model_attention(batch)

        loss_set_head = model_head.calc_loss(batch, out_head)
        loss_set_attention = model_attention.calc_loss(batch, out_attention, cfg)
        loss_set = {**loss_set_head, **loss_set_attention}

        # accumulate all loss
        for loss_idx, loss_val in enumerate(loss_set.values()):
            if loss_idx == 0:
                loss = loss_val
            else:
                loss = loss + loss_val
        loss_set['loss'] = loss

        if mode == 'train':
            loss.backward()

            if not cfg.exp_params.freeze_head_pose_estimator:
                optimizer_head.step()
            if not cfg.exp_params.freeze_saliency_extractor:
                optimizer_saliency.step()
            optimizer_attention.step()

        for loss_name, loss_val in loss_set.items():
            if iteration == 1:
                epoch_loss_dic[f'epoch_{loss_name}'] = loss_val.item()
            else:
                epoch_loss_dic[f'epoch_{loss_name}'] += loss_val.item()

        # print iteration log
        print(f"{mode} Epoch: [{epoch}/{cfg.exp_params.nEpochs}] [{iteration}/{data_length}]  loss: {loss.item():.8f}")

    # print epoch log
    print(f"===> Epoch {mode} Complete: Avg {mode} Loss: {epoch_loss_dic['epoch_loss'] / data_length}")

    # logging for wandb
    if cfg.exp_set.wandb_log:
        for loss_name, loss_val in epoch_loss_dic.items():
            wandb.log({f"{mode} {loss_name}": loss_val / data_length}, step=epoch)

    average_loss = epoch_loss_dic['epoch_loss'] / data_length
    return average_loss

# save a model in a check point
def checkpoint(epoch):
    weights_save_dir = os.path.join(cfg.exp_set.save_folder, cfg.data.name, cfg.exp_set.wandb_name)
    model_head_out_path = os.path.join(weights_save_dir, f"model_head_epoch_{epoch}.pth")
    model_saliency_out_path = os.path.join(weights_save_dir, f"model_saliency_epoch_{epoch}.pth")
    model_attention_out_path = os.path.join(weights_save_dir, f"model_gaussian_epoch_{epoch}.pth")
    torch.save(model_head.state_dict(), model_head_out_path)
    torch.save(model_saliency.state_dict(), model_saliency_out_path)
    torch.save(model_attention.state_dict(), model_attention_out_path)
    print(f"Checkpoint saved to {weights_save_dir}")

# save a model in a bast score
def best_checkpoint(epoch):
    weights_save_dir = os.path.join(cfg.exp_set.save_folder, cfg.data.name, cfg.exp_set.wandb_name)
    model_head_out_path = os.path.join(weights_save_dir, "model_head_best.pth.tar")
    model_saliency_out_path = os.path.join(weights_save_dir, "model_saliency_best.pth.tar")
    model_attention_out_path = os.path.join(weights_save_dir, "model_gaussian_best.pth.tar")
    torch.save(model_head.state_dict(), model_head_out_path)
    torch.save(model_saliency.state_dict(), model_saliency_out_path)
    torch.save(model_attention.state_dict(), model_attention_out_path)
    print(f"Best Checkpoint saved to {weights_save_dir}")

def load_multi_gpu_models(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

# fix some seeds for reproduction
np.random.seed(cfg.exp_set.seed_num)
torch.manual_seed(cfg.exp_set.seed_num)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms=True

# save setting files
saved_weights_dir = os.path.join(cfg.exp_set.save_folder, cfg.data.name, cfg.exp_set.wandb_name)
if not os.path.exists(saved_weights_dir):
    os.mkdir(saved_weights_dir)
shutil.copy(args.config, os.path.join(saved_weights_dir, args.config.split('/')[-1]))

print("===> Setting gpu numbers")
cuda = cfg.exp_set.gpu_mode
gpus_list = range(cfg.exp_set.gpu_start, cfg.exp_set.gpu_finish+1)

print("===> Loading datasets")
train_set = dataset_generator(cfg, 'train')

training_data_loader = DataLoader(dataset=train_set,
                                batch_size=cfg.exp_set.batch_size,
                                shuffle=True, 
                                num_workers=cfg.exp_set.num_workers,
                                pin_memory=False)

val_set = dataset_generator(cfg, 'valid')
validation_data_loader = DataLoader(dataset=val_set,
                                batch_size=cfg.exp_set.batch_size,
                                shuffle=False,
                                num_workers=cfg.exp_set.num_workers,
                                pin_memory=False)

print('{} Train samples found'.format(len(train_set)))
print('{} Test samples found'.format(len(val_set)))

print("===> Building model")
model_head, model_attention, model_saliency, cfg = model_generator(cfg)

if cfg.exp_params.use_pretrained_head_pose_estimator:
    print("===> Load pretrained model (head pose estimator)")
    model_name = cfg.exp_params.pretrained_head_pose_estimator_name
    model_weight_path = os.path.join(cfg.exp_params.pretrained_models_dir, cfg.data.name, model_name, "model_head_best.pth.tar")
    fixed_model_state_dict = load_multi_gpu_models(torch.load(model_weight_path,  map_location='cuda:'+str(gpus_list[0])))
    model_head.load_state_dict(fixed_model_state_dict)

if cfg.exp_params.use_pretrained_saliency_extractor:
    print("===> Load pretrained model (saliecny extractor)")
    model_name = cfg.exp_params.pretrained_saliency_extractor_name
    model_weight_path = os.path.join(cfg.exp_params.pretrained_models_dir, cfg.data.name, model_name, "model_demo.pt")
    model_saliency_dict = model_saliency.state_dict()
    pretrained_dict = torch.load(model_weight_path,  map_location='cuda:'+str(gpus_list[0]))
    pretrained_dict = pretrained_dict['model']
    model_saliency_dict.update(pretrained_dict)
    model_saliency.load_state_dict(model_saliency_dict)

if cfg.exp_params.use_pretrained_joint_attention_estimator:
    print("===> Load pretrained model (joint attention estimator)")
    model_name = cfg.pretrained_joint_attention_estimator_name
    model_weight_path = os.path.join(cfg.pretrained_model_dir, cfg.data.name, model_name, "model_gaussian_best.pth.tar")
    fixed_model_state_dict = load_multi_gpu_models(torch.load(model_weight_path,  map_location='cuda:'+str(gpus_list[0])))
    model_attention.load_state_dict(fixed_model_state_dict)

# scheduling learning rate 
optimizer_head = optim.Adam(model_head.parameters(), lr=cfg.exp_params.lr)
optimizer_attention = optim.Adam(model_attention.parameters(), lr=cfg.exp_params.lr)
optimizer_saliency = optim.Adam(model_saliency.parameters(), lr=cfg.exp_params.lr)

scheduler_head = optim.lr_scheduler.MultiStepLR(optimizer_head, 
                                           milestones=[i for i in range(cfg.exp_params.scheduler_start, cfg.exp_params.nEpochs, cfg.exp_params.scheduler_iter)],
                                           gamma=0.1)
scheduler_attention = optim.lr_scheduler.MultiStepLR(optimizer_attention, 
                                           milestones=[i for i in range(cfg.exp_params.scheduler_start, cfg.exp_params.nEpochs, cfg.exp_params.scheduler_iter)],
                                           gamma=0.1)
scheduler_saliency = optim.lr_scheduler.MultiStepLR(optimizer_saliency, 
                                           milestones=[i for i in range(cfg.exp_params.scheduler_start, cfg.exp_params.nEpochs, cfg.exp_params.scheduler_iter)],
                                           gamma=0.1)

if cuda:
    if (cfg.exp_set.gpu_finish - cfg.exp_set.gpu_start) >= 1:
        print("===> Use multiple GPUs")
        model_head = torch.nn.DataParallel(model_head, device_ids=gpus_list)
        model_attention = torch.nn.DataParallel(model_attention, device_ids=gpus_list)
        model_saliency = torch.nn.DataParallel(model_saliency, device_ids=gpus_list)

    else:
        print("===> Use single GPU")
    
    model_head = model_head.cuda(gpus_list[0])
    model_attention = model_attention.cuda(gpus_list[0])
    model_saliency = model_saliency.cuda(gpus_list[0])

if cfg.exp_set.wandb_log:
    print("===> Generate wandb system")
    wandb.login()
    wandb.init(project=f"config-action-aware-joint-attention-estimation-{cfg.data.name}", name=cfg.exp_set.wandb_name, config=cfg)
    wandb.watch(model_head)
    wandb.watch(model_attention)
    wandb.watch(model_saliency)

# Training
best_loss = 10000000000.0
print("===> Start training")
for epoch in range(cfg.exp_params.start_iter, cfg.exp_params.nEpochs + 1):
    _ = process_epoch(epoch, training_data_loader, 'train')

    # schedule learning rate
    scheduler_head.step()
    scheduler_attention.step()
    scheduler_saliency.step()

    current_val_loss = process_epoch(epoch, validation_data_loader, 'valid')

    if current_val_loss < best_loss:
        best_loss = current_val_loss
        best_checkpoint(epoch+1)
        print("Save Best Loss : {}".format(best_loss))

    if (epoch+1) % (cfg.exp_params.snapshots) == 0:
        checkpoint(epoch+1)