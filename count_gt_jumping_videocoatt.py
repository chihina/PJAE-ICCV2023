import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def count_gt_jumping(ja_ann, seq_num):
    ja_ann_frames = ja_ann[seq_num].keys()
    # print(f'=====Seq:{seq_num} Frames:{len(ja_ann_frames)}=====')

    jumping_count = 0
    diminishing_count = 0
    for iter, frame_id in enumerate(sorted(ja_ann_frames)):
        ja_ids = list(ja_ann[seq_num][frame_id].keys())
        if len(ja_ids) != 1:
            continue

        # obtain ja points
        ja_bbox = np.array(ja_ann[seq_num][frame_id][ja_ids[0]])
        ja_bbox_x_center = (ja_bbox[0] + ja_bbox[2]) / 2
        ja_bbox_y_center = (ja_bbox[1] + ja_bbox[3]) / 2
        ja_bbox_center = np.array([ja_bbox_x_center, ja_bbox_y_center])
        
        # set initial values
        if iter == 0:
            ja_bbox_center_prev = ja_bbox_center
            frame_id_prev = frame_id

        ja_dist = np.linalg.norm(ja_bbox_center - ja_bbox_center_prev)
        ja_dist_thre = ja_dist > 50
        frame_continous = (frame_id - frame_id_prev) == 1
        if ja_dist_thre and frame_continous:
            # print(f'Seq:{seq_num} Frame:{frame_id} {ja_dist}')
            jumping_count += 1
        elif ja_dist_thre and not frame_continous:
            # print(f'Seq:{seq_num} Frame:{frame_id} {ja_dist}')
            diminishing_count += 1

        # update previous values
        ja_bbox_center_prev = ja_bbox_center
        frame_id_prev = frame_id

    return jumping_count, diminishing_count

# load annotations
dataset_dir = os.path.join('data', 'VideoCoAtt_Dataset', 'annotations')
ja_ann = {}
for data_type in os.listdir(dataset_dir):
    data_type_dir = os.path.join(dataset_dir, data_type)
    if not data_type in ja_ann.keys():
        ja_ann[data_type] = {}
    for file in os.listdir(data_type_dir):
        file_path = os.path.join(data_type_dir, file)
        seq_num = int(file.split('.')[0])
        with open(file_path, 'r') as f:
            data = [x.strip().split() for x in f.readlines()]
        ja_ann[data_type][seq_num] = {}
        for i in range(len(data)):
            ja_id, frame_id = int(data[i][0]), int(data[i][1])
            ja_ann[data_type][seq_num][frame_id] = {}
            ja_ann[data_type][seq_num][frame_id][ja_id] = [int(x) for x in data[i][2:6]]

# count jumping
jumping_count_list = []
diminishing_count_list = []
data_type_list = ja_ann.keys()
for data_type in data_type_list:
    ja_ann_data_type = ja_ann[data_type]
    seq_num_list = ja_ann_data_type.keys()
    for seq_num in seq_num_list:
        jumping_count, diminishing_count = count_gt_jumping(ja_ann_data_type, seq_num)
        print(f'Seq:{seq_num} Jumping Count:{jumping_count} Diminishing Count:{diminishing_count}')
        jumping_count_list.append(jumping_count)
        diminishing_count_list.append(diminishing_count)

plt.figure()
plt.hist(jumping_count_list)
plt.xticks(np.arange(0, 20, 1))
plt.savefig('count_gt_jumping_videocoatt_jumping.png')

plt.figure()
plt.hist(diminishing_count_list)
plt.xticks(np.arange(0, 20, 1))
plt.savefig('count_gt_jumping_videocoatt_diminishing.png')