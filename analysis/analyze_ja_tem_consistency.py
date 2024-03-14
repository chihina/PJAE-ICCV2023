import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def read_csv_ann(ann_path):
    if not os.path.exists(ann_path):
        return pd.DataFrame()
    df_ann = pd.read_csv(ann_path, header=None, sep=' ')
    df_ann = df_ann.iloc[:, 1:5]

    return df_ann

dataset_dir = os.path.join('data', 'vatic_ball_annotation', 'annotation_data_sub')
program_name = os.path.basename(__file__).split('.')[0]
save_dir = os.path.join('analysis', program_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

center_idx, pad_idx = 20, 5
img_height, img_width = 720, 1280
stop_idx = 20

for ann_idx, ann_file_name in enumerate(tqdm(os.listdir(dataset_dir))):
    vid_num, seq_num = ann_file_name.split('_')[1:3]
    df_ann = read_csv_ann(os.path.join(dataset_dir, ann_file_name))
    if df_ann.shape[0] == 0:
        continue

    ja_xmid = (df_ann.iloc[:, 0] + df_ann.iloc[:, 2]) / 2
    ja_ymid = (df_ann.iloc[:, 1] + df_ann.iloc[:, 3]) / 2
    ja_mid = pd.concat([ja_xmid, ja_ymid], axis=1)
    ja_mid = ja_mid.iloc[center_idx-pad_idx:center_idx+pad_idx, :]
    ja_mid_diff = ja_mid.diff().fillna(0)

    save_dir_child = os.path.join(save_dir, f'{vid_num}_{seq_num}')
    if not os.path.exists(save_dir_child):
        os.makedirs(save_dir_child)

    plt.figure()
    for t in range(ja_mid.shape[0]):
        t_norm = str(t / (ja_mid.shape[0]))
        plt.plot(ja_mid.iloc[t, 0], ja_mid.iloc[t, 1], 'o', color=t_norm)
    plt.savefig(os.path.join(save_dir_child, f'move_abs.png'))
    plt.close()

    plt.figure()
    for t in range(ja_mid.shape[0]):
        t_norm = str(t / (ja_mid.shape[0]))
        plt.plot(ja_mid_diff.iloc[t, 0], ja_mid_diff.iloc[t, 1], 'o', color=t_norm)
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.savefig(os.path.join(save_dir_child, f'move_diff.png'))
    plt.close()

    if ann_idx > stop_idx:
        break