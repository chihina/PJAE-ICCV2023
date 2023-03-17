import os
import numpy as np
import glob
import sys

dataset_dir = 'data/VideoCoAtt_Dataset'

# joint attetntion detection analysis
ja_cnt_dic = {}
ja_cnt_dic[0] = 349468
ja_cnt_dic[1] = 139348
ja_cnt_dic[2] = 3284
all_ja_cnt = sum(ja_cnt_dic.values())
no_ja_cnt = ja_cnt_dic[0]
print('Joint attention exsistance ratio')
print(f'{no_ja_cnt/all_ja_cnt}={no_ja_cnt}/{all_ja_cnt}')

# joint attetion size analysis
img_id_list_ja = []
img_id_list_all = []
for ann_path in glob.glob(os.path.join(dataset_dir, 'annotations', 'test', '*.txt')):
    with open(ann_path, 'r') as f:
        ann_lines = f.readlines()

    vid_id = int(ann_path.split('/')[-1].split('.')[0])
    for line in ann_lines:
        co_id, img_id = map(float, line.strip().split()[0:2])

        data_id = f'{vid_id}_{img_id}'
        img_id_list_ja.append(data_id)

for img_path in glob.glob(os.path.join(dataset_dir, 'images_nk', 'test', '*', '*.jpg')):
    img_id_list_all.append(img_path)

print(len(img_id_list_ja))
print(len(img_id_list_all))
print(len(img_id_list_ja)/len(img_id_list_all))
sys.exit()
