import os
import numpy as np
import glob

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
size_list = []
for ann_path in glob.glob(os.path.join(dataset_dir, 'annotations', '*', '*.txt')):
    with open(ann_path, 'r') as f:
        ann_lines = f.readlines()
    for line in ann_lines:
        x_min, y_min, x_max, y_max = map(float, line.strip().split()[2:6])
        width = (x_max-x_min)
        height = (y_max-y_min)
        size = (width+height)//4
        size_list.append(size)

print('Joint attention size')
print(np.mean(np.array(size_list)))