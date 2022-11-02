from random import sample
import numpy as np
import pandas as pd
import os

# read a csv file
selection_csv_path = os.path.join('results', 'volleyball', 'volleyball-selection', 'selection.csv')
df = pd.read_csv(selection_csv_path, index_col=0)
df['ours'] = df['volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_psfix_fusion']
df['isa'] = df['volleyball-isa_bbox_GT_gaze_GT_act_GT']
df['davt'] = df['volleyball-dual-mid_p_p_field_middle_p_s_davt_bbox_GT_gaze_GT_act_GT_p_s_only']
df['isa-ours'] = df['isa']-df['ours']
df['davt-ours'] = df['davt']-df['ours']
print(df)

# get data ids by distance conditions
dist_thr = 130
sampled_data = df[(df['isa-ours']>dist_thr) & (df['davt-ours']>dist_thr)]
print(sampled_data['ours'])
print(sampled_data['isa'])
print(sampled_data['davt'])
print(sampled_data.index)