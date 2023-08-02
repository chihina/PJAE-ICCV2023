from random import sample
import numpy as np
import pandas as pd
import os

# read a csv file (volleyball)
# model_name = 'volleyball-dual-mid_p_p_field_middle_bbox_GT_gaze_GT_act_GT'
# selection_csv_path = os.path.join('results', 'volleyball', model_name, 'iccv_rebuttal_selection.csv')

# read a csv file (videocoatt)
model_name = 'videocoatt-dual-people_field_middle_token_only_bbox_GT_gaze_GT'
model_name = 'videocoatt-p_p_field_deep_p_s_davt_scalar_weight_fix_token_only_GT'
selection_csv_path = os.path.join('results', 'videocoatt', model_name, 'iccv_rebuttal_selection.csv')

df = pd.read_csv(selection_csv_path, index_col=0)
print(df)

# get data ids by distance conditions
dist_thr = 200
# sampled_data = df[df['Dist']>dist_thr]['Dist']
sampled_data = df[df[model_name]>dist_thr][model_name]
print(sampled_data)
print(sampled_data.index)