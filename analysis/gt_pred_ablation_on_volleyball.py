import json
import os
import sys
import pandas as pd
import numpy as np


saved_result_dir = os.path.join('results', 'volleyball')

# define analyze model type
analyze_name = 'volleyball-dual-mid_p_p_field_middle_p_s_cnn_loss_weight_1_1_1_simple_average'

# define test data type
test_data_type_list = []
test_data_type_list.append('bbox_GT_gaze_GT_act_GT_blur_False')
test_data_type_list.append('bbox_GT_gaze_PRED_act_GT_blur_False')
test_data_type_list.append('bbox_GT_gaze_GT_act_PRED_blur_False')
test_data_type_list.append('bbox_GT_gaze_PRED_act_PRED_blur_False')
test_data_type_list.append('bbox_PRED_gaze_PRED_act_PRED_blur_False')

eval_results_list = []
for test_data_type in test_data_type_list:
    print(f'==={test_data_type}===')
    json_file_path = os.path.join(saved_result_dir, analyze_name, 'eval_results', test_data_type, 'eval_results.json')
    with open(json_file_path, 'r') as f:
        eval_results_dic = json.load(f)
    eval_results_list.append(list(eval_results_dic.values()))
    eval_metrics_list = list(eval_results_dic.keys())

eval_results_array = np.array(eval_results_list)
df_eval_results = pd.DataFrame(eval_results_array, test_data_type_list, eval_metrics_list)
save_csv_file_path = os.path.join(saved_result_dir, f'gt_pred_ablation_{analyze_name}.csv')
df_eval_results.to_csv(save_csv_file_path)