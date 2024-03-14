import json
import os
import sys
import pandas as pd
import numpy as np


saved_result_dir = os.path.join('results', 'videocoatt')

# define analyze model type
analyze_name = 'videocoatt-p_p_field_deep_p_s_davt_scalar_weight_fix_token_only_GT'

# define test data type
test_data_type_list = []
test_data_type_list.append('bbox_gt_gaze_True_thresh_f_score')
test_data_type_list.append('bbox_gt_gaze_False_thresh_f_score')
test_data_type_list.append('bbox_det_gaze_False_thresh_f_score')

# define model name
model_name_list = []
model_name_list.append('Ours (p=GT, g=GT)')
model_name_list.append('Ours (p=GT, g=Pr)')
model_name_list.append('Ours (p=Pr, g=Pr)')

eval_results_list = []
for test_data_type in test_data_type_list:
    print(f'==={test_data_type}===')
    json_file_path = os.path.join(saved_result_dir, analyze_name, 'eval_results', test_data_type, 'eval_results.json')
    with open(json_file_path, 'r') as f:
        eval_results_dic = json.load(f)

    eval_results_dic_update = {}
    eval_results_dic_update['Dist final (x)'] = eval_results_dic['l2_dist_x_final']
    eval_results_dic_update['Dist final (y)'] = eval_results_dic['l2_dist_y_final']
    eval_results_dic_update['Dist final (euc)'] = eval_results_dic['l2_dist_euc_final']
    for i in range(20):
        thr = i*10
        eval_results_dic_update[f'Det final (Thr={thr})'] = eval_results_dic[f'Det final (Thr={thr})']
    eval_results_dic_update['Accuracy final'] = eval_results_dic['accuracy final']
    eval_results_dic_update['F-score final'] = eval_results_dic['f1 final']
    eval_results_dic_update['AUC final'] = eval_results_dic['auc final']

    eval_results_list.append(list(eval_results_dic_update.values()))
    eval_metrics_list = list(eval_results_dic_update.keys())

eval_results_array = np.array(eval_results_list)
df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
save_csv_file_path = os.path.join(saved_result_dir, f'gt_pred_ablation_{analyze_name}_videocoatt.csv')
df_eval_results.to_csv(save_csv_file_path)