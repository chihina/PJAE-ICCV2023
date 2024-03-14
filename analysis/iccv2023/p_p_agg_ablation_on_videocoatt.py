import json
import os
import sys
import pandas as pd
import numpy as np


saved_result_dir = os.path.join('results', 'videocoatt')

# define analyze model type
analyze_name_list = []
analyze_name_list.append('videocoatt-dual-people_field_middle')

# define ablate type
analyze_name_ablation_list = []
analyze_name_ablation_list.append('_ind_only')
analyze_name_ablation_list.append('_token_only')
# analyze_name_ablation_list.append('')
analyze_name_ablation_list.append('_ind_and_token_ind_based')
# analyze_name_ablation_list.append('_ind_and_token_token_based')

# define model names
model_name_list = []
model_name_list.append('Ind only')
model_name_list.append('Token only')
model_name_list.append('Ind and Token (ind-based)')
# model_name_list.append('Ind and Token (token-based)')

# define test data type
test_data_type_list = []
test_data_type_list.append('bbox_gt_gaze_True')
# test_data_type_list.append('bbox_det_gaze_False')
for test_data_type in test_data_type_list:
    print(f'==={test_data_type}===')
    for analyze_name in analyze_name_list:
        eval_results_list = []
        for ablation_name in analyze_name_ablation_list:
                
            if test_data_type == 'bbox_gt_gaze_True':
                model_name = f'{analyze_name}{ablation_name}_bbox_GT_gaze_GT'
            else:
                model_name = f'{analyze_name}{ablation_name}'

            json_file_path = os.path.join(saved_result_dir, model_name, 'eval_results', test_data_type, 'eval_results.json')
            with open(json_file_path, 'r') as f:
                eval_results_dic = json.load(f)

            eval_results_dic_update = {}
            eval_results_dic_update['Dist final (x)'] = eval_results_dic['l2_dist_x_p_p']
            eval_results_dic_update['Dist final (y)'] = eval_results_dic['l2_dist_y_p_p']
            eval_results_dic_update['Dist final (euc)'] = eval_results_dic['l2_dist_euc_p_p']
            for i in range(20):
                thr = i*10
                eval_results_dic_update[f'Det final (Thr={thr})'] = eval_results_dic[f'Det p-p (Thr={thr})']
            eval_results_dic_update['Accuracy final'] = eval_results_dic['accuracy final']
            eval_results_dic_update['F-score final'] = eval_results_dic['f1 final']
            eval_results_dic_update['AUC final'] = eval_results_dic['auc final']

            eval_results_list.append(list(eval_results_dic_update.values()))
            eval_metrics_list = list(eval_results_dic_update.keys())

        eval_results_array = np.array(eval_results_list)
        df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
        save_csv_file_path = os.path.join(saved_result_dir, f'p_p_agg_ablation_videocoatt_{test_data_type}.csv')
        df_eval_results.to_csv(save_csv_file_path)