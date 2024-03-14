import json
import os
import sys
import pandas as pd
import numpy as np


saved_result_dir = os.path.join('results', 'videocoatt')

# define analyze model type
analyze_name_list = []
analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_scalar_weight_fix_token_only_GT')
# analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_scalar_weight_fix_token_only')

# define ablate type
analyze_name_ablation_list = []
analyze_name_ablation_list.append('_wo_position')
analyze_name_ablation_list.append('_wo_gaze')
analyze_name_ablation_list.append('_wo_p_p')
analyze_name_ablation_list.append('_wo_p_s')
analyze_name_ablation_list.append('')

# define model name
model_name_list = []
model_name_list.append('Ours w/o p')
model_name_list.append('Ours w/o g')
model_name_list.append('Ours w/o branch (a)')
model_name_list.append('Ours w/o branch (b)')
model_name_list.append('Ours')

# define test data type
test_data_type_list = []
test_data_type_list.append('bbox_gt_gaze_True')
# test_data_type_list.append('bbox_det_gaze_False')
for test_data_type in test_data_type_list:
    print(f'==={test_data_type}===')
    for analyze_name in analyze_name_list:
        eval_results_list = []
        analyze_name_type = analyze_name
        for ablation_name in analyze_name_ablation_list:

            model_name = f'{analyze_name_type}{ablation_name}'        
            json_file_path = os.path.join(saved_result_dir, model_name, 'eval_results', test_data_type, 'eval_results.json')

            with open(json_file_path, 'r') as f:
                eval_results_dic = json.load(f)

            eval_results_dic_update = {}
            if ablation_name == '_wo_p_p':
                eval_results_dic_update['Dist p-p (euc)'] = eval_results_dic['l2_dist_euc_p_p']
                eval_results_dic_update['Dist p-s (euc)'] = eval_results_dic['l2_dist_euc_p_s']
                eval_results_dic_update['Dist final (euc)'] = eval_results_dic['l2_dist_euc_p_s']
                for i in range(20):
                    thr = i*10
                    eval_results_dic_update[f'Det p-p (Thr={thr})'] = eval_results_dic[f'Det p-p (Thr={thr})']
                for i in range(20):
                    thr = i*10
                    eval_results_dic_update[f'Det p-s (Thr={thr})'] = eval_results_dic[f'Det p-s (Thr={thr})']
                for i in range(20):
                    thr = i*10
                    eval_results_dic_update[f'Det final (Thr={thr})'] = eval_results_dic[f'Det p-s (Thr={thr})']
                eval_results_dic_update['Accuracy p-p'] = eval_results_dic['accuracy p-p']
                eval_results_dic_update['Accuracy p-s'] = eval_results_dic['accuracy p-s']
                eval_results_dic_update['Accuracy final'] = eval_results_dic['accuracy p-s']
                eval_results_dic_update['F-score p-p'] = eval_results_dic['f1 p-p']
                eval_results_dic_update['F-score p-s'] = eval_results_dic['f1 p-s']
                eval_results_dic_update['F-score final'] = eval_results_dic['f1 p-s']
                eval_results_dic_update['AUC p-p'] = eval_results_dic['auc p-p']
                eval_results_dic_update['AUC p-s'] = eval_results_dic['auc p-s']
                eval_results_dic_update['AUC final'] = eval_results_dic['auc p-s']
            else:
                eval_results_dic_update['Dist p-p (euc)'] = eval_results_dic['l2_dist_euc_p_p']
                eval_results_dic_update['Dist p-s (euc)'] = eval_results_dic['l2_dist_euc_p_s']
                eval_results_dic_update['Dist final (euc)'] = eval_results_dic['l2_dist_euc_final']
                for i in range(20):
                    thr = i*10
                    eval_results_dic_update[f'Det p-p (Thr={thr})'] = eval_results_dic[f'Det p-p (Thr={thr})']
                for i in range(20):
                    thr = i*10
                    eval_results_dic_update[f'Det p-s (Thr={thr})'] = eval_results_dic[f'Det p-s (Thr={thr})']
                for i in range(20):
                    thr = i*10
                    eval_results_dic_update[f'Det final (Thr={thr})'] = eval_results_dic[f'Det final (Thr={thr})']
                eval_results_dic_update['Accuracy p-p'] = eval_results_dic['accuracy p-p']
                eval_results_dic_update['Accuracy p-s'] = eval_results_dic['accuracy p-s']
                eval_results_dic_update['Accuracy final'] = eval_results_dic['accuracy final']
                eval_results_dic_update['F-score p-p'] = eval_results_dic['f1 p-p']
                eval_results_dic_update['F-score p-s'] = eval_results_dic['f1 p-s']
                eval_results_dic_update['F-score final'] = eval_results_dic['f1 final']
                eval_results_dic_update['AUC p-p'] = eval_results_dic['auc p-p']
                eval_results_dic_update['AUC p-s'] = eval_results_dic['auc p-s']
                eval_results_dic_update['AUC final'] = eval_results_dic['auc final']

            eval_results_list.append(list(eval_results_dic_update.values()))
            eval_metrics_list = list(eval_results_dic_update.keys())

        eval_results_array = np.array(eval_results_list)
        df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
        save_csv_file_path = os.path.join(saved_result_dir, f'ablation_{analyze_name}_{test_data_type}_videocoatt.csv')
        df_eval_results.to_csv(save_csv_file_path)