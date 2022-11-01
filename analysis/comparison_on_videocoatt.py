import json
import os
import sys
import pandas as pd
import numpy as np
import glob

saved_result_dir = os.path.join('results', 'videocoatt')

analyze_name_list = []
analyze_name_list.append('videocoatt-isa')
analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_freeze')
analyze_name_list.append('videoattentiontarget-hgt-high')
analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_simple_average')

# define model names
model_name_list = []
model_name_list.append('ISA')
model_name_list.append('DAVT')
model_name_list.append('HGTD')
model_name_list.append('Ours')

eval_results_list = []
for model_name, analyze_name in zip(model_name_list, analyze_name_list):
    print(model_name)

    json_file_path = glob.glob(os.path.join(saved_result_dir, analyze_name, 'eval_results', '*', 'eval_results.json'))[0]
    with open(json_file_path, 'r') as f:
        eval_results_dic = json.load(f)

    eval_results_dic_update = {}
    if model_name in ['ISA', 'HGTD']:
        eval_results_dic_update['Dist(x)'] = eval_results_dic['l2_dist_x']
        eval_results_dic_update['Dist(y)'] = eval_results_dic['l2_dist_y']
        eval_results_dic_update['Dist(euc)'] = eval_results_dic['l2_dist_euc']
        for i in range(10):
            thr = i*10
            eval_results_dic_update[f'Det(Thr={thr})'] = eval_results_dic[f'Det (Thr={thr})']
        eval_results_dic_update['Accuracy'] = eval_results_dic['accuracy']
        eval_results_dic_update['Precision'] = eval_results_dic['precision']
        eval_results_dic_update['Recall'] = eval_results_dic['recall']
        eval_results_dic_update['F-score'] = eval_results_dic['f1']
        eval_results_dic_update['AUC'] = eval_results_dic['auc']
    elif model_name in ['DAVT']:
        eval_results_dic_update['Dist(x)'] = eval_results_dic['l2_dist_x_p_s']
        eval_results_dic_update['Dist(y)'] = eval_results_dic['l2_dist_y_p_s']
        eval_results_dic_update['Dist(euc)'] = eval_results_dic['l2_dist_euc_p_s']
        for i in range(10):
            thr = i*10
            eval_results_dic_update[f'Det(Thr={thr})'] = eval_results_dic[f'Det p-s (Thr={thr})']
        eval_results_dic_update['Accuracy'] = eval_results_dic['accuracy p-s']
        eval_results_dic_update['Precision'] = eval_results_dic['precision p-s']
        eval_results_dic_update['Recall'] = eval_results_dic['recall p-s']
        eval_results_dic_update['F-score'] = eval_results_dic['f1 p-s']
        eval_results_dic_update['AUC'] = eval_results_dic['auc p-s']
    elif model_name in ['Ours']:
        eval_results_dic_update['Dist(x)'] = eval_results_dic['l2_dist_x_final']
        eval_results_dic_update['Dist(y)'] = eval_results_dic['l2_dist_y_final']
        eval_results_dic_update['Dist(euc)'] = eval_results_dic['l2_dist_euc_final']
        for i in range(10):
            thr = i*10
            eval_results_dic_update[f'Det(Thr={thr})'] = eval_results_dic[f'Det final (Thr={thr})']
        eval_results_dic_update['Accuracy'] = eval_results_dic['accuracy final']
        eval_results_dic_update['Precision'] = eval_results_dic['precision final']
        eval_results_dic_update['Recall'] = eval_results_dic['recall final']
        eval_results_dic_update['F-score'] = eval_results_dic['f1 final']
        eval_results_dic_update['AUC'] = eval_results_dic['auc final']

    eval_results_list.append(list(eval_results_dic_update.values()))
    eval_metrics_list = list(eval_results_dic_update.keys())

eval_results_array = np.array(eval_results_list)
df_eval_results = pd.DataFrame(eval_results_array, model_name_list, eval_metrics_list)
save_csv_file_path = os.path.join(saved_result_dir, f'comparision_on_videocoatt.csv')
df_eval_results.to_csv(save_csv_file_path)