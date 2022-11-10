import json
import os
import sys
import pandas as pd
import numpy as np
import glob

saved_result_dir = os.path.join('results', 'videocoatt')

# define model names
model_name_list = []
model_name_list.append('ISA')
model_name_list.append('DAVT')
model_name_list.append('HGTD')
model_name_list.append('Ours')

# define test data type
test_data_type_list = []
test_data_type_list.append('bbox_det_gaze_False')
test_data_type_list.append('bbox_det_gaze_False')
test_data_type_list.append('bbox_gt_gaze_True')

# define training modality type
train_mode_list = []
train_mode_list.append('Pr')
train_mode_list.append('GT')
train_mode_list.append('GT')

# define analize model names
analyze_name_list_dic = {}

# (Train:Test = Pr:Pr)
analyze_name_list = []
analyze_name_list.append('videocoatt-isa_bbox_PRED_gaze_PRED')
# analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_freeze')
analyze_name_list.append('videoattentiontarget-only_davt_PRED')
# analyze_name_list.append('videoattentiontarget-hgt-high')
analyze_name_list.append('videoattentiontarget-hgt-hgt_bbox_PRED')
analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_scalar_weight_fix')
analyze_name_list_dic[0] = analyze_name_list

# (Train:Test = GT:Pr)
analyze_name_list = []
analyze_name_list.append('videocoatt-isa_bbox_GT_gaze_GT')
analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_freeze')
analyze_name_list.append('videoattentiontarget-hgt-high')
analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_scalar_weight_fix_token_only_GT')
analyze_name_list_dic[1] = analyze_name_list

# (Train:Test = GT:GT)
analyze_name_list = []
analyze_name_list.append('videocoatt-isa_bbox_GT_gaze_GT')
analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_freeze')
analyze_name_list.append('videoattentiontarget-hgt-high')
analyze_name_list.append('videocoatt-p_p_field_deep_p_s_davt_scalar_weight_fix_token_only_GT')
analyze_name_list_dic[2] = analyze_name_list

for data_type_idx, analyze_name_list in analyze_name_list_dic.items():
    eval_results_list = []
    test_data_type = test_data_type_list[data_type_idx]
    print(f'==={test_data_type}===')
    for analyze_idx, analyze_name in enumerate(analyze_name_list):
        model_name = model_name_list[analyze_idx]
        print(model_name, analyze_name)
        json_file_path = os.path.join(saved_result_dir, analyze_name, 'eval_results', test_data_type, 'eval_results.json')
        with open(json_file_path, 'r') as f:
            eval_results_dic = json.load(f)

            eval_results_dic_update = {}
            if model_name in ['ISA', 'HGTD']:
                eval_results_dic_update['Dist(x)'] = eval_results_dic['l2_dist_x']
                eval_results_dic_update['Dist(y)'] = eval_results_dic['l2_dist_y']
                eval_results_dic_update['Dist(euc)'] = eval_results_dic['l2_dist_euc']
                for i in range(20):
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
                for i in range(20):
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
                for i in range(20):
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
    save_csv_file_path = os.path.join(saved_result_dir, f'comparision_on_videocoatt_{train_mode_list[data_type_idx]}_{test_data_type}.csv')
    df_eval_results.to_csv(save_csv_file_path)