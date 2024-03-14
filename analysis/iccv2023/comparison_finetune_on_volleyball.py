import json
import os
import sys
import pandas as pd
import numpy as np

saved_result_dir = os.path.join('results', 'volleyball')

# define training modality type
train_mode_list = []
train_mode_list.append('GT')
# train_mode_list.append('Pr')

# define test data type
test_data_type_list = []
test_data_type_list.append('bbox_GT_gaze_GT_act_GT_blur_False')
# test_data_type_list.append('bbox_PRED_gaze_PRED_act_PRED_blur_False')

# define analyze model type
analyze_name_list_dic = {}

# (Train:Test, GT:GT)
analyze_name_list = []
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_bbox_GT_gaze_GT_act_GT_wo_action_wo_video_tuned')
analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_bbox_GT_gaze_GT_act_GT_wo_action_w_video_tuned_lr_0001')
analyze_name_list_dic[0] = analyze_name_list

# (Train:Test, Pr:Pr)
# analyze_name_list = []
# analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_bbox_PRED_gaze_PRED_act_PRED_wo_action_video_tuned')
# analyze_name_list.append('volleyball-dual-mid_p_p_field_middle_bbox_PRED_gaze_PRED_act_PRED_wo_action')
# analyze_name_list_dic[1] = analyze_name_list

# define model names
model_name_list = []
model_name_list.append('Ours w/o finetune')
model_name_list.append('Ours w/ finetune')

epoch_sum = 105
epoch_div = 5
for data_idx, analyze_name_list in analyze_name_list_dic.items():
    train_mode = train_mode_list[data_idx]
    test_data_type = test_data_type_list[data_idx]
    print(f'==={train_mode}:{test_data_type}===')

    model_wo_finetune = analyze_name_list[0]
    model_w_finetune = analyze_name_list[1]
    l2_dist_array = np.zeros((2, (epoch_sum//epoch_div)-1))
    
    epoch_num_list = [epoch_num for epoch_num in range(epoch_div, epoch_sum, epoch_div)]
    for epoch_idx, epoch_num in enumerate(epoch_num_list):
        json_file_path_wo_finetune = os.path.join(saved_result_dir, model_wo_finetune, 'eval_results', test_data_type, f'epoch_{epoch_num}', 'eval_results.json')
        json_file_path_w_finetune = os.path.join(saved_result_dir, model_w_finetune, 'eval_results', test_data_type, f'epoch_{epoch_num}', 'eval_results.json')
        with open(json_file_path_wo_finetune, 'r') as f:
            eval_results_dic_wo_finetune = json.load(f)
        with open(json_file_path_w_finetune, 'r') as f:
            eval_results_dic_w_finetune = json.load(f)

        l2_dist_array[0, epoch_idx] = eval_results_dic_wo_finetune['l2_dist_euc_p_p']
        l2_dist_array[1, epoch_idx] = eval_results_dic_w_finetune['l2_dist_euc_p_p']

    df_eval_results = pd.DataFrame(l2_dist_array, model_name_list, epoch_num_list)
    save_csv_file_path = os.path.join(saved_result_dir, f'comparision_finetune_on_volleyball_train_{train_mode}_{test_data_type}.csv')
    df_eval_results.to_csv(save_csv_file_path)