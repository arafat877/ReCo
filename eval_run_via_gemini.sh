
# Step 1: First use gemini-2.5-flash-thinking to eval each dimansion
python eval_step1_run_gemini_api.py --edited_video_folder $YOUR_edited_video_folder \
    --src_video_folder $Ori_videos_folder --base_txt_folder $base_config_folder \
    --task_name $Eval_task_name(e.g.,: add, remove, replace, style)


# Step 2: Compute the final scores; 
# the final calculation can be run after all 4 tasks have been fully evaluated.
# default: all_results/gemini_results
python eval_step2_get_final_scores.py --json_folder $YOUR_json_output_folder_last_step \
    --base_txt_folder $base_config_folder





