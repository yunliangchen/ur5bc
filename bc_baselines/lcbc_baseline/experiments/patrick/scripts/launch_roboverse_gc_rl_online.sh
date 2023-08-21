python experiments/patrick/roboverse_online_gc.py \
	--env_name Widow250DiversePickPlacePositionMultiObjectFixGripper-v0 \
    --data_path /media/ashvin/data1/patrickhaoy/data/roboverse_Widow250DiversePickPlacePositionMultiObjectFixGripper-v0_10K_save_all_noise_0.1_2022-12-20T13-51-45 \
	--seed 1 \
    --num_devices 1 \
    --save_video True \
    --save_dir /media/ashvin/data1/patrickhaoy/logs/ \
    --name 'rl_128_diverse_v2_fix_gripper_widowx_online' \
    --online_fraction 0.6 \
    --pretrained_dir /media/ashvin/data1/patrickhaoy/logs/jaxrl_m_roboverse/gc_roboverse_offline/rl_128_diverse_v2_fix_gripper_widowx_20230109_045740 \
    --pretrained_config experiments/patrick/configs/offline_pixels_config_2b.py:roboverse_gc_iql \