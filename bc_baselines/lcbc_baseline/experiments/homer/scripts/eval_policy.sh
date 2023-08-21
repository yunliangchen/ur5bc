MODELS=(
    "jaxrl_m_bridgedata/bc_resnet50_20230524_074533"
    "jaxrl_m_bridgedata/bc_resnet50_20230524_074533"
)

CKPTS=(
    "80000"
    "120000"
)

VIDEO_DIR="5-26"

CMD="python experiments/homer/eval_policy.py \
    --num_timesteps 60 \
    --video_save_path ../../trainingdata/homer/videos/$VIDEO_DIR \
    $(for i in "${!MODELS[@]}"; do echo "--checkpoint_path gs://rail-tpus-homer/log/${MODELS[$i]}/checkpoint_${CKPTS[$i]} "; done) \
    $(for i in "${!MODELS[@]}"; do echo "--wandb_run_name widowx-gcrl/${MODELS[$i]} "; done) \
    --blocking \
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.15" --initial_eep "0.3 0.0 0.15"