POLICY="bridge-ablation/gc-baseline_20230317_010909"
DIFFUSION="joint-0.05_20230317_003100"

POLICY_CKPT="checkpoint_60000"
DIFFUSION_CKPT="checkpoint_400000"

VIDEO_DIR="3-21-autonomous"

CMD="python experiments/homer/eval_policy_diffusion.py \
    --policy_checkpoint gs://rail-tpus-kevin/log/$POLICY/$POLICY_CKPT \
    --policy_wandb $POLICY \
    --diffusion_checkpoint gs://rail-tpus-kevin/logs/diffusion-affordance/$DIFFUSION/$DIFFUSION_CKPT \
    --diffusion_wandb kvablack/bridgedata-affordance/$DIFFUSION \
    --video_save_path ../../trainingdata/homer/videos/$VIDEO_DIR \
    --num_subgoals 10000 \
    --num_timesteps 100"

$CMD