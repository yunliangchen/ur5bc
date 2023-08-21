POLICY="iql_20230420_220108"
DIFFUSION="new-joint-0.05_20230427_213735"

POLICY_CKPT="checkpoint_410000"
DIFFUSION_CKPT="checkpoint_200000"

CMD="python experiments/homer/collect_autonomous_data.py \
    --policy_checkpoint gs://rail-tpus-homer/log/jaxrl_m_bridgedata/$POLICY/$POLICY_CKPT \
    --policy_wandb widowx-gcrl/jaxrl_m_bridgedata/$POLICY \
    --diffusion_checkpoint gs://rail-tpus-kevin/logs/diffusion-affordance/$DIFFUSION/$DIFFUSION_CKPT \
    --diffusion_wandb kvablack/bridgedata-affordance/$DIFFUSION \
    --save_dir ../../trainingdata/robonetv2/bridge_data_v2/learned/ \
    --num_trajectories 10000 \
    --num_timesteps 60 \ 
    --blocking \
    --teleop_conf ../../widowx_envs/experiments/bridge_data_v2/conf.py"

$CMD