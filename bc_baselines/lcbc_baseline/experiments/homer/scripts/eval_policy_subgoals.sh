export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
# MODEL="jaxrl_m_bridgedata/all_iql_20230303_064422"
# MODEL="jaxrl_m_bridgedata/all_joint_training_20230303_072703"
# MODEL="jaxrl_m_bridgedata/settable_scripted_bridge_pnp_joint_training_20230303_071621"
# MODEL="jaxrl_m_bridgedata/all_bc_20230303_064643"
# MODEL="jaxrl_m_bridgedata/all_bc_joint_training_20230307_053541"
# MODEL="jaxrl_m_bridgedata/all_bc_finetuning_85k_20230307_055949"

MODEL="jaxrl_m_bridgedata/bc_joint_autonomous_20230311_061117"
VIDEO_DIR="3-14/bc_joint_autonomous"

CKPT="checkpoint_60000"

CMD="python experiments/homer/eval_policy_subgoals.py \
    --num_timesteps 60 \
    --video_save_path ../../trainingdata/homer/videos/$VIDEO_DIR \
    --checkpoint_path gs://rail-tpus-homer/log/$MODEL/$CKPT \
    --wandb_run_name $MODEL \
    --high_res \
    --num_subgoals 3"

# $CMD --goal_eep "0.4 0.0 0.2" --initial_eep "0.4 0.0 0.1"

$CMD
