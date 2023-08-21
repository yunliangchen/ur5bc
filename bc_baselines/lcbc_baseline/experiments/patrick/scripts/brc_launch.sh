#! /bin/bash

if [ -z "$1" ]; then
    SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    cd "$SCRIPT_DIR"
    EXP_NAME='jaxrl'
    mkdir -p ./logs/$EXP_NAME
    slurm_run () {
        sbatch \
            --job-name=$EXP_NAME \
            --time=72:00:00 \
            --account=co_rail \
            --qos=savio_lowprio \
            --partition=savio3_gpu \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=8 \
            --gres=gpu:A40:1 \
            --error ./logs/$EXP_NAME/slurm-%j.out \
            --output ./logs/$EXP_NAME/slurm-%j.out \
            --exclude n0214.savio3 \
            $@
    }

    job_count=6
    runs_per_job=1
    for job_idx in $(seq 0 $(($job_count - 1))); do
        slurm_run "${BASH_SOURCE[0]}" $job_count $job_idx $runs_per_job $SCRIPT_DIR
    done

else
    export JOB_COUNT=$1
    export JOB_IDX=$2
    export RUNS_PER_JOB=$3
    export SCRIPT_DIR=$4
    export PROJECT_HOME=$SCRIPT_DIR

    module load gnu-parallel

    parallel --colsep ',' --delay 20 --linebuffer -j $RUNS_PER_JOB \
        '[ $JOB_IDX == $(({#} % $JOB_COUNT)) ] && 'singularity run \
            --env CODEPATH='/global/scratch/users/patrickhaoy/jaxrl/code/jaxrl_minimal:/global/scratch/users/patrickhaoy/jaxrl/code/roboverse' \
            -B /var/lib/dcv-gl --nv --writable-tmpfs \
            $PROJECT_HOME/base_img.sif \
            python experiments/patrick/roboverse_offline_gc.py \
                --config experiments/patrick/configs/offline_pixels_config.py:roboverse_gc_iql \
                --env_name Widow250DiversePickPlacePositionMultiObject-v0 \
                --data_path /global/scratch/users/patrickhaoy/jaxrl/data/roboverse_Widow250DiversePickPlacePositionMultiObject-v0_10K_save_all_noise_0.1_2022-12-17T00-46-26/ \
                --seed 1 \
                --eval_episodes 20 \
                --log_interval 2000 \
                --eval_interval 2000 \
                --save_interval 100000 \
                --batch_size 256 \
                --num_devices 1 \
                --save_video True \
                --deterministic_eval \
                --save_dir /global/scratch/users/patrickhaoy/jaxrl/log \
                --name 'rl_128_diverse_widowx' \
                --config.model_config.dataset_kwargs.use_proportion={1} \
                --config.model_config.agent_kwargs.temperature={2} \
            ::: 1.0 0.1 0.01 \
            ::: 1.0 0.1
fi