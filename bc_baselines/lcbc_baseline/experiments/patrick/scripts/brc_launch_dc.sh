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
            # --exclude n0214.savio3 \
            $@
    }

    job_count=10
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
            python ../roboverse/scripts/scripted_collect_parallel.py \
            -e Widow250DiversePickPlacePositionMultiObject-v0 \
            -pl pickplacewrist \
            -a place_success \
            -n 15000 \
            -t 50 \
            --save-all \
            -d /global/scratch/users/patrickhaoy/jaxrl/data/roboverse{1} \
            -p 15 \
            --noise 0.1 \
            ::: 0 1 2 3 4 5 6 7 8 9
fi