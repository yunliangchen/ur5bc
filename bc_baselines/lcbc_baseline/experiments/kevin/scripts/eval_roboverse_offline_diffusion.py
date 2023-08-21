"""
Evaluate a pretrained RL checkpoint with goals proposed from a diffusion model.
"""
import itertools
import os
import pickle

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

from ray.util.multiprocessing import Pool

from absl import app, flags, logging
from functools import partial
import wandb

import jax
import numpy as np
import tqdm
from flax.training import checkpoints
import tensorflow as tf

from jaxrl_m.agents.continuous.gc_iql import create_iql_learner  # NOQA
from jaxrl_m.agents.continuous.gc_bc import create_bc_learner  # NOQA
from jaxrl_m.common.evaluation import supply_rng
from jaxrl_m.common.evaluation import evaluate_gc
from jaxrl_m.vision import encoders
from experiments.kevin.diffusion_utils import (
    load_diffusion_checkpoint,
)

from jaxrl_m.utils.gc_utils import make_gc_env
from jaxrl_m.utils.gc_utils import load_recorded_video


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Environment name.", required=True)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 20, "Number of episodes used for evaluation.")
flags.DEFINE_boolean("deterministic_eval", True, "Take mode of action dist. for eval")
flags.DEFINE_integer("max_episode_steps", int(55), "Time horizon of episode.")
flags.DEFINE_string("save_dir", "./log/", "Video/buffer logging dir.")
flags.DEFINE_boolean("save_video", True, "Save videos during evaluation.")
flags.DEFINE_integer(
    "num_episodes_per_video", 8, "Number of episodes to plot in each video."
)
flags.DEFINE_integer(
    "num_episodes_per_row", 4, "Number of episodes to plot in each row."
)
flags.DEFINE_integer("num_threads", 8, "Number of threads to use for evaluation.")

# flags.DEFINE_string(
#     "policy_checkpoint", None, "Path to policy checkpoint.", required=True
# )
# flags.DEFINE_string("policy_wandb", None, "Policy wandb run name.", required=True)

# flags.DEFINE_string(
#     "diffusion_checkpoint", None, "Path to diffusion checkpoint.", required=True
# )
# flags.DEFINE_string("diffusion_wandb", None, "Diffusion wandb run name.", required=True)

flags.DEFINE_integer("diffusion_num_sample_steps", 100, "Number of diffusion steps.")
flags.DEFINE_float("diffusion_eta", 0.0, "Diffusion eta.")
flags.DEFINE_float("diffusion_w", 0.0, "Diffusion w.")


def load_policy_checkpoint(wandb_run_name, checkpoint_path):
    assert tf.io.gfile.exists(
        checkpoint_path
    ), f"Checkpoint path {checkpoint_path} does not exist."

    # load information from wandb
    api = wandb.Api()
    run = api.run(wandb_run_name)
    model_config = run.config["model_config"]
    model_constructor = run.config["model_constructor"]

    # load action metadata (assumes we can still access the same path; this
    # always works when the dataset is on cloud storage)
    with tf.io.gfile.GFile(
        tf.io.gfile.join(run.config["data_path"], "train/metadata.npy"), "rb"
    ) as f:
        action_metadata = np.load(f, allow_pickle=True).item()

    # create agent
    model_config["agent_kwargs"]["optim_kwargs"] = {"learning_rate": 0.0}

    encoder_def = encoders[model_config["encoder"]](**model_config["encoder_kwargs"])
    agent = globals()[model_constructor](
        seed=0,
        encoder_def=encoder_def,
        observations=np.zeros((128, 128, 3), dtype=np.uint8),
        goals=np.zeros((128, 128, 3), dtype=np.uint8),
        actions=np.zeros(7, dtype=np.float32),
        **model_config["agent_kwargs"],
    )
    agent_dict = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    params = agent_dict["model"]["params"]
    target_params = agent_dict["target_model"]["params"]
    agent = agent.replace(
        model=agent.model.replace(params=params),
        target_model=agent.target_model.replace(params=target_params),
    )

    return agent, action_metadata


def eval_diffusion_expl(
    policy_wandb,
    policy_checkpoint,
    diffusion_wandb,
    diffusion_checkpoint,
    diffusion_num_sample_steps,
    diffusion_eta,
    diffusion_w,
    seed,
    num_threads,
    deterministic_eval,
    env_name,
    max_episode_steps,
    eval_episodes,
    num_episodes_per_video,
    num_episodes_per_row,
    save_video,
    save_dir,
):
    # load policy checkpoint
    agent, action_metadata = load_policy_checkpoint(policy_wandb, policy_checkpoint)

    # load diffusion checkpoint
    diffusion_state, diffusion_sample_fn = load_diffusion_checkpoint(
        diffusion_wandb,
        diffusion_checkpoint,
        diffusion_num_sample_steps,
        diffusion_eta,
        diffusion_w,
    )
    jit_diffusion_sample = jax.jit(diffusion_sample_fn)

    def worker_fn(i):
        device_diffusion_state = jax.device_put(
            diffusion_state, jax.local_devices()[i % jax.local_device_count()]
        )
        device_agent = jax.device_put(
            agent, jax.local_devices()[i % jax.local_device_count()]
        )

        def sample_fn(x, seed):
            return jit_diffusion_sample(
                seed,
                device_diffusion_state,
                jax.device_put(
                    x["image"][None],
                    device=jax.local_devices()[i % jax.local_device_count()],
                ),
            )[0]

        sample_fn = supply_rng(sample_fn, rng=jax.random.PRNGKey(seed + i))

        def policy_fn(x, *args, **kwargs):
            return device_agent.sample_actions(
                jax.device_put(
                    x,
                    device=jax.local_devices()[i % jax.local_device_count()],
                ),
                *args,
                argmax=deterministic_eval,
                **kwargs,
            )

        policy_fn = supply_rng(policy_fn, rng=jax.random.PRNGKey(seed + i))

        eval_env = make_gc_env(
            env_name=env_name,
            max_episode_steps=max_episode_steps,
            action_metadata=action_metadata,
            save_video=save_video,
            save_video_dir=tf.io.gfile.join(save_dir, "videos"),
            save_video_prefix=f"eval_{i}",
            goals=sample_fn,
        )

        eval_env.start_recording(num_episodes_per_video, num_episodes_per_row)
        eval_info = evaluate_gc(
            policy_fn,
            eval_env,
            num_episodes=eval_episodes // num_threads
            + int(i < eval_episodes % num_threads),
            return_trajectories=False,
        )
        return eval_info

    with Pool(num_threads) as pool:
        eval_infos = pool.map(worker_fn, range(num_threads))

    eval_info = jax.tree_map(lambda *xs: np.mean(xs), *eval_infos)

    return eval_info


def main(_):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    import pdb

    pdb.set_trace()
    diffusion_wandb = (
        "kvablack/diffusion-affordance/newsim_filter_cos_p2_20230127_234302"
    )
    diffusion_checkpoints = [
        "gs://rail-tpus-kevin/logs/diffusion-affordance/newsim_filter_cos_p2_20230127_234302/checkpoint_50000",
        "gs://rail-tpus-kevin/logs/diffusion-affordance/newsim_filter_cos_p2_20230127_234302/checkpoint_100000",
        "gs://rail-tpus-kevin/logs/diffusion-affordance/newsim_filter_cos_p2_20230127_234302/checkpoint_150000",
    ]

    policy_wandb = (
        "kvablack/jaxrl_m_roboverse/roboverse_diverse_filtered_20230128_085711"
    )
    policy_checkpoints = [
        "gs://rail-tpus-kevin/log/jaxrl_m_sim/jaxrl_m_roboverse/roboverse_diverse_filtered_20230128_085711/checkpoint_100000",
        "gs://rail-tpus-kevin/log/jaxrl_m_sim/jaxrl_m_roboverse/roboverse_diverse_filtered_20230128_085711/checkpoint_200000",
        "gs://rail-tpus-kevin/log/jaxrl_m_sim/jaxrl_m_roboverse/roboverse_diverse_filtered_20230128_085711/checkpoint_300000",
    ]

    results = []
    for diffusion_checkpoint, policy_checkpoint in itertools.product(
        diffusion_checkpoints, policy_checkpoints
    ):
        logging.info(
            f"Evaluating diffusion checkpoint {diffusion_checkpoint} and policy checkpoint {policy_checkpoint}"
        )
        eval_info = eval_diffusion_expl(
            policy_wandb=policy_wandb,
            policy_checkpoint=policy_checkpoint,
            diffusion_wandb=diffusion_wandb,
            diffusion_checkpoint=diffusion_checkpoint,
            diffusion_num_sample_steps=FLAGS.diffusion_num_sample_steps,
            diffusion_eta=FLAGS.diffusion_eta,
            diffusion_w=FLAGS.diffusion_w,
            seed=FLAGS.seed,
            num_threads=FLAGS.num_threads,
            deterministic_eval=FLAGS.deterministic_eval,
            env_name=FLAGS.env_name,
            max_episode_steps=FLAGS.max_episode_steps,
            eval_episodes=FLAGS.eval_episodes,
            num_episodes_per_video=FLAGS.num_episodes_per_video,
            num_episodes_per_row=FLAGS.num_episodes_per_row,
            save_video=FLAGS.save_video,
            save_dir=tf.io.gfile.join(
                FLAGS.save_dir,
                diffusion_checkpoint.split("_")[-1],
                policy_checkpoint.split("_")[-1],
            ),
        )
        results.append((diffusion_checkpoint, policy_checkpoint, eval_info))

    with open(tf.io.gfile.join(FLAGS.save_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    app.run(main)
