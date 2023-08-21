"""
Offline RL with roboverse environments with negative diffusion goals
and evaluation of diffusion-based exploration.
"""
from absl import app, flags, logging
from functools import partial

import jax
import numpy as np
import tqdm
from flax.jax_utils import replicate
from flax.jax_utils import unreplicate
from ml_collections import config_flags
from flax.training import checkpoints
import tensorflow as tf

from jaxrl_m.agents.continuous.gc_iql import create_iql_learner  # NOQA
from jaxrl_m.agents.continuous.gc_bc import create_bc_learner  # NOQA
from jaxrl_m.common.evaluation import supply_rng
from jaxrl_m.common.evaluation import evaluate_gc
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import BridgeDataset
from jaxrl_m.vision import encoders
from experiments.kevin.diffusion_utils import (
    load_diffusion_checkpoint,
    add_diffusion_goals,
)

from jaxrl_m.utils.gc_utils import make_gc_env
from jaxrl_m.utils.gc_utils import load_recorded_video
from jaxrl_m.utils.gc_utils import run_validation
from jaxrl_m.utils.timer_utils import Timer

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Environment name.", required=True)
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 20, "Number of episodes used for evaluation.")
flags.DEFINE_boolean("deterministic_eval", True, "Take mode of action dist. for eval")
flags.DEFINE_integer("log_interval", 500, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("save_interval", 5000, "Save interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_episode_steps", int(55), "Time horizon of episode.")
flags.DEFINE_integer("num_train_iters", int(5e5), "Number of training steps.")
flags.DEFINE_integer("num_eval_batches", int(32), "Number of eval batches.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("save_dir", "./log/", "Video/buffer logging dir.")
flags.DEFINE_boolean("save_video", True, "Save videos during evaluation.")
flags.DEFINE_integer(
    "num_episodes_per_video", 8, "Number of episodes to plot in each video."
)
flags.DEFINE_integer(
    "num_episodes_per_row", 4, "Number of episodes to plot in each row."
)
flags.DEFINE_string("name", "", "Experiment name.")

flags.DEFINE_string(
    "diffusion_checkpoint", None, "Path to diffusion checkpoint.", required=True
)
flags.DEFINE_string("diffusion_wandb", None, "Diffusion wandb run name.", required=True)

config_flags.DEFINE_config_file(
    "config",
    "configs/offline_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    num_devices = jax.local_device_count()
    assert FLAGS.batch_size % num_devices == 0

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # init rng
    rng = jax.random.PRNGKey(FLAGS.seed)

    # wandb/logging setup
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_roboverse",
            "exp_descriptor": FLAGS.name,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
    )

    save_dir = tf.io.gfile.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",  # NOQA
    )

    # load action metadata
    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.data_path, "train/metadata.npy"), "rb"
    ) as f:
        action_metadata = np.load(f, allow_pickle=True).item()

    # load eval goals
    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.data_path, "val/eval_goals.npy"), "rb"
    ) as f:
        eval_goals = np.load(f, allow_pickle=True).item()

    # load diffusion stuff
    diffusion_config = FLAGS.config.model_config.diffusion
    diffusion_state, diffusion_sample_fn = load_diffusion_checkpoint(
        FLAGS.diffusion_wandb,
        FLAGS.diffusion_checkpoint,
        diffusion_config.num_sample_steps,
        diffusion_config.eta,
        diffusion_config.w,
    )
    jit_diffusion_sample = jax.jit(diffusion_sample_fn)
    pmap_diffusion_sample = jax.pmap(jit_diffusion_sample)
    rep_diffusion_state = replicate(diffusion_state)
    pmap_sample_fn = lambda seed, x: pmap_diffusion_sample(seed, rep_diffusion_state, x)
    jit_sample_fn = lambda seed, x: jit_diffusion_sample(seed, diffusion_state, x)
    rng, jit_sample_fn_key = jax.random.split(rng)
    rng_jit_sample_fn = supply_rng(jit_sample_fn, rng=jit_sample_fn_key)

    # create sim environment
    eval_env = make_gc_env(
        env_name=FLAGS.env_name,
        max_episode_steps=FLAGS.max_episode_steps,
        action_metadata=action_metadata,
        save_video=FLAGS.save_video,
        save_video_dir=tf.io.gfile.join(save_dir, "videos"),
        save_video_prefix="eval",
        goals=eval_goals,  # overridden later
    )

    # create model
    kwargs = FLAGS.config.model_config.to_dict()
    encoder_def = encoders[kwargs["encoder"]](**kwargs["encoder_kwargs"])
    unrep_agent = globals()[FLAGS.config.model_constructor](
        seed=FLAGS.seed,
        observations=eval_env.observation_space.sample(),
        goals=eval_env.observation_space.sample(),
        actions=eval_env.action_space.sample(),
        encoder_def=encoder_def,
        **kwargs["agent_kwargs"],
    )
    rep_agent = replicate(unrep_agent)

    # create data loaders
    train_paths = tf.io.gfile.glob(f"{FLAGS.data_path}/train/*.tfrecord")
    train_data = BridgeDataset(
        train_paths,
        FLAGS.seed,
        action_metadata=action_metadata,
        batch_size=FLAGS.batch_size,
        num_devices=num_devices,
        relabel_actions=False,
        train=True,
        **kwargs["dataset_kwargs"],
    )
    train_data_iter = train_data.get_iterator()

    val_paths = tf.io.gfile.glob(f"{FLAGS.data_path}/val/*.tfrecord")
    val_data = BridgeDataset(
        val_paths,
        FLAGS.seed,
        action_metadata=action_metadata,
        batch_size=FLAGS.batch_size,
        num_devices=1,
        relabel_actions=False,
        train=False,
        **kwargs["dataset_kwargs"],
    )

    # add diffusion goals to train loader
    if diffusion_config.generate_proportion > 0.0:
        rng, add_diffusion_goals_key = jax.random.split(rng)
        train_data_iter = add_diffusion_goals(
            add_diffusion_goals_key,
            train_data_iter,
            pmap_sample_fn,
            generate_proportion=diffusion_config.generate_proportion,
        )

    # train loop
    timer = Timer()

    for i in tqdm.tqdm(
        range(FLAGS.num_train_iters), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        timer.tick("total")

        timer.tick("dataset")
        batch = next(train_data_iter)
        timer.tock("dataset")

        timer.tick("training")
        rep_agent, rep_update_info = rep_agent.update(batch)
        timer.tock("training")

        # log train info
        if i % FLAGS.log_interval == 0:
            update_info = unreplicate(rep_update_info)
            for k, v in update_info.items():
                wandb_logger.log({f"training/{k}": np.array(v)}, step=i)

        # evaluate
        if i % FLAGS.eval_interval == 0:
            agent = unreplicate(rep_agent)
            rng, policy_key = jax.random.split(rng)
            policy_fn = supply_rng(
                partial(agent.sample_actions, argmax=FLAGS.deterministic_eval),
                rng=policy_key,
            )

            logging.info("Validation...")
            timer.tick("validation")
            metrics = run_validation(agent, val_data, FLAGS.num_eval_batches)
            for k, v in metrics.items():
                wandb_logger.log({f"validation/{k}": np.array(v)}, step=i)
            timer.tock("validation")

            logging.info("Evaluation...")
            timer.tick("evaluation")
            eval_env.goal_sampler = eval_goals
            eval_env.start_recording(
                FLAGS.num_episodes_per_video, FLAGS.num_episodes_per_row
            )
            eval_info = evaluate_gc(
                policy_fn,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                return_trajectories=False,
            )
            for k, v in eval_info.items():
                wandb_logger.log({f"evaluation/{k}": np.array(v)}, step=i + 1)
            if FLAGS.save_video:
                eval_video = load_recorded_video(video_path=eval_env.current_save_path)
                wandb_logger.log({"evaluation/video": eval_video}, step=i)
            timer.tock("evaluation")

            logging.info("Exploration...")
            timer.tick("exploration")
            # eval_env.goal_sampler = rng_jit_sample_fn
            # eval_env.start_recording(
            #     FLAGS.num_episodes_per_video, FLAGS.num_episodes_per_row
            # )
            # explore_info = evaluate_gc(
            #     policy_fn,
            #     eval_env,
            #     num_episodes=FLAGS.eval_episodes,
            #     return_trajectories=False,
            # )
            # for k, v in explore_info.items():
            #     wandb_logger.log({f"exploration/{k}": np.array(v)}, step=i + 1)
            # if FLAGS.save_video:
            #     eval_video = load_recorded_video(video_path=eval_env.current_save_path)
            #     wandb_logger.log({"exploration/video": eval_video}, step=i)
            timer.tock("exploration")

        timer.tock("total")

        # log timers
        if i % FLAGS.log_interval == 0:
            for k, v in timer.get_average_times().items():
                wandb_logger.log({f"timer/{k}": np.array(v)}, step=i)

        # save checkpoint
        if (i + 1) % FLAGS.save_interval == 0:
            agent = unreplicate(rep_agent)
            checkpoints.save_checkpoint(save_dir, agent, step=i + 1, keep=1e6)


if __name__ == "__main__":
    app.run(main)
