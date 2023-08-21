import os
from absl import app
from absl import flags

# from collections import defaultdict
from functools import partial

from jaxrl_m.utils import memory_patch  # Prevent out-of-memory issue. # NOQA

import tqdm
import glob
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.jax_utils import unreplicate
from flax.training import checkpoints
from flax.core import frozen_dict
from ml_collections import config_flags
from flax.training import checkpoints
import tensorflow as tf

from jaxrl_m.agents.continuous.gc_iql import create_iql_learner  # NOQA
from jaxrl_m.agents.continuous.gc_bc import create_bc_learner  # NOQA
from jaxrl_m.common.evaluation import supply_rng
from jaxrl_m.common.evaluation import evaluate_gc

# from jaxrl_m.common.evaluation import evaluate_gc_with_trajectories
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import BridgeDataset
from jaxrl_m.data.goal_conditioned_buffer import GCReplayBuffer
from jaxrl_m.data.goal_conditioned_buffer import load_gc_trajectories
from jaxrl_m.vision import encoders

from jaxrl_m.utils.gc_utils import concat_batches
from jaxrl_m.utils.gc_utils import make_gc_env
from jaxrl_m.utils.gc_utils import load_recorded_video
from jaxrl_m.utils.gc_utils import run_validation
from jaxrl_m.utils.timer_utils import Timer

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Environment name.", required=True)
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 2000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 2000, "Eval interval.")
flags.DEFINE_integer("eval_episodes", 20, "Number of episodes used for evaluation.")
flags.DEFINE_boolean("deterministic_eval", True, "Take mode of action dist. for eval")
flags.DEFINE_integer("expl_interval", 2000, "Expl interval.")
flags.DEFINE_integer("expl_episodes", 20, "Number of episodes used for exploration.")
flags.DEFINE_boolean("deterministic_expl", False, "Take mode of action dist. for expl")
flags.DEFINE_integer("save_interval", 25000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("num_devices", 1, "Number of GPUs to use.")
flags.DEFINE_integer("max_episode_steps", int(55), "Time horizon of episode.")
flags.DEFINE_integer("num_train_iters", int(5e5), "Number of training steps.")
flags.DEFINE_integer("num_eval_batches", int(32), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("save_dir", "./log/", "Video/buffer logging dir.")
flags.DEFINE_boolean("save_video", False, "Save videos during exploration.")
flags.DEFINE_integer(
    "num_episodes_per_video", 10, "Number of episodes to plot in each video."
)
flags.DEFINE_integer(
    "num_episodes_per_row", 5, "Number of episodes to plot in each row."
)
flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_boolean("debug", False, "Debugging mode.")

flags.DEFINE_string(
    "pretrained_dir", None, "Location of pretrained model", required=True
)
flags.DEFINE_float("online_fraction", 0.6, "Proportion of online data in batch")
flags.DEFINE_boolean("freeze_encoder", True, "Freeze encoder")

config_flags.DEFINE_config_file(
    "pretrained_config",
    "configs/offline_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):  # NOQA
    assert FLAGS.batch_size % FLAGS.num_devices == 0
    devices = jax.local_devices()[: FLAGS.num_devices]

    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_roboverse",
            "exp_prefix": "gc_roboverse_online",
            "exp_descriptor": FLAGS.name,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.pretrained_config.to_dict(),
        debug=FLAGS.debug,
    )

    save_dir = tf.io.gfile.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        wandb_logger.config.exp_prefix,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",  # NOQA
    )

    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.data_path, "train/metadata.npy"), "rb"
    ) as f:
        action_metadata = np.load(f, allow_pickle=True).item()

    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.data_path, "val/eval_goals.npy"), "rb"
    ) as f:
        eval_goals = np.load(f, allow_pickle=True).item()

    expl_goals = eval_goals  # TODO

    expl_env = make_gc_env(
        env_name=FLAGS.env_name,
        max_episode_steps=FLAGS.max_episode_steps,
        action_metadata=action_metadata,
        goals=expl_goals,
        save_video=FLAGS.save_video,
        save_video_dir=tf.io.gfile.join(save_dir, "videos"),
        save_video_prefix="expl",
    )

    eval_env = make_gc_env(
        env_name=FLAGS.env_name,
        max_episode_steps=FLAGS.max_episode_steps,
        action_metadata=action_metadata,
        goals=eval_goals,
        save_video=FLAGS.save_video,
        save_video_dir=tf.io.gfile.join(save_dir, "videos"),
        save_video_prefix="eval",
    )

    kwargs = FLAGS.pretrained_config.model_config.to_dict()

    encoder_def = encoders[kwargs["encoder"]](**kwargs["encoder_kwargs"])

    unrep_agent = globals()[FLAGS.pretrained_config.model_constructor](
        seed=FLAGS.seed,
        observations=expl_env.observation_space.sample(),
        goals=expl_env.observation_space.sample(),
        actions=expl_env.action_space.sample(),
        encoder_def=encoder_def,
        **kwargs["agent_kwargs"],
    )
    unrep_agent = checkpoints.restore_checkpoint(FLAGS.pretrained_dir, unrep_agent)
    rep_agent = replicate(unrep_agent, devices=devices)

    offline_batch_size = int(FLAGS.batch_size * (1 - FLAGS.online_fraction))
    # Round up to fit evenly into devices
    offline_batch_size += offline_batch_size % FLAGS.num_devices
    online_batch_size = int(FLAGS.batch_size - offline_batch_size)

    goal_keys = ["image"]

    if offline_batch_size > 0:
        train_paths = tf.io.gfile.glob(f"{FLAGS.data_path}/train/*.tfrecord")
        train_data = BridgeDataset(
            train_paths,
            FLAGS.seed,
            action_metadata=action_metadata,
            goal_keys=goal_keys,
            batch_size=offline_batch_size,
            num_devices=FLAGS.num_devices,
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
        goal_keys=goal_keys,
        batch_size=FLAGS.batch_size,
        num_devices=1,
        relabel_actions=False,
        train=False,
        **kwargs["dataset_kwargs"],
    )

    if online_batch_size > 0:
        online_buffer = GCReplayBuffer(
            observation_space=expl_env.observation_space,
            action_space=expl_env.action_space,
            goal_keys=goal_keys,
            seed=FLAGS.seed,
            num_devices=FLAGS.num_devices,
            **kwargs["replay_buffer_kwargs"],
        )

    rng = jax.random.PRNGKey(FLAGS.seed)
    timer = Timer()

    for i in tqdm.tqdm(
        range(FLAGS.num_train_iters), smoothing=0.1, disable=not FLAGS.tqdm
    ):

        timer.tick("total")

        # Evaluate.
        if i % FLAGS.eval_interval == 0:
            agent = unreplicate(rep_agent)
            policy_fn = supply_rng(
                partial(agent.sample_actions, argmax=FLAGS.deterministic_eval)
            )

            timer.tick("validation")
            metrics = run_validation(agent, val_data, FLAGS.num_eval_batches)
            for k, v in metrics.items():
                wandb_logger.log({f"validation/{k}": np.array(v)}, step=i)
            timer.tock("validation")

            timer.tick("evaluation")
            eval_env.start_recording(
                FLAGS.num_episodes_per_video, FLAGS.num_episodes_per_row
            )
            eval_info, eval_trajs = evaluate_gc(
                policy_fn,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                return_trajectories=True,
            )
            for k, v in eval_info.items():
                wandb_logger.log({f"evaluation/{k}": np.array(v)}, step=i)
            timer.tock("evaluation")

            if FLAGS.save_video:
                eval_video = load_recorded_video(video_path=eval_env.current_save_path)
                wandb_logger.log({"evaluation/video": eval_video}, step=i)

        # Collect.
        timer.tick("collection")
        if i % FLAGS.expl_interval == 0:
            agent = unreplicate(rep_agent)
            policy_fn = supply_rng(
                partial(agent.sample_actions, argmax=FLAGS.deterministic_expl)
            )

            expl_env.start_recording(
                FLAGS.num_episodes_per_video, FLAGS.num_episodes_per_row
            )
            expl_info, expl_trajs = evaluate_gc(
                policy_fn,
                expl_env,
                num_episodes=FLAGS.expl_episodes,
                return_trajectories=True,
            )
            load_gc_trajectories(expl_trajs, online_buffer)
            for k, v in expl_info.items():
                wandb_logger.log({f"exploration/{k}": np.array(v)}, step=i)

            if FLAGS.save_video:
                expl_video = load_recorded_video(video_path=expl_env.current_save_path)
                wandb_logger.log({"exploration/video": expl_video}, step=i)
        timer.tock("collection")

        # Train.
        timer.tick("dataset")

        if offline_batch_size > 0:
            offline_batch = next(train_data_iter)
            offline_batch = offline_batch.unfreeze()
            offline_batch["online_mask"] = np.zeros(
                offline_batch["rewards"].shape, dtype=np.float32
            )

        if online_batch_size > 0:
            rng, key = jax.random.split(rng)
            online_batch = online_buffer.sample(seed=key, batch_size=online_batch_size)
            online_batch = online_batch.unfreeze()
            online_batch["online_mask"] = np.ones(
                online_batch["rewards"].shape, dtype=np.float32
            )

        if offline_batch_size > 0 and online_batch_size > 0:
            batch = concat_batches(offline_batch, online_batch)
        elif offline_batch_size > 0:
            batch = offline_batch
        elif online_batch_size > 0:
            batch = online_batch
        else:
            raise ValueError

        timer.tock("dataset")

        timer.tick("training")
        rep_agent, rep_update_info = rep_agent.update(batch)
        timer.tock("training")

        # Log.
        if i % FLAGS.log_interval == 0:
            update_info = unreplicate(rep_update_info)
            for k, v in update_info.items():
                wandb_logger.log({f"training/{k}": np.array(v)}, step=i)

        # Evaluate.
        if i % FLAGS.eval_interval == 0:
            agent = unreplicate(rep_agent)
            policy_fn = supply_rng(
                partial(agent.sample_actions, argmax=FLAGS.deterministic_eval)
            )

            timer.tick("validation")
            metrics = run_validation(agent, val_data, FLAGS.num_eval_batches)
            for k, v in metrics.items():
                wandb_logger.log({f"validation/{k}": np.array(v)}, step=i)
            timer.tock("validation")

            timer.tick("evaluation")
            eval_env.start_recording(
                FLAGS.num_episodes_per_video, FLAGS.num_episodes_per_row
            )
            eval_info, eval_trajs = evaluate_gc(
                policy_fn,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                return_trajectories=True,
            )
            for k, v in eval_info.items():
                wandb_logger.log({f"evaluation/{k}": np.array(v)}, step=i)
            timer.tock("evaluation")

            if FLAGS.save_video:
                eval_video = load_recorded_video(
                    file_manager=file_manager, video_path=eval_env.current_save_path
                )
                wandb_logger.log({"evaluation/video": eval_video}, step=i)

        timer.tock("total")

        # Timer
        if i % FLAGS.log_interval == 0:
            for k, v in timer.get_average_times().items():
                wandb_logger.log({f"timer/{k}": np.array(v)}, step=i)

        # Save.
        if i % FLAGS.save_interval == 0:
            agent = unreplicate(rep_agent)
            checkpoints.save_checkpoint(save_dir, agent, step=i, keep=1e6)


if __name__ == "__main__":
    app.run(main)
