import os
from absl import app
from absl import flags
from functools import partial

from jaxrl_m.utils import memory_patch  # Prevent out-of-memory issue. # NOQA

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

from jaxrl_m.utils.gc_utils import make_gc_env
from jaxrl_m.utils.gc_utils import load_recorded_video
from jaxrl_m.utils.gc_utils import run_validation
from jaxrl_m.utils.timer_utils import Timer

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Environment name.", required=True)
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_boolean("deterministic_eval", True, "Take mode of action dist. for eval")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("save_interval", 25000, "Save interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("num_devices", 1, "Number of GPUs to use.")
flags.DEFINE_integer("max_episode_steps", int(55), "Time horizon of episode.")
flags.DEFINE_integer("num_train_iters", int(5e5), "Number of training steps.")
flags.DEFINE_integer("num_eval_batches", int(32), "Number of training steps.")
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
flags.DEFINE_boolean("debug", False, "Debugging mode.")

config_flags.DEFINE_config_file(
    "config",
    "configs/offline_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    assert FLAGS.batch_size % FLAGS.num_devices == 0
    devices = jax.local_devices()[: FLAGS.num_devices]

    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_kuan",
            "exp_prefix": "gc_roboverse_offline",
            "exp_descriptor": FLAGS.name,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
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

    eval_env = make_gc_env(
        env_name=FLAGS.env_name,
        max_episode_steps=FLAGS.max_episode_steps,
        action_metadata=action_metadata,
        goals=eval_goals,
        save_video=FLAGS.save_video,
        save_video_dir=tf.io.gfile.join(save_dir, "videos"),
        save_video_prefix="eval",
    )

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
    rep_agent = replicate(unrep_agent, devices=devices)

    assert FLAGS.data_path[-1] != "/"

    goal_keys = ["image"]

    train_paths = tf.io.gfile.glob(f"{FLAGS.data_path}/train/*.tfrecord")
    train_data = BridgeDataset(
        train_paths,
        FLAGS.seed,
        action_metadata=action_metadata,
        goal_keys=goal_keys,
        batch_size=FLAGS.batch_size,
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

    timer = Timer()

    for i in tqdm.tqdm(
        range(FLAGS.num_train_iters), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        timer.tick("total")

        # Train.
        timer.tick("dataset")
        batch = next(train_data_iter)
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
                eval_video = load_recorded_video(video_path=eval_env.current_save_path)
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
