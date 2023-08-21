import os
from absl import app
from absl import flags

from jaxrl_m.utils import memory_patch  # Prevent out-of-memory issue.  # NOQA

import flax
import gym
import tqdm
import jax
import numpy as np
import roboverse
from flax.jax_utils import replicate
from flax.jax_utils import unreplicate
from ml_collections import config_flags
from flax.training import checkpoints
import tensorflow as tf

from jaxrl_m.agents.continuous.gc_iql import create_affordance_learner  # NOQA
from jaxrl_m.common.evaluation import supply_rng
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import BridgeDataset
from jaxrl_m.envs.wrappers.roboverse import GCRoboverseWrapper
from jaxrl_m.envs.wrappers.action_norm import UnnormalizeAction
from jaxrl_m.vision import encoders

from jaxrl_m.utils.timer_utils import Timer

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Environment name.", required=True)
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_string(
    "pretrained_dir", None, "Location of pretrained model", required=True
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 2000, "Logging interval.")
flags.DEFINE_integer("save_interval", 25000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("num_devices", 1, "Number of GPUs to use.")
flags.DEFINE_integer("num_train_iters", int(5e5), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("save_dir", "./log/", "Video/buffer logging dir.")
flags.DEFINE_string("name", "", "Experiment name.")

config_flags.DEFINE_config_file(
    "config",
    "configs/offline_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "pretrained_config",
    "configs/offline_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def wrap(env: gym.Env, action_metadata: dict, eval_goals: np.ndarray):
    env = GCRoboverseWrapper(env, eval_goals)
    env = UnnormalizeAction(env, action_metadata)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=55)
    return env


def main(_):
    assert FLAGS.batch_size % FLAGS.num_devices == 0
    devices = jax.local_devices()[: FLAGS.num_devices]

    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_roboverse",
            "exp_prefix": "gc_roboverse_offline_affordance",
            "exp_descriptor": FLAGS.name,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
    )

    FLAGS.save_dir = tf.io.gfile.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        wandb_logger.config.exp_prefix,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",  # NOQA
    )

    # Get sample obs, goal, action
    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.data_path, "train/metadata.npy"), "rb"
    ) as f:
        action_metadata = np.load(f, allow_pickle=True).item()
    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.data_path, "val/eval_goals.npy"), "rb"
    ) as f:
        eval_goals = np.load(f, allow_pickle=True).item()
    env = roboverse.make(FLAGS.env_name, transpose_image=False)
    env = wrap(env, action_metadata, eval_goals)
    env.reset()
    observation = env.observation_space.sample()
    goal = env.observation_space.sample()
    action = env.action_space.sample()
    env.close()

    # Load pretrained RL model
    pretrained_kwargs = FLAGS.pretrained_config.model_config.to_dict()
    encoder_def = encoders[pretrained_kwargs["encoder"]](
        **pretrained_kwargs["encoder_kwargs"]
    )
    unrep_pretrained_agent = globals()[FLAGS.pretrained_config.model_constructor](
        seed=FLAGS.seed,
        observations=observation,
        goals=goal,
        actions=action,
        encoder_def=encoder_def,
        **pretrained_kwargs["agent_kwargs"],
    )
    unrep_pretrained_agent = checkpoints.restore_checkpoint(
        FLAGS.pretrained_dir, unrep_pretrained_agent
    )

    # Get obs encoder latent dim
    encoder_output = unrep_pretrained_agent.model(
        observation, goal, action, capture_intermediates=True
    )[1]["intermediates"]["encoders_actor"]["__call__"]
    if type(encoder_output) == tuple:
        encoder_output = encoder_output[0]
    assert len(encoder_output.shape) == 1
    encoder_output_dim = encoder_output.shape[0]

    # Create affordance model
    kwargs = FLAGS.config.model_config.to_dict()
    kwargs["agent_kwargs"]["affordance_kwargs"]["output_dim"] = encoder_output_dim
    encoder_def = encoders[kwargs["encoder"]](**kwargs["encoder_kwargs"])
    unrep_agent = globals()[FLAGS.config.model_constructor](
        seed=FLAGS.seed,
        observations=observation,
        goals=goal,
        encoder_def=encoder_def,
        **kwargs["agent_kwargs"],
    )

    # Load pretrained encoder into affordance model
    init_params = unrep_agent.model.params
    init_params = flax.core.frozen_dict.unfreeze(init_params)
    init_params["networks_encoder"] = unrep_pretrained_agent.model.params[
        "encoders_actor"
    ]
    init_params = flax.core.frozen_dict.freeze(init_params)
    unrep_agent = unrep_agent.replace(
        model=unrep_agent.model.replace(params=init_params)
    )
    rep_agent = replicate(unrep_agent, devices=devices)

    agent_update_fn = supply_rng(rep_agent.update)
    agent_validate_fn = supply_rng(rep_agent.get_debug_metrics)

    assert FLAGS.data_path[-1] != "/"

    train_paths = tf.io.gfile.glob(f"{FLAGS.data_path}/train/*.tfrecord")
    train_data = BridgeDataset(
        train_paths,
        FLAGS.seed,
        action_metadata=action_metadata,
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

        # Train
        timer.tick("dataset")
        batch = next(train_data_iter)
        timer.tock("dataset")

        timer.tick("training")
        rep_agent, rep_update_info = agent_update_fn(batch)
        timer.tock("training")

        # Log.
        if i % FLAGS.log_interval == 0:
            update_info = unreplicate(rep_update_info)
            for k, v in update_info.items():
                wandb_logger.log({f"training/{k}": np.array(v)}, step=i)

        # Evaluate.
        if i % FLAGS.eval_interval == 0:
            timer.tick("validation")

            metrics = []
            for batch in val_data.get_iterator():
                metrics.append(agent_validate_fn(batch))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)

            for k, v in metrics.items():
                wandb_logger.log({f"validation/{k}": np.array(v)}, step=i)

            timer.tock("validation")

        # Timer
        if i % FLAGS.log_interval == 0:
            for k, v in timer.get_average_times().items():
                wandb_logger.log({f"timer/{k}": np.array(v)}, step=i)

        # Save.
        if i % FLAGS.save_interval == 0:
            agent = unreplicate(rep_agent)
            checkpoints.save_checkpoint(FLAGS.save_dir, agent, step=i, keep=1e6)


if __name__ == "__main__":
    app.run(main)
