import os
from jaxrl_m.common.evaluation import supply_rng, evaluate
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import BridgeDataset
from jaxrl_m.envs.wrappers.action_norm import UnnormalizeAction
from jaxrl_m.envs.wrappers.roboverse import RoboverseWrapper
from jaxrl_m.envs.wrappers.video_recorder import VideoRecorder
from jaxrl_m.agents.continuous.bc import create_bc_learner
from jaxrl_m.vision import encoders

import gym
import tqdm
import glob
import jax
import wandb
from absl import app, flags
from ml_collections import config_flags
import numpy as np
from flax.training import checkpoints
from flax.jax_utils import replicate, unreplicate
import roboverse

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Environment name.", required=True)
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("encoder", "resnetv1-18-bridge", "Encoder name")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 100, "Logging interval.")
flags.DEFINE_integer("eval_interval", 1000, "Eval interval.")
flags.DEFINE_integer("save_interval", 1000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("num_devices", 1, "Number of GPUs to use.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_string("save_dir", "./log/", "Video/buffer logging dir.")

config_flags.DEFINE_config_file(
    "config",
    "configs/offline_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def wrap(env: gym.Env, action_metadata: dict):
    env = RoboverseWrapper(env)
    env = UnnormalizeAction(env, action_metadata)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=40)
    return env


def main(_):
    assert FLAGS.batch_size % FLAGS.num_devices == 0
    devices = jax.local_devices()[: FLAGS.num_devices]

    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_roboverse",
            "exp_prefix": "offline",
            "exp_descriptor": f"{FLAGS.env_name}",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
    )

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        wandb_logger.config.exp_prefix,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    action_metadata = np.load(
        os.path.join(FLAGS.data_path, "metadata.npy"), allow_pickle=True
    ).item()

    eval_env = roboverse.make(FLAGS.env_name, transpose_image=False)
    eval_env = wrap(eval_env, action_metadata)
    if FLAGS.save_video:
        eval_env = VideoRecorder(eval_env, os.path.join(FLAGS.save_dir, "videos"))
    eval_env.reset(seed=FLAGS.seed + 42)

    encoder_def = encoders[FLAGS.encoder](**FLAGS.config.model_config.encoder_kwargs)

    kwargs = dict(FLAGS.config.model_config)

    unrep_agent = globals()[FLAGS.config.model_constructor](
        seed=FLAGS.seed,
        observations=eval_env.observation_space.sample(),
        actions=eval_env.action_space.sample(),
        encoder_def=encoder_def,
        **kwargs,
    )
    rep_agent = replicate(unrep_agent, devices=devices)

    data_paths = glob.glob(f"{FLAGS.data_path}/*.hdf5")
    dataset_iterator = BridgeDataset(
        data_paths,
        FLAGS.seed,
        action_metadata=action_metadata,
        goal_relabel_reached_proportion=0.1,
        batch_size=FLAGS.batch_size,
        num_devices=FLAGS.num_devices,
        train=True,
        augment=False,
    ).get_iterator()

    for i in tqdm.tqdm(range(FLAGS.max_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = next(dataset_iterator)
        rep_agent, rep_update_info = rep_agent.update(batch)

        if i % FLAGS.log_interval == 0:
            update_info = unreplicate(rep_update_info)
            for k, v in update_info.items():
                wandb_logger.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            agent = unreplicate(rep_agent)
            policy_fn = supply_rng(agent.sample_actions)
            eval_info = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb_logger.log({f"evaluation/{k}": v}, step=i)

        if i % FLAGS.save_interval == 0:
            agent = unreplicate(rep_agent)
            checkpoints.save_checkpoint(FLAGS.save_dir, agent, step=i, keep=1e6)


if __name__ == "__main__":
    app.run(main)
