import functools
import sys
from widowx_envs.widowx.widowx_env import (
    BridgeDataRailRLPrivateWidowX,
    BridgeDataRailRLPrivateVRWidowX,
)
import os
import numpy as np
from PIL import Image
from flax.training import checkpoints
import tensorflow as tf
import traceback
import wandb
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
from absl import app, flags, logging
import time
from datetime import datetime
import jax
import time
from denoising_diffusion_flax.model import EmaTrainState, create_model_def
from denoising_diffusion_flax import scheduling, sampling
import ml_collections

np.set_printoptions(suppress=True)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "policy_checkpoint", None, "Path to policy checkpoint", required=True
)
flags.DEFINE_string(
    "diffusion_checkpoint", None, "Path to diffusion checkpoint", required=True
)
flags.DEFINE_string(
    "policy_wandb", None, "Policy checkpoint wandb run name", required=True
)
flags.DEFINE_string(
    "diffusion_wandb", None, "Diffusion checkpoint wandb run name", required=True
)

flags.DEFINE_integer(
    "diffusion_sample_steps", 200, "Number of timesteps to use for diffusion sampler"
)
flags.DEFINE_float("diffusion_eta", 0.1, "Eta to use for diffusion sampler")
flags.DEFINE_float("diffusion_w", 1.0, "CFG weight to use for diffusion sampler")

flags.DEFINE_string("video_save_path", None, "Path to save video")

flags.DEFINE_integer("num_timesteps", 50, "Number of timesteps per subgoal")
flags.DEFINE_integer("num_subgoals", 5, "Number of subgoals to run")

flags.DEFINE_bool("blocking", True, "Use the blocking controller")

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 2

START_IMAGE_EEP = [0.3, 0.0, 0.15]

FIXED_STD = np.array([0.005, 0.005, 0.005, 0.0, 0.0, 0.0, 0.0])


def unnormalize_action(action, mean, std):
    return action * std + mean


def load_policy_checkpoint(checkpoint_path, wandb_run_name):
    assert tf.io.gfile.exists(checkpoint_path)

    # load information from wandb
    api = wandb.Api()
    run = api.run(wandb_run_name)

    # create encoder from wandb config
    encoder_def = encoders[run.config["encoder"]](**run.config["encoder_kwargs"])

    example_batch = {
        "observations": {"image": np.zeros((128, 128, 3), dtype=np.uint8)},
        "goals": {"image": np.zeros((128, 128, 3), dtype=np.uint8)},
        "actions": np.zeros(7, dtype=np.float32),
    }

    # create agent from wandb config
    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[run.config["agent"]].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **run.config["agent_kwargs"],
    )

    # load action metadata from wandb
    action_metadata = run.config["bridgedata_config"]["action_metadata"]
    action_mean = np.array(action_metadata["mean"])
    action_std = np.array(action_metadata["std"])

    # hydrate agent with parameters from checkpoint
    agent = checkpoints.restore_checkpoint(checkpoint_path, agent)

    return agent, action_mean, action_std


def load_diffusion_checkpoint(
    checkpoint_path, wandb_run_name, num_sample_steps, w, eta
):
    assert tf.io.gfile.exists(checkpoint_path)

    # load config from wandb
    api = wandb.Api()
    run = api.run(wandb_run_name)
    config = ml_collections.ConfigDict(run.config["config"])

    # create model def
    model_def = create_model_def(
        config.model,
    )

    # load weights
    ckpt_dict = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    state = EmaTrainState(
        step=0,
        apply_fn=model_def.apply,
        params=ckpt_dict["params"],
        params_ema=ckpt_dict["params_ema"],
        tx=None,
        opt_state=None,
    )

    # parse ddpm params
    log_snr_fn = scheduling.create_log_snr_fn(config.ddpm)

    rng = jax.random.PRNGKey(0)

    def sample_fn(image):
        nonlocal rng
        image = image / 127.5 - 1.0
        rng, key = jax.random.split(rng)
        sample = sampling.sample_loop(
            key,
            state,
            image[None],
            log_snr_fn=log_snr_fn,
            num_timesteps=num_sample_steps,
            w=w,
            eta=eta,
            self_condition=config.ddpm.self_condition,
        )[0]
        return np.array(np.clip(sample * 127.5 + 127.5 + 0.5, 0, 255).astype(np.uint8))

    return sample_fn


def rollout_subgoal(
    rng, env, agent, goal_image, action_mean, action_std, num_timesteps
):
    goal_obs = {
        "image": goal_image,
    }

    is_gripper_closed = False
    num_consecutive_gripper_change_actions = 0

    env.reset()
    env.start()

    obs = env._get_obs()
    last_tstep = time.time()
    images = []
    t = 0
    try:
        while t < num_timesteps:
            if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                image_obs = (
                    obs["image"].reshape(3, 128, 128).transpose(1, 2, 0) * 255
                ).astype(np.uint8)
                obs = {"image": image_obs, "proprio": obs["state"]}

                last_tstep = time.time()

                rng, key = jax.random.split(rng)
                action = np.array(
                    agent.sample_actions(obs, goal_obs, seed=key, argmax=True)
                )
                action = unnormalize_action(action, action_mean, action_std)
                action += np.random.normal(0, FIXED_STD)

                # sticky gripper logic
                if (action[-1] < 0.5) != is_gripper_closed:
                    num_consecutive_gripper_change_actions += 1
                else:
                    num_consecutive_gripper_change_actions = 0

                if num_consecutive_gripper_change_actions >= STICKY_GRIPPER_NUM_STEPS:
                    is_gripper_closed = not is_gripper_closed
                    num_consecutive_gripper_change_actions = 0

                action[-1] = 0.0 if is_gripper_closed else 1.0

                ### Preprocess action ###
                if NO_PITCH_ROLL:
                    action[3] = 0
                    action[4] = 0
                if NO_YAW:
                    action[5] = 0

                ### Env step ###
                obs, rew, done, info = env.step(
                    action, last_tstep + STEP_DURATION, blocking=FLAGS.blocking
                )

                image_formatted = np.concatenate((goal_image, image_obs), axis=0)
                images.append(Image.fromarray(image_formatted))

                t += 1
    except Exception as e:
        logging.error(traceback.format_exc())
        return images, False

    return images, True


def main(_):
    if FLAGS.video_save_path is not None:
        os.makedirs(FLAGS.video_save_path, exist_ok=True)

    # load policy checkpoint
    agent, action_mean, action_std = load_policy_checkpoint(
        FLAGS.policy_checkpoint, FLAGS.wandb_run_name
    )

    # load diffusion checkpoint
    sample_fn = load_diffusion_checkpoint(
        FLAGS.diffusion_checkpoint,
        FLAGS.wandb_run_name,
        FLAGS.num_sample_steps,
        FLAGS.w,
        FLAGS.eta,
    )

    ### Setup env ###
    env_params = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.23, -0.11, 0, -1.57, 0],
            [0.44, 0.23, 0.18, 1.57, 0],
        ],
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": None,
        "return_full_image": False,
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=128)

    rng = jax.random.PRNGKey(0)
    images = []
    for i in range(FLAGS.num_subgoals):
        success = False
        while not success:
            try:
                env._controller.open_gripper(True)
                env._controller.move_to_state(START_IMAGE_EEP, 0, duration=2.0)
                env._reset_previous_qpos()
                success = True
            except Exception as e:
                continue

        obs = env._get_obs()
        image_obs = (obs["image"].reshape(3, 128, 128).transpose(1, 2, 0) * 255).astype(
            np.uint8
        )

        # sample from diffusion model
        logging.info(f"Subgoal {i}: sampling from diffusion model...")
        image_goal = sample_fn(image_obs)

        Image.fromarray(np.concatenate([image_obs, image_goal], axis=1)).save(
            os.path.join(FLAGS.video_save_path, "goal.png")
        )

        # rollout subgoal
        logging.info(f"Subgoal {i}: rolling out...")
        rng, key = jax.random.split(rng)
        new_images, success = rollout_subgoal(
            key, env, agent, image_goal, action_mean, action_std, FLAGS.num_timesteps
        )
        images += new_images
        if not success:
            break

    # Save Video
    if FLAGS.video_save_path is not None:
        save_path = os.path.join(
            FLAGS.video_save_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.gif")
        )
        logging.info(f"Saving Video at {save_path}...")
        images[0].save(
            save_path,
            format="GIF",
            append_images=images[1:],
            save_all=True,
            duration=200,
            loop=0,
        )


if __name__ == "__main__":
    app.run(main)
