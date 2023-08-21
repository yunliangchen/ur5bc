from widowx_envs.widowx.widowx_env import (
    BridgeDataRailRLPrivateWidowX,
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
import imp
import json
from widowx_envs.trajectory_collector import TrajectoryCollector
import pickle
import multiprocessing
import signal
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic

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

flags.DEFINE_string("save_dir", None, "Path to save videos/data")

flags.DEFINE_integer("num_timesteps", 50, "Number of timesteps per subgoal")
flags.DEFINE_integer("num_trajectories", 5, "Number of trajectories to run")

flags.DEFINE_bool("blocking", False, "Use the blocking controller")

flags.DEFINE_string("teleop_conf", None, "Path to teleop conf.")

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 2

START_IMAGE_EEP = [0.3, 0.0, 0.2]

FIXED_STD = np.array([0.005, 0.005, 0.005, 0.0, 0.0, 0.05, 0.0])


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    for key in dict_of_lists:
        dict_of_lists[key] = np.array(dict_of_lists[key])
    return dict_of_lists


def unnormalize_action(action, mean, std):
    return action * std + mean


def log_floor_height(save_dir):
    hyperparams = imp.load_source("hyperparams", FLAGS.teleop_conf).config
    hyperparams["log_dir"] = "."
    meta_data_dict = json.load(open(hyperparams["collection_metadata"], "r"))
    meta_data_dict["date_time"] = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    s = TrajectoryCollector(hyperparams)
    print("#################################################")
    print("#################################################")
    print(
        "Move the gripper all the way to the lowest point of the workspace and end the trajectory."
    )
    _, obs_dict, _ = s.agent.sample(s.policies[0], 0)
    floor_height = np.min(obs_dict["full_state"][:, 2])
    meta_data_dict["floor_height"] = floor_height
    with open(os.path.join(save_dir, "collection_metadata.json"), "w") as outfile:
        json.dump(meta_data_dict, outfile)
    os.kill(os.getpid(), signal.SIGKILL)


def load_policy_checkpoint():
    assert tf.io.gfile.exists(FLAGS.policy_checkpoint)

    # load information from wandb
    api = wandb.Api()
    run = api.run(FLAGS.policy_wandb)

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
    agent = checkpoints.restore_checkpoint(FLAGS.policy_checkpoint, agent)

    return agent, action_mean, action_std


def load_diffusion_checkpoint():
    assert tf.io.gfile.exists(FLAGS.diffusion_checkpoint)

    # load config from wandb
    api = wandb.Api()
    run = api.run(FLAGS.diffusion_wandb)
    config = ml_collections.ConfigDict(run.config["config"])

    # create model def
    model_def = create_model_def(
        config.model,
    )

    # load weights
    ckpt_dict = checkpoints.restore_checkpoint(FLAGS.diffusion_checkpoint, target=None)
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
            num_timesteps=FLAGS.diffusion_sample_steps,
            w=FLAGS.diffusion_w,
            eta=FLAGS.diffusion_eta,
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

    images = []
    full_images = []
    obs_dict = []
    policy_out = []
    obs = env._get_obs()

    last_tstep = time.time()
    t = 0
    try:
        while t < num_timesteps:
            if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                image_obs = (
                    obs["image"].reshape(3, 128, 128).transpose(1, 2, 0) * 255
                ).astype(np.uint8)
                policy_obs = {"image": image_obs, "proprio": obs["state"]}

                last_tstep = time.time()

                rng, key = jax.random.split(rng)
                action = np.array(
                    agent.sample_actions(policy_obs, goal_obs, seed=key, argmax=True)
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

                # last action should always open gripper
                if t == num_timesteps - 1:
                    action[-1] = 1.0

                ### Store data ###
                obs["full_obs"].pop("images")
                obs_dict.append(obs["full_obs"])
                policy_out.append({"actions": action})
                full_images.append(obs["full_image"][0])
                image_formatted = np.concatenate((goal_image, image_obs), axis=0)
                images.append(Image.fromarray(image_formatted))

                ### Env step ###
                obs, rew, done, info = env.step(
                    action, last_tstep + STEP_DURATION, blocking=FLAGS.blocking
                )

                t += 1
        # Store final obs
        obs["full_obs"].pop("images")
        obs_dict.append(obs["full_obs"])
        full_images.append(obs["full_image"][0])
    except Exception as e:
        logging.error(traceback.format_exc())
        return None, None, None, None, False

    return (
        images,
        full_images,
        list_of_dicts_to_dict_of_lists(obs_dict),
        policy_out,
        True,
    )


def main(_):
    # load policy checkpoint
    agent, action_mean, action_std = load_policy_checkpoint()

    # load diffusion checkpoint
    sample_fn = load_diffusion_checkpoint()

    directory_name = "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.now())
    save_dir = os.path.join(FLAGS.save_dir, directory_name)
    os.makedirs(save_dir, exist_ok=True)

    process = multiprocessing.Process(target=log_floor_height, args=(save_dir,))
    process.start()
    process.join()

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
        'camera_topics': [IMTopic('/D435/color/image_raw', flip=True)]
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=128)

    rng = jax.random.PRNGKey(0)
    images = []
    for i in range(FLAGS.num_trajectories):
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

        # rollout subgoal
        logging.info(f"Subgoal {i}: rolling out...")
        rng, key = jax.random.split(rng)
        images, full_images, obs_dict, policy_out, rollout_success = rollout_subgoal(
            key, env, agent, image_goal, action_mean, action_std, FLAGS.num_timesteps
        )
        if not rollout_success:
            continue

        # saving data for training
        current_save_dir = os.path.join(save_dir, f"raw/traj_group0/traj{i}")
        image_save_dir = os.path.join(current_save_dir, "images0")
        os.makedirs(image_save_dir)
        for j, full_im in enumerate(full_images):
            im = Image.fromarray(full_im)
            imfilepath = os.path.join(image_save_dir, "im_{}.jpg".format(j))
            im.save(imfilepath)
        im = Image.fromarray(image_goal)
        imfilepath = os.path.join(current_save_dir, "goal.png")
        im.save(imfilepath)
        with open(os.path.join(current_save_dir, "obs_dict.pkl"), "wb") as handle:
            pickle.dump(obs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(current_save_dir, "policy_out.pkl"), "wb") as handle:
            pickle.dump(policy_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # saving video for visualization
        video_save_dir = os.path.join(
            FLAGS.save_dir, "gifs"
        )
        os.makedirs(video_save_dir, exist_ok=True)
        video_save_path = os.path.join(
            video_save_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.gif")
        )
        logging.info(f"Saving Video at {video_save_path}...")
        images[0].save(
            video_save_path,
            format="GIF",
            append_images=images[1:],
            save_all=True,
            duration=200,
            loop=0,
        )


if __name__ == "__main__":
    app.run(main)
