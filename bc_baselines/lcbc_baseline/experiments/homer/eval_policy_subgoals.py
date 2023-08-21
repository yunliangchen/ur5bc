import sys
from widowx_envs.widowx.widowx_env import BridgeDataRailRLPrivateWidowX
import os
import numpy as np
from PIL import Image
from flax.training import checkpoints
import traceback
import wandb
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
import matplotlib
from absl import app, flags, logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from datetime import datetime
from experiments.kevin.configs.bridgedata_config import get_config
import jax
import time
import tensorflow as tf

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("wandb_run_name", None, "Name of wandb run", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_string(
    "goal_image_path",
    None,
    "Path to a single goal image",
)
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", True, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", None, "Goal position")
flags.DEFINE_spaceseplist("initial_eep", None, "Initial position")
flags.DEFINE_bool("high_res", False, "Save high-res video and goal")
flags.DEFINE_integer("num_subgoals", 1, "Number of subgoals")

STEP_DURATION = 0.2
INITIAL_STATE_I = 0  # 7 #35 #1
FINAL_GOAL_I = -1  # -1 #70 #27
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 4

FIXED_STD = np.array([0.005, 0.005, 0.005, 0.0, 0.0, 0.0, 0.0])


def unnormalize_action(action, mean, std):
    return action * std + mean


def main(_):
    assert tf.io.gfile.exists(FLAGS.checkpoint_path)

    os.environ["WANDB_API_KEY"] = "12c8b2a3459714c891b05d7e3dcff4d7b19b41d2"  # Homer
    # os.environ["WANDB_API_KEY"] = "12e7b70d4a8aaba1d5f40a0660c173f53f05135a" # Kevin

    # set up environment
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
        "return_full_image": FLAGS.high_res,
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=128)

    # load image goal
    if FLAGS.goal_image_path is not None:
        image_goal = np.array(Image.open(FLAGS.goal_image_path))

    # load information from wandb
    api = wandb.Api()
    run = api.run(FLAGS.wandb_run_name)

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
    agent = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent)

    # goal sampling loop
    while True:
        # ask for new goal
        if input("new goal?") == "y":
            image_goals = []
            full_goal_images = []
            ns = 0
            while ns < FLAGS.num_subgoals:
                if FLAGS.goal_eep is not None:
                    assert isinstance(FLAGS.goal_eep, list)
                    goal_eep = [float(e) for e in FLAGS.goal_eep]
                else:
                    low_bound = [0.24, -0.1, 0.05, -1.57, 0]
                    high_bound = [0.4, 0.20, 0.15, 1.57, 0]
                    goal_eep = np.random.uniform(low_bound[:3], high_bound[:3])
                env._controller.open_gripper(True)
                try:
                    env._controller.move_to_state(goal_eep, 0, duration=1.5)
                    env._reset_previous_qpos()
                    ns += 1
                except Exception as e:
                    continue
                input("take image?")
                obs = env._get_obs()
                image_goals.append((
                    obs["image"].reshape(3, 128, 128).transpose(1, 2, 0) * 255
                ).astype(np.uint8))
                full_goal_images.append(obs['full_image'][0])

        input("start?")
        try:
            env.reset()
            env.start()
        except Exception as e:
            continue

        # move to initial position
        if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            env._controller.move_to_state(initial_eep, 0, duration=1.5)
            env._reset_previous_qpos()

        images = []
        full_images = []
        for goal_idx in range(FLAGS.num_subgoals):
            # do rollout
            obs = env._get_obs()
            last_tstep = time.time()
            t = 0
            # keep track of our own gripper state to implement sticky gripper
            is_gripper_closed = False
            num_consecutive_gripper_change_actions = 0
            try:
                while t < FLAGS.num_timesteps:
                    if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                        image_obs = (
                            obs["image"].reshape(3, 128, 128).transpose(1, 2, 0) * 255
                        ).astype(np.uint8)
                        if FLAGS.high_res:
                            full_images.append(Image.fromarray(obs["full_image"][0]))
                        obs = {"image": image_obs, "proprio": obs["state"]}
                        goal_obs = {
                            "image": image_goals[goal_idx],
                        }

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

                        if (
                            num_consecutive_gripper_change_actions
                            >= STICKY_GRIPPER_NUM_STEPS
                        ):
                            is_gripper_closed = not is_gripper_closed
                            num_consecutive_gripper_change_actions = 0

                        action[-1] = 0.0 if is_gripper_closed else 1.0

                        # remove degrees of freedom
                        if NO_PITCH_ROLL:
                            action[3] = 0
                            action[4] = 0
                        if NO_YAW:
                            action[5] = 0

                        # perform environment step
                        obs, rew, done, info = env.step(
                            action, last_tstep + STEP_DURATION, blocking=FLAGS.blocking
                        )

                        # save image
                        image_formatted = np.concatenate((image_goals[goal_idx], image_obs), axis=0)
                        images.append(Image.fromarray(image_formatted))

                        t += 1
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            checkpoint_name = "_".join(FLAGS.checkpoint_path.split("/")[-2:])
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{checkpoint_name}.gif",
            )
            print(f"Saving Video at {save_path}")
            images[0].save(
                save_path,
                format="GIF",
                append_images=images[1:],
                save_all=True,
                duration=200,
                loop=0,
            )
        # save high-res video
        if FLAGS.high_res:
            checkpoint_name = "_".join(FLAGS.checkpoint_path.split("/")[-2:])
            base_path = os.path.join(
                FLAGS.video_save_path,
                "high_res"
            )
            os.makedirs(base_path, exist_ok=True)
            print(f"Saving Video and Goal at {base_path}")
            curr_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            video_path = os.path.join(\
                base_path, 
                f"{curr_time}_{checkpoint_name}.gif"
            )
            full_images[0].save(
                video_path,
                format="GIF",
                append_images=full_images[1:],
                save_all=True,
                duration=200,
                loop=0,
            )
            for goal_idx in range(FLAGS.num_subgoals):
                goal_path = os.path.join(
                    base_path, 
                    f"{curr_time}_{checkpoint_name}_subgoal_{goal_idx}.png"
                )
                plt.imshow(full_goal_images[goal_idx])
                plt.axis('off')
                plt.savefig(goal_path)


if __name__ == "__main__":
    app.run(main)
