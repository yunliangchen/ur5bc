from functools import partial
from absl import app, flags
from jaxrl_m.agents.continuous.gc_iql import create_iql_learner
from jaxrl_m.vision import encoders as vision_encoders
from jaxrl_m.envs.bridge import visualization
from flax.training import checkpoints
import numpy as np
import wandb
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint_path", None, "Path to checkpoint to resume from.", required=True
)
flags.DEFINE_string(
    "wandb_run_name", None, "Name of wandb run to resume from.", required=True
)
flags.DEFINE_string("demo_path", None, "Path to demo to visualize.", required=True)
flags.DEFINE_integer("demo_id", None, "ID of demo to visualize.", required=True)
flags.DEFINE_string(
    "output_path", None, "Path to save visualization to.", required=True
)


def main(_):
    # restore agent
    api = wandb.Api()
    run = api.run(FLAGS.wandb_run_name)

    assert run.config["model_constructor"] == "create_iql_learner"
    model_config = run.config["model_config"]
    encoder_def = vision_encoders[model_config["encoder"]](
        **model_config["encoder_kwargs"]
    )
    agent = create_iql_learner(
        seed=0,
        encoder_def=encoder_def,
        observations=np.zeros((128, 128, 3), dtype=np.uint8),
        goals=np.zeros((128, 128, 3), dtype=np.uint8),
        actions=np.zeros(7, dtype=np.float32),
        **model_config["agent_kwargs"]
    )
    agent_dict = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, target=None)
    params = agent_dict["model"]["params"]
    target_params = agent_dict["target_model"]["params"]
    agent = agent.replace(
        model=agent.model.replace(params=params),
        target_model=agent.target_model.replace(params=target_params),
    )

    # load action metadata
    action_mean = np.array(run.config["bridgedata_config"]["action_metadata"]["mean"])
    action_std = np.array(run.config["bridgedata_config"]["action_metadata"]["std"])

    # load demo
    demo = np.load(FLAGS.demo_path, allow_pickle=True)[FLAGS.demo_id]
    image_goal = demo["observations"][-1]["images0"]

    # create demo data
    demo_batched = {
        "observations": {
            "image": np.array([o["images0"] for o in demo["observations"]]),
            "proprio": np.array([o["state"] for o in demo["observations"]]),
        },
        "next_observations": {
            "image": np.array([o["images0"] for o in demo["next_observations"]]),
            "proprio": np.array([o["state"] for o in demo["next_observations"]]),
        },
        "goals": {
            "image": np.array([image_goal for _ in range(len(demo["actions"]))]),
        },
        "rewards": np.array(
            [
                0 if i == len(demo["actions"]) - 1 else -1
                for i in range(len(demo["actions"]))
            ]
        ),
        "masks": np.ones(len(demo["actions"])),
    }
    relabeled_actions = (
        demo_batched["next_observations"]["proprio"]
        - demo_batched["observations"]["proprio"]
    )
    relabeled_actions[:, -1] = np.array(demo["actions"])[:, -1]
    demo_batched["actions"] = (relabeled_actions - action_mean) / action_std

    # run inference
    gripper_close_val = np.min(demo_batched["actions"][:, -1])
    print("gripper close val", gripper_close_val)
    metrics = agent.get_debug_metrics(demo_batched, gripper_close_val)
    metrics["gripper_action"] = demo_batched["actions"][:, -1]

    # create visualization
    visualize_keys = [
        "gripper_action",
        "online_v",
        "advantage",
        "qf_advantage",
        "gripper_close_q",
        "gripper_close_adv",
    ]
    what_to_visualize = [
        partial(visualization.visualize_metric, metric_name=k) for k in visualize_keys
    ]
    image = visualization.make_visual(
        demo_batched["observations"]["image"],
        metrics,
        what_to_visualize=what_to_visualize,
    )
    plt.imsave(FLAGS.output_path, image)


if __name__ == "__main__":
    app.run(main)
