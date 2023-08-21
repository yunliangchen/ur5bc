import sys
import os
import numpy as np
from PIL import Image
from flax.training import checkpoints
import traceback
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
from jaxrl_m.data.bridge_dataset import multi_embed
# from jaxrl_m.data.language import load_mapping, lang_encode
# from jaxrl_m.vision.clip import process_image, process_text
import jaxrl_m.data.language

# from absl import app, flags, logging
# import time
# from datetime import datetime
import jax
import time
# import tensorflow as tf
import jax.numpy as jnp
# from flax.core import FrozenDict


import nest_asyncio
import json
nest_asyncio.apply()


# val_data = jnp.load('/home/lawrence/robotlerf/ur5bc/bc_baselines/lcbc_baseline/ur5_val/out.npy', allow_pickle=True)


# NAME = "all_multimodal_lawrence_lcbc_20230817_002757"
# wandb_run = f"widowx-gcrl/jaxrl_m_bridgedata/{NAME}"
# api = wandb.Api()
# run = api.run(wandb_run)
class run: config = json.load(open("/home/lawrence/robotlerf/ur5bc/bc_baselines/lcbc_baseline/run_config_ur5.json", "rb")) # use cached config to avoid wandb login


jaxrl_m.data.language.lang_to_code = {
 'Pick up the blue cup and put it into the brown cup.': 0,
 'Put the ranch bottle into the pot.': 1,
 'Take the tiger out of the red bowl and put it in the grey bowl.': 2,
 'Sweep the green cloth to the left side of the table.': 3,
 'Put the marker into the bowl.': 4
}

jaxrl_m.data.language.code_to_lang = {
 0: 'Pick up the blue cup and put it into the brown cup.',
 1: 'Put the ranch bottle into the pot.',
 2: 'Take the tiger out of the red bowl and put it in the grey bowl.',
 3: 'Sweep the green cloth to the left side of the table.',
 4: 'Put the marker into the bowl.'
}


def unnormalize_action(action, mean, std):
    return action * std + mean

class LCBCModel:
    def __init__(self, checkpoint_path):
        print("Loading model ...")
        example_batch = {
            "observations": {"image": jnp.zeros((10, 128, 128, 3), dtype=jnp.uint8)},
            "initial_obs": {"image": jnp.zeros((10, 128, 128, 3), dtype=jnp.uint8)},
            "goals": {
                "image": jnp.zeros((10, 128, 128, 3), dtype=jnp.uint8),
                "language": jnp.zeros((10, 512)),
                "language_mask": jnp.ones(10, dtype=bool),
            },
            "actions": jnp.zeros((10, 7), dtype=jnp.float32),
        }

        encoder_def = encoders[run.config["encoder"]](**run.config["encoder_kwargs"])
        task_encoder_defs = {
            k: encoders[run.config["task_encoders"][k]](
                **run.config["task_encoder_kwargs"][k]
            )
            for k in ("image", "language")
            if k in run.config["task_encoders"]
        }

        action_metadata = run.config["bridgedata_config"]["action_metadata"]
        self.action_mean = np.array(action_metadata["mean"])
        self.action_std = np.array(action_metadata["std"])


        rng = jax.random.PRNGKey(0)
        self.rng, construct_rng = jax.random.split(rng)
        agent = agents[run.config["agent"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            initial_obs=example_batch["initial_obs"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            task_encoder_defs=task_encoder_defs,
            **run.config["agent_kwargs"],
        )
        restored = checkpoints.restore_checkpoint(checkpoint_path, agent)
        assert restored is not agent
        self.agent = restored


        self.initial_obs = None

        print("Finished initializing the model.")


    def preprocess_image(self, image):
        image = Image.fromarray(np.uint8(image))
        image = image.resize((224, 224), Image.LANCZOS)
        return np.asarray(image).astype(np.uint8)

    def set_initial_obs(self, image):
        self.initial_obs = self.preprocess_image(image)
    


    def _predict_action(self, initial_obs, current_obs, language):
        return self.agent.sample_actions(
            dict(image=current_obs),
            dict(language=multi_embed(language)),
            dict(image=initial_obs),
            seed=self.rng,
            modality="language",
            argmax=True,
        )




    def predict_action(self, image, language):
        # images: (480, 640, 3) with values in [0, 255]

        # Run inference to obtain predicted actions for each image in the episode
        # The input to the model is the image and natural_language_embedding.
        image = self.preprocess_image(image)
        predicted_action = self._predict_action(self.initial_obs, image, language)
        action = np.append(np.append(unnormalize_action(predicted_action[:-1], self.action_mean[:-1], self.action_std[:-1]), predicted_action[-1]), 0) # append 0 becaues the model does not predict terminate action
        return action