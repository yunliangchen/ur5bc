from typing import Iterable, Callable, Optional, Tuple
from jaxrl_m.common.typing import Array
import jax
import jax.numpy as jnp
import tensorflow as tf
from flax.core.frozen_dict import freeze
import wandb
import ml_collections
from flax.training import checkpoints
from denoising_diffusion_flax.model import EmaTrainState, create_model_def
from denoising_diffusion_flax.sampling import sample_loop
from denoising_diffusion_flax.scheduling import get_ddpm_params


def add_diffusion_goals(
    rng: jax.random.PRNGKey,
    data_loader: Iterable,
    goal_generator_fn: Callable[[jax.random.PRNGKey, Array], Array],
    generate_proportion: float,
) -> Iterable:
    """Add generated goals to a data loader.

    Args:
        rng: A Jax PRNG key.
        data_loader: A data loader (iterable).
        goal_generator_fn: A function that takes in a batch of batches of images
            and returns a batch of batches of goals.
        generate_proportion: The proportion of the batch to overwrite with generated goals.

    Returns:
        A new data loader that yields generated goals. Batch elements with generated goals
        will have the key "actor_loss_mask" set to False.
    """

    def iterator():
        nonlocal rng
        for batch in data_loader:
            batch = batch.unfreeze()
            num_devices, num_per_device = batch["observations"]["image"].shape[:2]
            mask_len = int(num_per_device * generate_proportion)
            rng, *gen_keys = jax.random.split(rng, num=num_devices + 1)
            gen_keys = jnp.asarray(gen_keys)
            batch["goals"]["image"] = (
                batch["goals"]["image"]
                .at[:, :mask_len]
                .set(
                    goal_generator_fn(
                        gen_keys, batch["observations"]["image"][:, :mask_len]
                    )
                )
            )

            # make sure proprio is not used
            batch["goals"]["proprio"] = (
                batch["goals"]["proprio"].at[:].set(float("nan"))
            )

            # set actor loss mask
            batch["actor_loss_mask"] = (
                jnp.ones_like(batch["rewards"], dtype=bool).at[:, :mask_len].set(False)
            )

            # make sure reward is -1 for fake goals
            batch["rewards"] = batch["rewards"].at[:, :mask_len].set(-1)

            yield freeze(batch)

    return iterator()


def load_diffusion_checkpoint(
    wandb_run_name: str,
    checkpoint_path: str,
    diffusion_sample_steps: Optional[int] = None,
    diffusion_eta: float = 0.0,
    diffusion_w: float = 0.0,
) -> Tuple[EmaTrainState, Callable[[jax.random.PRNGKey, EmaTrainState, Array], Array]]:
    assert tf.io.gfile.exists(
        checkpoint_path
    ), f"Checkpoint path {checkpoint_path} does not exist."

    # load config from wandb
    api = wandb.Api()
    run = api.run(wandb_run_name)
    config = ml_collections.ConfigDict(run.config["config"])

    # create model def
    model_def = create_model_def(config.model)

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
    ddpm_params = get_ddpm_params(config.ddpm)

    # compile sample loop
    if diffusion_sample_steps is None:
        num_timesteps = config.ddpm.timesteps
    else:
        num_timesteps = diffusion_sample_steps

    def sample_fn(rng, state, x):
        x = x.astype(jnp.float32) / 127.5 - 1.0
        x = sample_loop(
            rng,
            state,
            x,
            ddpm_params=ddpm_params,
            num_timesteps=num_timesteps,
            eta=diffusion_eta,
            w=diffusion_w,
            self_condition=config.ddpm.self_condition,
        )
        return jnp.clip(x * 127.5 + 127.5 + 0.5, 0, 255).astype(jnp.uint8)

    return state, sample_fn
