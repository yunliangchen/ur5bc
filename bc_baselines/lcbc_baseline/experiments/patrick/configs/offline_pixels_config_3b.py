import ml_collections


def get_roboverse_gc_iql_config():
    return ml_collections.ConfigDict(
        dict(
            agent_kwargs=dict(
                network_kwargs=dict(
                    filters=(64, 64, 64),
                    kernels=(3, 3, 3),
                    strides=(1, 1, 1),
                    pooling_method="avg",
                    hidden_dims=(256, 256),  # TODO
                ),
                policy_kwargs=dict(
                    tanh_squash_distribution=False,
                    state_dependent_std=False,
                ),
                optim_kwargs=dict(
                    learning_rate=3e-4,
                ),
                discount=0.98,
                expectile=0.7,
                temperature=1.0,
                target_update_rate=0.002,
                shared_encoder=True,
                shared_goal_encoder=True,
                early_goal_concat=True,
                use_proprio=False,
            ),
            encoder="resnetv1-18-bridge-lite-v2",
            encoder_kwargs=dict(
                num_filters=16,  # TODO
                # pooling_method='avg',
                pooling_method=None,
                add_spatial_coordinates=True,
            ),
            dataset_kwargs=dict(
                use_proportion=1.0,
                shuffle_buffer_size=10000,
                prefetch_num_batches=5,
                chunk_size=1000,
                goal_relabel_reached_proportion=0.1,
                augment=True,
                augment_kwargs=dict(
                    random_resized_crop=dict(scale=[0.9, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.05],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.1],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            ),
            ## Online Dataset
            replay_buffer_kwargs=dict(
                capacity=int(1e6),
                fraction_next=0.1,
                fraction_uniform=0.6,
                fraction_last=0.0,
                fraction_negative=0.0,
                reward_key="image",
                reward_thresh=0.3,
                augment=True,
                augment_next_obs_goal_differently=False,
                augment_kwargs=dict(
                    random_resized_crop=dict(scale=[0.9, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.05],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.1],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            ),
        )
    )


def get_roboverse_affordance_config():
    return ml_collections.ConfigDict(
        dict(
            agent_kwargs=dict(
                affordance_kwargs=dict(
                    encoder_hidden_dims=[256, 256, 256],
                    latent_dim=8,
                    decoder_hidden_dims=[256, 256, 256],
                ),
                affordance_beta=0.02,
            ),
            encoder="resnetv1-18-bridge-lite",
            encoder_kwargs=dict(
                pooling_method="avg",
                add_spatial_coordinates=True,
            ),
            dataset_kwargs=dict(
                use_proportion=1.0,
                shuffle_buffer_size=10000,
                prefetch_num_batches=5,
                chunk_size=1000,
                goal_relabel_reached_proportion=0.0,
                augment=True,
                augment_kwargs=dict(
                    random_resized_crop=dict(scale=[0.9, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.05],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.1],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            ),
        )
    )


def get_config(config_string):
    possible_structures = {
        "roboverse_gc_iql": ml_collections.ConfigDict(
            {
                "model_constructor": "create_iql_learner",
                "model_config": get_roboverse_gc_iql_config(),
            }
        ),
        "roboverse_affordance": ml_collections.ConfigDict(
            {
                "model_constructor": "create_affordance_learner",
                "model_config": get_roboverse_affordance_config(),
            }
        ),
    }
    return possible_structures[config_string]
