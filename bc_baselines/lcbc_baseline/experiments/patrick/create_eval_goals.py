import string
import numpy as np
from absl import flags, app, logging
from jaxrl_m.utils.tf_utils import load_tf_dataset
import tensorflow as tf
import jax

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_integer("num_goals", None, "Number of goals", required=True)
flags.DEFINE_string("accept_trajectory_key", None, "Success key", required=True)


def main(_):
    dataset = load_tf_dataset(FLAGS.data_path)
    data = []
    for traj in iter(dataset):
        if traj["infos"][FLAGS.accept_trajectory_key][-1]:
            data.append(traj)

    logging.info(f"Number of successful trajectories: {len(data)}")
    data = np.random.choice(data, size=FLAGS.num_goals, replace=False)

    # turn list of dicts into dict of lists, selecting last element of each trajectory
    data = jax.tree_map(lambda *xs: np.array(xs)[:, -1], *data)

    # decode strings
    data["infos"]["target_object"] = [
        s.decode("UTF-8") for s in data["infos"]["target_object"]
    ]
    data["infos"]["object_names"] = [
        [s.decode("UTF-8") for s in names] for names in data["infos"]["object_names"]
    ]

    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.data_path, "eval_goals.npy"), "wb"
    ) as f:
        np.save(f, data)


if __name__ == "__main__":
    app.run(main)
