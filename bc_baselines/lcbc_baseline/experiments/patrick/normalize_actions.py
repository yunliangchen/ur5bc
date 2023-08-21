import numpy as np
from absl import flags, app
from jaxrl_m.utils.tf_utils import load_tf_dataset
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)


def main(_):
    dataset = load_tf_dataset(FLAGS.data_path)
    for f in iter(dataset):
        actions.append(f["actions"][:])
    actions = np.concatenate(actions)
    metadata = {}
    metadata["mean"] = np.mean(actions, axis=0)
    metadata["std"] = np.std(actions, axis=0)
    # don't normalize gripper
    metadata["mean"][6] = 0
    metadata["std"][6] = 1
    np.save(tf.io.gfile.join(FLAGS.data_path, "metadata.npy"), metadata)


if __name__ == "__main__":
    app.run(main)
