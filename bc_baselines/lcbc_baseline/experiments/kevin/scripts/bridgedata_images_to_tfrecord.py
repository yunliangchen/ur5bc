from absl import app, flags, logging
import numpy as np
import os
import tqdm
import tensorflow as tf
from PIL import Image
import glob
from collections import defaultdict

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")


def squash(path):  # squash from 480x640 to 128x128 and flattened as a tensor
    im = Image.open(path)
    im = im.resize((128, 128), Image.ANTIALIAS)
    out = np.asarray(im).astype(np.uint8)
    return out


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def process(images, out_path):
    trajs = []
    for i in range(len(images) - 1):
        trajs.append(images[i : i + 2])
    trajs = np.array(trajs)

    if tf.io.gfile.exists(out_path):
        if FLAGS.overwrite:
            logging.info(f"Deleting {out_path}")
            tf.io.gfile.remove(out_path)
        else:
            logging.info(f"Skipping {out_path}")
            return

    if len(trajs) == 0:
        logging.info(f"Skipping {out_path}, empty")
        return

    tf.io.gfile.makedirs(FLAGS.output_path)

    with tf.io.TFRecordWriter(out_path) as writer:
        for traj in trajs:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "observations/images0": tensor_feature(traj),
                        "observations/state": tensor_feature(
                            np.zeros((2, 7), dtype=np.float32)
                        ),
                        "next_observations/images0": tensor_feature(traj),
                        "next_observations/state": tensor_feature(
                            np.zeros((2, 7), dtype=np.float32)
                        ),
                        "actions": tensor_feature(np.zeros((2, 7), dtype=np.float32)),
                        "terminals": tensor_feature(np.zeros((2,), dtype=np.bool_)),
                        "truncates": tensor_feature(np.zeros((2,), dtype=np.bool_)),
                    }
                )
            )
            writer.write(example.SerializeToString())


TRAIN_SPLIT = 0.9


def main(_):
    # sort
    image_paths = sorted(
        glob.glob(os.path.join(FLAGS.input_path, "*")),
        key=lambda s: int(s.split(".")[-2].split("_")[-1]),
    )

    # load and resize
    images = np.array(list(map(squash, image_paths)))

    # train/val split
    train_images = images[: int(len(images) * TRAIN_SPLIT)]
    val_images = images[int(len(images) * TRAIN_SPLIT) :]

    logging.info(f"Train: {len(train_images)}")
    logging.info(f"Val: {len(val_images)}")

    process(train_images, f"{FLAGS.output_path}/train/out.tfrecord")
    process(val_images, f"{FLAGS.output_path}/val/out.tfrecord")


if __name__ == "__main__":
    app.run(main)
