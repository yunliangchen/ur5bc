from absl import app, flags
import glob
import numpy as np
import os
import h5py
import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")


def process(path):
    arr = np.load(path, allow_pickle=True)
    dirname = os.path.dirname(os.path.normpath(path))
    dirname = os.path.join(FLAGS.output_path, *dirname.split(os.sep)[-4:])
    out_path = f"{dirname}/out.hdf5"

    if os.path.exists(out_path):
        if FLAGS.overwrite:
            print("Deleting ", out_path)
            os.remove(out_path)
        else:
            print("Skipping ", out_path)
            return

    if len(arr) == 0:
        print("Skipping ", path, ", empty")
        return

    os.makedirs(dirname, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f["observations/images0"] = np.array(
            [o["images0"] for t in arr for o in t["observations"]]
        )
        f["observations/state"] = np.array(
            [o["state"] for t in arr for o in t["observations"]]
        )
        f["next_observations/images0"] = np.array(
            [o["images0"] for t in arr for o in t["next_observations"]]
        )
        f["next_observations/state"] = np.array(
            [o["state"] for t in arr for o in t["next_observations"]]
        )
        f["actions"] = np.array(
            [a for t in arr for a in t["actions"]], dtype=np.float32
        )
        f["terminals"] = np.zeros(f["actions"].shape[0], dtype=np.bool_)
        f["truncates"] = np.zeros(f["actions"].shape[0], dtype=np.bool_)
        f["steps_remaining"] = np.zeros(f["actions"].shape[0], dtype=np.uint32)
        end = 0
        for traj in arr:
            start = end
            end += len(traj["actions"])
            f["truncates"][end - 1] = True
            f["steps_remaining"][start:end] = np.arange(end - start)[::-1]


def main(_):
    paths = glob.glob(f"{FLAGS.input_path}/*/*")
    for path in tqdm.tqdm(paths):
        train_path = f"{path}/train/out.npy"
        val_path = f"{path}/val/out.npy"
        process(train_path)
        process(val_path)


if __name__ == "__main__":
    app.run(main)
