import numpy as np
from ur5py.ur5 import UR5Robot
import subprocess
import cv2
import os
from autolab_core import CameraIntrinsics,RigidTransform
import time
import threading
import queue
import pickle
from PIL import Image
from ur5.robot_env import RobotEnv





import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.train_utils as TrainUtils

model_config_mapping = {
    "bottle_less_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/bottle_less_obs/20230815225106/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/bottle_less_obs/20230815225106/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/bottle_less_obs/20230815225106/inference_figures/",
        "less_obs": True
    },
    "bottle_more_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/bottle_more_obs/20230815233426/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/bottle_more_obs/20230815233426/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/bottle_more_obs/20230815233426/inference_figures/",
        "less_obs": False
    },
    "cloth_less_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cloth_less_obs/20230816005824/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cloth_less_obs/20230816005824/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cloth_less_obs/20230816005824/inference_figures/",
        "less_obs": True
    },
    "cloth_more_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cloth_more_obs/20230816013247/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cloth_more_obs/20230816013247/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cloth_more_obs/20230816013247/inference_figures/",
        "less_obs": False
    },
    "cup_less_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cup_less_obs/20230816023857/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cup_less_obs/20230816023857/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cup_less_obs/20230816023857/inference_figures/",
        "less_obs": True
    },
    "cup_more_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cup_more_obs/20230816032746/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cup_more_obs/20230816032746/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/cup_more_obs/20230816032746/inference_figures/",
        "less_obs": False
    },
    "tiger_less_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/tiger_less_obs/20230816050205/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/tiger_less_obs/20230816050205/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/tiger_less_obs/20230816050205/inference_figures/",
        "less_obs": True
    },
    "tiger_more_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/tiger_more_obs/20230816054332/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/tiger_more_obs/20230816054332/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/tiger_more_obs/20230816054332/inference_figures/",
        "less_obs": False
    },
    "alldata_less_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/alldata_less_obs/20230816070355/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/alldata_less_obs/20230816070355/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/alldata_less_obs/20230816070355/inference_figures/",
        "less_obs": True
    },
    "alldata_more_obs": {
        "model":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/alldata_more_obs/20230816100006/models/model_epoch_60.pth",
        "config":"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/alldata_more_obs/20230816100006/config.json",
        'folder':"/home/lawrence/robotlerf/ur5bc/bc_baselines/google_bc_baseline/alldata_more_obs/20230816100006/inference_figures/",
        "less_obs": False
    },
}





def main():
    env = RobotEnv()
    index = 5
    # task_string = "Take the tiger out of the red bowl and put it in the grey bowl." # tiger pick and place (gripper initial position fixed)
    # task_string = "Sweep the green cloth to the left side of the table." # cloth sweeping (gripper initial position random)
    # task_string = "Put the ranch bottle into the pot." # bottle pick and place (gripper initial position fixed)
    # task_string = "Pick up the blue cup and put it into the brown cup. " # cup stacking (gripper initial position random)

    model_name = "alldata_more_obs"
    saving_directory = "/home/lawrence/robotlerf/ur5bc/data/robomimic/alldata_more_obs/cup/"
    
    
    
    ckpt_path = model_config_mapping[model_name]['model']
    use_hand_image = not model_config_mapping[model_name]['less_obs']

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print("Using device: {}".format(device))

    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)



    while True:
        print("Starting index {}".format(index))
        # Ask the user whether to continue or not
        if input("Continue? (y/n): ") == "n":
            break

        env.reset(randomize=True, noise_type="joint") # True for cloth and cup
        # policy.reset()

        standard_output, action_traj, state_traj, obs_traj = env.evaluate_robomimic_model_trajectory(policy, use_hand_image, \
                                                                traj_index=index, saving_directory=saving_directory, gripper_history_window=(1, 0.5, 0.5))
        # print("Saving Trajectory ...")
        # os.makedirs(os.path.join(saving_directory, f"traj{index}"), exist_ok=True)
        # # save standard_output as a pkl file
        # with open(os.path.join(saving_directory, f"traj{index}", "standard_output.pkl"), "wb") as f:
        #     pickle.dump(standard_output, f)
        # # save action_traj as a pkl file
        # with open(os.path.join(saving_directory, f"traj{index}", "action_traj.pkl"), "wb") as f:
        #     pickle.dump(action_traj, f)
        # # save state_traj as a pkl file
        # with open(os.path.join(saving_directory, f"traj{index}", "state_traj.pkl"), "wb") as f:
        #     pickle.dump(state_traj, f)
        # # save obs_traj as a pkl file
        # with open(os.path.join(saving_directory, f"traj{index}", "obs_traj.pkl"), "wb") as f:
        #     pickle.dump(obs_traj, f)

        # save the images
        # os.makedirs(os.path.join(saving_directory, f"traj{index}", "images"), exist_ok=True)
        # for i in range(len(obs_traj["hand_image"])):
        #     cv2.imwrite(os.path.join(saving_directory, f"traj{index}", "images", f"hand_img_{i}.jpg"), obs_traj["hand_image"][i])
        # for i in range(len(obs_traj["third_person_image"])):
        #     Image.fromarray(obs_traj["third_person_image"][i][0]).save(os.path.join(saving_directory, f"traj{index}", "images", f"third_person_img_{i}.jpg"))
        index += 1

    action_traj = np.array(action_traj)
    action_blocked = state_traj["action_blocked"]
    starting_state = np.array(state_traj["joints"][0])
    env.play_teleop_trajectory(action_traj, action_blocked, starting_state)

if __name__=="__main__":
    main()