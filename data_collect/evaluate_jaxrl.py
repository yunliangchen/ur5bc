import sys
import os
import numpy as np
from PIL import Image
import cv2

from ur5.robot_env import RobotEnv
from ur5.jaxrl_model import LCBCModel

import nest_asyncio
import json
nest_asyncio.apply()


def main():
    env = RobotEnv()
    index = 1
    task_string = "Take the tiger out of the red bowl and put it in the grey bowl." # tiger pick and place (gripper initial position fixed)
    # task_string = "Sweep the blue cloth to the right side of the table." # cloth sweeping (gripper initial position random)
    # task_string = "Put the ranch bottle into the pot." # bottle pick and place (gripper initial position fixed)
    # task_string = "Pick up the blue cup and put it into the brown cup. " # cup stacking (gripper initial position random)

    saving_directory = "/home/lawrence/robotlerf/ur5bc/data/lcbc/tiger/"

    STEP = 20_000
    checkpoint_path = f"/home/lawrence/robotlerf/ur5bc/bc_baselines/lcbc_baseline/checkpoint_{STEP}"
    model = LCBCModel(checkpoint_path=checkpoint_path)

    while True:
        print("Starting index {}".format(index))
        # Ask the user whether to continue or not
        if input("Continue? (y/n): ") == "n":
            break

        env.reset(randomize=False, noise_type="joint") # True for cloth and cup
        # policy.reset()

        standard_output, action_traj, state_traj, obs_traj = env.evaluate_lcbc_model_trajectory(model, task_string, use_hand_image=False, \
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

    # action_traj = np.array(action_traj)
    # action_blocked = state_traj["action_blocked"]
    # starting_state = np.array(state_traj["joints"][0])
    # env.play_teleop_trajectory(action_traj, action_blocked, starting_state)

if __name__=="__main__":
    main()