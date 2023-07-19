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
from ur5.model import RT1Model


def main():
    env = RobotEnv()
    index = 9
    # task_string = "Take the tiger out of the red bowl and put it in the grey bowl." # tiger pick and place (gripper initial position fixed)
    # task_string = "Sweep the green cloth to the left side of the table." # cloth sweeping (gripper initial position random)
    task_string = "Put the ranch bottle into the pot." # bottle pick and place (gripper initial position fixed)
    # task_string = "Pick up the blue cup and put it into the brown cup. " # cup stacking (gripper initial position random)


    # saved_model_path = '/home/lawrence/robotlerf/ur5bc/berkeley_ur5/xid_58975173/000631960' # bad
    # saved_model_path = '/home/lawrence/robotlerf/ur5bc/berkeley_ur5/xid_59180571/000568960' # best
    # saved_model_path = '/home/lawrence/robotlerf/ur5bc/berkeley_ur5/001009680' # mediocre
    # saved_model_path = '/home/lawrence/robotlerf/ur5bc/berkeley_ur5/xid_59466802/001048600' # mediocre (gripper often not trigger and just gets stuck)
    saved_model_path = '/home/lawrence/robotlerf/ur5bc/berkeley_ur5/xid_59470521/000832160' # second best

    # saving_directory = "/home/lawrence/robotlerf/ur5bc/data/rt1/test/xid_58975173"
    # saving_directory = "/home/lawrence/robotlerf/ur5bc/data/rt1/test/xid_59180571"
    # saving_directory = "/home/lawrence/robotlerf/ur5bc/data/rt1/bottle/001009680"
    # saving_directory = "/home/lawrence/robotlerf/ur5bc/data/rt1/bottle/xid_59466802" # scale_model_output=True
    saving_directory = "/home/lawrence/robotlerf/ur5bc/data/rt1/bottle/xid_59470521" # scale_model_output=True


    model = RT1Model(model_path=saved_model_path)
    task_embedding = model.compute_embedding(task_string)

    while True:
        print("Starting index {}".format(index))
        # Ask the user whether to continue or not
        if input("Continue? (y/n): ") == "n":
            break

        env.reset(randomize=False, noise_type="joint") # True for cloth and cup
        model.reset()

        standard_output, action_traj, state_traj, obs_traj = env.evaluate_model_trajectory(model, task_embedding, \
                                                                traj_index=index, saving_directory=saving_directory, scale_model_output=True)
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