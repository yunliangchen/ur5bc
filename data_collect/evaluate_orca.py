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
# from ur5.rt1_model import RT1Model
# from orca.utils.pretrained_utils import PretrainedModel

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def main():
    env = RobotEnv()
    print("robot env done")
    index = 0
    #task_string = "Take the tiger out of the red bowl and put it in the grey bowl." # tiger pick and place (gripper initial position fixed)
    #task_string = "Sweep the green cloth to the left side of the table." # cloth sweeping (gripper initial position random)
    task_string = "Put the ranch bottle into the pot." # bottle pick and place (gripper initial position fixed)
    #task_string = "Pick up the blue cup and put it into the brown cup. " # cup stacking (gripper initial position random)

    while True:
        print("Starting index {}".format(index))
        # Ask the user whether to continue or not
        # if input("Continue? (y/n): ") == "n":
            # break

        env.reset(randomize=False, noise_type="joint") # True for cloth and cup
        standard_output, action_traj, state_traj, obs_traj = env.evaluate_orca_model_trajectory(task_string, traj_index=index, saving_directory="/home/lawrence/robotlerf/ur5bc/data/orca")
        index += 1
if __name__=="__main__":
    main()