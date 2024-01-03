from copy import deepcopy
import os
import subprocess
import gym
import numpy as np
import pyrealsense2 as rs
import cv2
import time
import threading
import queue
import torchvision.transforms as transforms
from pynput import keyboard
from autolab_core import RigidTransform
from autolab_core.transformations import quaternion_from_euler, quaternion_matrix, euler_matrix, quaternion_from_matrix, translation_from_matrix
from ur5py.ur5 import UR5Robot
from misc.time import time_ms
from ur5.ur5_kinematics import UR5Kinematics
from matplotlib import pyplot as plt
from utils.async_writer import AsyncWrite, AsyncWriteStandardizedFormat
from ur5.spacemouse import SpaceMouseRobotController
from misc.policy_wrapper import FrameStackWrapper

class WebCam:
    def __init__(self, port_num = 0, resolution = "480p"):

        # v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=0 --set-ctrl=focus_absolute=10 --set-ctrl=white_balance_temperature_auto=0 -c exposure_auto_priority=0 --set-ctrl=saturation=60 --set-ctrl=gain=140
        # command = [
        #             "v4l2-ctl",
        #             "-d /dev/video{}".format(port_num),
        #             "--set-ctrl=focus_auto=0"
        #             "--set-ctrl=focus_absolute=10",
        #             "--set-ctrl=white_balance_temperature_auto=0"
        #             # "--set-ctrl=exposure_auto=0",
        #             "-c exposure_auto_priority=0",
        #             # "--set-ctrl=exposure_absolute=100",
        #             "--set-ctrl=saturation=60", "--set-ctrl=gain=140",

        #         ] # See `v4l2-ctl -d /dev/video0 --list-ctrls` for a list of available controls
        # FNULL = open(os.devnull, "w")
        # subprocess.call(
        #     command,
        #     stdout=FNULL,
        #     stderr=subprocess.STDOUT,
        # )

        self.cam = cv2.VideoCapture(port_num)
        if resolution == "480p":
            self.make_480p()
        elif resolution == "720p":
            self.make_720p()
        elif resolution == "1080p":
            self.make_1080p()
        
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        print("here")

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
            
    def make_1080p(self):
        self.cam.set(3, 1920)
        self.cam.set(4, 1080)

    def make_720p(self):
        self.cam.set(3, 1280)
        self.cam.set(4, 720)

    def make_480p(self):
        self.cam.set(3, 640)
        self.cam.set(4, 480)

    def change_res(self, width, height):
        self.cam.set(3, width)
        self.cam.set(4, height)


class DepthCam:
    def __init__(self, port_num = 2, resolution = "480p"):
        ctx = rs.context()
        if len(ctx.devices) > 0:
            for d in ctx.devices:
                print ('Found device: ', \
                        d.get_info(rs.camera_info.name), ' ', \
                        d.get_info(rs.camera_info.serial_number))
        else:
            print("No Intel Device connected")

        self.pipeline = rs.pipeline()
        config = rs.config()
    
        # config.enable_device(str(rs.camera_info.serial_number))
        self.profile = self.pipeline.start(config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale() # convert depth to meters

        # Skip 5 first frames to give the Auto-Exposure time to adjust
        for x in range(5):
            self.pipeline.wait_for_frames()

    
    #     self.q = queue.Queue()
    #     t = threading.Thread(target=self._reader)
    #     t.daemon = True
    #     t.start()

    # # read frames as soon as they are available, keeping only most recent one
    # def _reader(self):
    #     while True:
    #         frames = self.pipeline.wait_for_frames(timeout_ms = 1000)
    #         # if not ret:
    #         #     break
    #         if not self.q.empty():
    #             try:
    #                 self.q.get_nowait()   # discard previous (unprocessed) frame
    #             except queue.Empty:
    #                 pass
    #         self.q.put(frames)

    def read(self):
        frames = self.pipeline.wait_for_frames(timeout_ms = 1000)
        # frames = self.q.get()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() # not aligned


        color_frame_np = np.asanyarray(color_frame.get_data())
        depth_frame_np = np.asanyarray(depth_frame.get_data())
        
        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Update color and depth frames:
        aligned_depth_frame = frames.get_depth_frame()
        aligned_depth_frame_np = (np.asanyarray(aligned_depth_frame.get_data(), np.float32) * self.depth_scale)

        # plt.imsave("/home/lawrence/robotlerf/ur5bc/data/depth_images/color_{}.png".format(time.time()), color_frame_np)
        # plt.imsave("/home/lawrence/robotlerf/ur5bc/data/depth_images/depth_{}.png".format(time.time()), aligned_depth_frame_np)
        return [color_frame_np.copy(), aligned_depth_frame_np.copy()] # (480, 640, 3), (480, 640)
            
    def __del__(self):
        self.pipeline.stop()



class RobotEnv(gym.Env):
    def __init__(self, camera_kwargs={}):
        # Initialize Gym Environment
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

        # Robot Configuration
        # self.reset_joints = np.array([-np.pi, -105/180*np.pi, 105/180*np.pi, -np.pi/2, -np.pi/2, np.pi])
        # self.reset_joints = np.array([-190/180*np.pi, -81/180*np.pi, 131/180*np.pi, -141/180*np.pi, -np.pi/2, 164/180*np.pi])
        self.reset_joints = np.array([-190/180*np.pi, -102/180*np.pi, 141/180*np.pi, -130/180*np.pi, -np.pi/2, 168/180*np.pi])
        self.joint_randomize_low = -np.array([1, 1, 0.6, 1, 1, 1]) * 10 / 180 * np.pi
        self.joint_randomize_high = np.array([1, 1, 0.3, 1, 1, 1]) * 10 / 180 * np.pi
        self.cartesian_randomize_low = np.array([-0.1, -0.1, -0.1, -0.3, -0.3, -0.3])
        self.cartesian_randomize_high = np.array([0.1, 0., 0.1, 0.3, 0.3, 0.3])
        self.control_hz = 5

        # import pdb; pdb.set_trace()
        self._robot = UR5Robot(gripper=True)
        self._robot.set_tcp(RigidTransform(translation=[0,0.0,.17]))
        self._robot.gripper.set_speed(100) # from 0 to 100 %
        self._kinematics = UR5Kinematics(urdf_filename="/ur5.urdf")
        self._kinematics.set_tcp(RigidTransform(translation=[0,0.0,.07], from_frame="tcp", to_frame="ee_link"))
        self._gripper_is_closed = None
        self._gripper_being_blocked = False

        # Create Cameras
        # import pdb; pdb.set_trace()
        self.webcam = WebCam(**camera_kwargs)
        self.depthcam = DepthCam(**camera_kwargs)

        # Initialize the space mouse
        # self._controller = SpaceMouseRobotController()
        time.sleep(0.1)

        # Reset Robot
        self.reset()

    # def step(self, action):
    #     # The action space is 8D, with 6 for the delta gripper pose, 1 for opening/closing the gripper, and 1 for termination action
    #     assert (action.max() <= 1) and (action.min() >= -1)

    #     # Update Robot
    #     action_info = self.update_robot(action, action_space=self.action_space)

    #     # Return Action Info
    #     return action_info

    def reset(self, randomize=False, noise_type="joint"):
        self._robot.gripper.open()
        self._gripper_is_closed = False

        reset_joints = self.reset_joints.copy()
        if randomize:
            if noise_type == "joint":
                noise = np.random.uniform(low=self.joint_randomize_low, high=self.joint_randomize_high)
                reset_joints += noise
            elif noise_type == "cartesian":
                noise = np.random.uniform(low=self.cartesian_randomize_low, high=self.cartesian_randomize_high)
                pose = self._kinematics.fk(reset_joints)
                delta_pose = RigidTransform(translation=noise[:3], rotation=euler_matrix(noise[3], noise[4], noise[5], axes="ryxz")[:3, :3], from_frame="tcp", to_frame="tcp")

                # current_pose = self._robot.get_pose()
                # current_pose.from_frame = "tcp"
                noisy_pose = pose * delta_pose
                # self._robot.move_pose(new_pose, vel=1, acc=10)
                reset_joints = self._kinematics.ik(noisy_pose, reset_joints)
        
        self._robot.move_joint(reset_joints)

    

    def read_cameras(self):
        timestamp_dict = {}
        timestamp_dict["read_start"] = time_ms() - self.trajectory_start_time
        third_person_img = self.depthcam.read()
        ############### Temporary ###############
        hand_img = self.webcam.read()
        # hand_img = third_person_img[0].copy()
        #########################################
        timestamp_dict["read_end"] = time_ms() - self.trajectory_start_time
        return hand_img, third_person_img, timestamp_dict

    def get_state(self):
        timestamp_dict = {}
        timestamp_dict["read_start"] = time_ms() - self.trajectory_start_time
        robot_pose = self._robot.get_pose()
        robot_joints = self._robot.get_joints()
        timestamp_dict["read_end"] = time_ms() - self.trajectory_start_time
        return robot_pose, robot_joints, timestamp_dict

    def get_camera_extrinsics(self, state_dict):
        # Adjust gripper camere by current pose
        pass

    def get_observation(self):
        state_dict = {}
        obs_dict = {}

        # Robot State #
        robot_pose, robot_joints, robot_timestamp_dict = self.get_state()
        state_dict["robot_pose"] = robot_pose
        state_dict["robot_joints"] = robot_joints
        state_dict["timestamp"] = robot_timestamp_dict

        # Camera Readings #
        hand_img, third_person_img, camera_timestamp_dict = self.read_cameras()
        obs_dict["hand_image"] = hand_img
        obs_dict["third_person_image"] = third_person_img
        obs_dict["timestamp"] = camera_timestamp_dict


        return state_dict, obs_dict

    def infer_action_from_observation(self, previous_pose:RigidTransform, new_pose:RigidTransform, previous_gripper_is_closed:bool, new_gripper_is_closed:bool):
        '''
        converts from rigidtransform pose to the UR format pose (x,y,z,rx,ry,rz)
        # gripper_action = 1 if gripper_is_closed changes from False to True, i.e., closing the gripper
        # gripper_action = -1 if gripper_is_closed changes from True to False, i.e., opening the gripper
        # no change = 0 if gripper_is_closed does not change
        '''
        previous_state = previous_pose.translation.tolist() + previous_pose.axis_angle.tolist()
        new_state = new_pose.translation.tolist() + new_pose.axis_angle.tolist()
        delta_state = list(np.array(new_state) - np.array(previous_state))

        gripper_action = int(new_gripper_is_closed) - int(previous_gripper_is_closed)
        return delta_state + [gripper_action] + [0] # 0 is the termination action



    def record_teleop_trajectory(self, task_string, traj_index=0, saving_directory="/home/lawrence/robotlerf/ur5bc/data/raw/teleop/"):
        # Listen to the state from space mouse
        # Convert the state to ur5 action at 10 Hz
        # Execute the action
        # Record the action and state

        input("enter to start recording traj")

        gripper_is_being_blocked = False
        def trigger_gripper():
            print("Enter into trigger gripper at time {}".format(time_ms() - self.trajectory_start_time))
            nonlocal gripper_is_being_blocked
            def update_gripper_status():
                time.sleep(1.3)
                print("Start")
                self._gripper_is_closed = not self._gripper_is_closed
                print("End")
                    
            gripper_is_closed = self._gripper_is_closed
            t = threading.Thread(target=update_gripper_status) # start a child thread
            t.daemon = True
            t.start()
            gripper_is_being_blocked = True
            print("Start blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))
            if gripper_is_closed:
                self._robot.gripper.open()
            else:
                self._robot.gripper.close()
            gripper_is_being_blocked = False
            print("End blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))


        def keep_recording():
            nonlocal state_traj, obs_traj, action_traj, standard_output, gripper_is_being_blocked, last_timestep, i
            gripper_is_being_blocked = True # otherwise it will not enter the loop before trigger_gripper set gripper_is_being_blocked to True
            # if time.time() - last_timestep < 1 / self.control_hz:
            #     time.sleep(last_timestep + 1 / self.control_hz - time.time())
            # else:
            #     print("gripper_is_being_blocked")
            #     print("Warning: Control Loop is running slower than desired frequency")
            #     print(time.time() - last_timestep, " seconds has passed")
            while gripper_is_being_blocked:
                # print("Doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)

                # continue recording the state and observation at the same frequency
                state_dict, obs_dict = self.get_observation()
                state_traj["poses"].append(state_dict["robot_pose"].matrix)
                state_traj["joints"].append(state_dict["robot_joints"])
                state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
                state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
                state_traj["gripper_closed"].append(self._gripper_is_closed)
                state_traj["action_blocked"].append(True)
                obs_traj["hand_image"].append(obs_dict["hand_image"])
                obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
                obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
                obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])
                writer = AsyncWrite(obs_dict["hand_image"], obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
                writer.start()

                action_traj.append([0,0,0,0,0,0,0,0]) # do nothing
                
                # for the standardized format:
                pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
                robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [True] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
                image = obs_dict["third_person_image"][0]
                task = task_string
                standard_output["robot_state"].append(robot_state)
                standard_output["image"].append(image)
                standard_output["task"].append([task])
                standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
                standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1
            print("Stop doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)
            return

        

        stop = False
        gripper_action = False
        state_traj = {"poses": [], "joints": [], "gripper_closed": [], "action_blocked": [], "timestamp":{"read_start": [], "read_end":[]}} # "poses": [RigidTransforms], "joints": Array(T, 6); "timestamp"/ "read_start": Array(T,), "read_end": Array(T,)
        obs_traj = {"hand_image": [], "third_person_image": [], "timestamp":{"read_start": [], "read_end":[]}}
        action_traj = []
        standard_output = {"robot_state": [], "action": [], "image": [], "task": [], "other": {"hand_image": [], "third_person_image": []}}
        last_timestep = time.time()
        i = 0
        self.trajectory_start_time = time_ms()

        while True:
            state_dict, obs_dict = self.get_observation()
            state_traj["poses"].append(state_dict["robot_pose"].matrix)
            state_traj["joints"].append(state_dict["robot_joints"])
            state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
            state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
            state_traj["gripper_closed"].append(self._gripper_is_closed)
            state_traj["action_blocked"].append(False)
            obs_traj["hand_image"].append(obs_dict["hand_image"])
            obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
            obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
            obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])

            writer = AsyncWrite(obs_dict["hand_image"], obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
            writer.start()

            # for the standardized format:
            pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
            robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [False] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
            image = obs_dict["third_person_image"][0]
            task = task_string
            standard_output["robot_state"].append(robot_state)
            standard_output["image"].append(image)
            standard_output["task"].append([task])
            standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
            standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            

            if self._controller.button_1_pressed:
                stop = True
                action_traj.append([0,0,0,0,0,0,0,1]) # termination action
                break
            elif self._controller.button_0_pressed:
                last_timestep = time.time()
                # print("Go into the button pressed if statement at time ", time_ms() - self.trajectory_start_time, " milliseconds")
                if self._gripper_is_closed:
                    action_traj.append([0,0,0,0,0,0,-1,0]) # open gripper
                else:
                    action_traj.append([0,0,0,0,0,0,1,0]) # close gripper
                
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

                threadList = [threading.Thread(target=trigger_gripper), threading.Thread(target=keep_recording)]
                for threads in threadList:
                    threads.start()
                print("All threads started at time ", time_ms() - self.trajectory_start_time, " milliseconds")
                for threads in threadList:
                    threads.join()
                print("All threads ended at time ", time_ms() - self.trajectory_start_time, " milliseconds")
                
            else:    
                # import pdb; pdb.set_trace()
                # TODO: Tune the control gain
                self._controller_state = self._controller.current_action # x,y,z,roll,pitch,yaw
                translation = np.array(self._controller_state[:3]) * 0.02 # 1 will be 2cm
                translation[0] *= -1
                translation[-1] *= -1
                rotation = np.array(self._controller_state[3:]) / 15
                gripper_action = 0
                action = np.array([*translation, *rotation, gripper_action, 0]).tolist()
                action_traj.append(action)
                # q = quaternion_from_euler(action[3], action[4], action[5], 'ryxz')
                
                # print(euler_matrix(action[3], action[4], action[5], axes="syxz")[:3, :3] - euler_matrix(action[3], action[4], action[5], axes="ryxz")[:3, :3])
                delta_pose = RigidTransform(translation=translation, rotation=euler_matrix(action[3], action[4], action[5], axes="ryxz")[:3, :3], from_frame="tcp", to_frame="tcp")

                # Update Robot
                current_pose = self._robot.get_pose()
                current_pose.from_frame = "tcp"
                new_pose = current_pose * delta_pose
                # self._robot.move_pose(new_pose, vel=1, acc=10)
                self._robot.servo_pose(new_pose, 0.01, 0.2, 100)
                # with open('error_log.txt', 'a') as f:
                #     f.write(str(new_pose.translation - self._robot.get_pose().translation) + str(new_pose.rotation - self._robot.get_pose().rotation))
                #     f.write("\n")
            
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

        # Turn everything into numpy arrays
        standard_output["robot_state"] = np.array(standard_output["robot_state"]) # (T, 15)
        standard_output["image"] = np.stack(standard_output["image"], axis=0) # (T, 480, 640, 3)
        standard_output["action"] = np.array(action_traj) # (T, 8)
        standard_output["task"] = np.array(standard_output["task"]) # (T, 1)
        standard_output["other"]["hand_image"] = np.stack(standard_output["other"]["hand_image"], axis=0) # (T, 480, 640, 3)
        standard_output["other"]["third_person_image"] = np.stack(standard_output["other"]["third_person_image"], axis=0) # (T, 480, 640, 4)
        
        return standard_output, action_traj, state_traj, obs_traj


    def robot_trigger_gripper(self, command="close"):
        def update_gripper_status():
            time.sleep(1.3)
            print("Start")
            self._gripper_is_closed = not self._gripper_is_closed
            print("End")
        
        if command == "open" and self._gripper_is_closed:
            t = threading.Thread(target=update_gripper_status) # start a child thread
            t.daemon = True
            t.start()
            self._gripper_being_blocked = True
            self._robot.gripper.open()
            self._gripper_being_blocked = False
        elif command == "close" and not self._gripper_is_closed:
            t = threading.Thread(target=update_gripper_status) # start a child thread
            t.daemon = True
            t.start()
            self._gripper_being_blocked = True
            self._robot.gripper.close()
            self._gripper_being_blocked = False


    def step(self, action):
        """
        Execute the action
        """
        if self._gripper_being_blocked:
            print("Gripper is being blocked, cannot execute action")
            return 
        else:
            if action[-2] == 0:
                delta_pose = RigidTransform(translation=np.array([action[0], action[1], action[2]]), rotation=euler_matrix(action[3], action[4], action[5], axes="ryxz")[:3, :3], from_frame="tcp", to_frame="tcp")

                # Update Robot
                current_pose = self._robot.get_pose()
                current_pose.from_frame = "tcp"
                new_pose =  current_pose * delta_pose
                # self._robot.servo_pose(new_pose, 0.002, 0.1, 100) # 10 Hz
                self._robot.servo_pose(new_pose, 0.01, 0.2, 100) # 5 Hz
            elif action[-2] == 1:
                t = threading.Thread(target=self.trigger_gripper, args=("close",))
            elif action[-2] == -1:
                t = threading.Thread(target=self.trigger_gripper, args=("open",))
                
                t.daemon = True
                t.start()


    def evaluate_robomimic_model_trajectory(self, model, use_hand_image=False, traj_index=0, saving_directory="/home/lawrence/robotlerf/ur5bc/data/robomimic/", gripper_history_window = (1, 0.5, 0.5)):
        """
        gripper_history_window = (window_length, prediction_sum_threshold_close, prediction_sum_threshold_open): the last `window_length` frames of gripper prediction history will be used. 
        If the sum of the last `window_length` frames is greater than `prediction_sum_threshold_close`, then the gripper will be closed. 
        If the sum of the last `window_length` frames is greater than `prediction_sum_threshold_open`, then the gripper will be opened. 
        """
        gripper_is_being_blocked = False
        fs_wrapper = FrameStackWrapper(num_frames=10)
        fs_wrapper.reset()
        imsize = 128

        gripper_action_history = queue.Queue()
        # populate the queue with zeros
        for i in range(gripper_history_window[0]):
            gripper_action_history.put(0)


        def trigger_gripper():
            print("Enter into trigger gripper at time {}".format(time_ms() - self.trajectory_start_time))
            nonlocal gripper_is_being_blocked
            def update_gripper_status():
                time.sleep(1.3)
                print("Start")
                self._gripper_is_closed = not self._gripper_is_closed
                print("End")
                    
            gripper_is_closed = self._gripper_is_closed
            t = threading.Thread(target=update_gripper_status) # start a child thread
            t.daemon = True
            t.start()
            gripper_is_being_blocked = True
            print("Start blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))
            if gripper_is_closed:
                self._robot.gripper.open()
            else:
                self._robot.gripper.close()
            gripper_is_being_blocked = False
            print("End blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))


        def keep_recording():
            nonlocal state_traj, obs_traj, action_traj, standard_output, gripper_is_being_blocked, last_timestep, i
            gripper_is_being_blocked = True # otherwise it will not enter the loop before trigger_gripper set gripper_is_being_blocked to True
            
            while gripper_is_being_blocked:
                # print("Doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)

                # continue recording the state and observation at the same frequency
                state_dict, obs_dict = self.get_observation()
                state_traj["poses"].append(state_dict["robot_pose"].matrix)
                state_traj["joints"].append(state_dict["robot_joints"])
                state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
                state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
                state_traj["gripper_closed"].append(self._gripper_is_closed)
                state_traj["action_blocked"].append(True)
                obs_traj["hand_image"].append(obs_dict["hand_image"])
                obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
                obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
                obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])
                writer = AsyncWrite(None, obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
                writer.start()

                # for the standardized format:
                pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
                robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [True] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
                image = obs_dict["third_person_image"][0]
                # task = task_string
                standard_output["robot_state"].append(robot_state)
                standard_output["image"].append(image)
                # standard_output["task"].append([task])
                standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
                standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            

                to_tensor = transforms.ToTensor()
                hand_image = to_tensor(cv2.resize(obs_dict["hand_image"], (imsize, imsize)))
                side_image = to_tensor(cv2.resize(obs_dict["third_person_image"][0], (imsize, imsize)))

                robot_state = np.array(robot_state)
                if not use_hand_image:
                    obs = {
                        "robot_state/cartesian_position": robot_state[6:13],
                        "robot_state/joint_position": robot_state[:6],
                        "robot_state/gripper_obs": robot_state[[13]],
                        "camera/image/varied_camera_right_image": side_image
                    }
                else:
                    obs = {
                        "robot_state/cartesian_position": robot_state[6:13],
                        "robot_state/joint_position": robot_state[:6],
                        "robot_state/gripper_obs": robot_state[[13]],
                        "camera/image/hand_camera_image": hand_image,
                        "camera/image/varied_camera_right_image": side_image
                    }
                fs_wrapper.add_obs(obs)
                obs_history = fs_wrapper.get_obs_history()
                action = model(obs_history)

                # clip action
                action = np.clip(action, a_min=-1, a_max=1)
                position_action = action[:3] * 0.02
                rotation_action = action[3:6] / 15
                
                action = np.append(np.append(position_action, rotation_action), action[6:])
                gripper_action_history.get()
                gripper_action_history.put(action[-2])
                
                print(action, "Do nothing because the gripper is being blocked")
                action_traj.append(action)
                
                
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1
            print("Stop doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)
            return

        

        state_traj = {"poses": [], "joints": [], "gripper_closed": [], "action_blocked": [], "timestamp":{"read_start": [], "read_end":[]}} # "poses": [RigidTransforms], "joints": Array(T, 6); "timestamp"/ "read_start": Array(T,), "read_end": Array(T,)
        obs_traj = {"hand_image": [], "third_person_image": [], "timestamp":{"read_start": [], "read_end":[]}}
        action_traj = []
        standard_output = {"robot_state": [], "action": [], "image": [], "task": [], "other": {"hand_image": [], "third_person_image": []}}
        last_timestep = time.time()
        i = 0
        self.trajectory_start_time = time_ms()

        while True:
            state_dict, obs_dict = self.get_observation()
            state_traj["poses"].append(state_dict["robot_pose"].matrix)
            state_traj["joints"].append(state_dict["robot_joints"])
            state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
            state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
            state_traj["gripper_closed"].append(self._gripper_is_closed)
            state_traj["action_blocked"].append(False)
            obs_traj["hand_image"].append(obs_dict["hand_image"])
            obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
            obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
            obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])

            writer = AsyncWrite(None, obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
            writer.start()

            # for the standardized format:
            pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
            robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [False] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
            image = obs_dict["third_person_image"][0]
            # task = task_string
            standard_output["robot_state"].append(robot_state)
            standard_output["image"].append(image)
            # standard_output["task"].append([task])
            standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
            standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            
            
            to_tensor = transforms.ToTensor()
            hand_image = to_tensor(cv2.resize(obs_dict["hand_image"], (imsize, imsize)))
            side_image = to_tensor(cv2.resize(obs_dict["third_person_image"][0], (imsize, imsize)))
            robot_state = np.array(robot_state)
            if not use_hand_image:
                obs = {
                    "robot_state/cartesian_position": robot_state[6:13],
                    "robot_state/joint_position": robot_state[:6],
                    "robot_state/gripper_obs": robot_state[[13]],
                    "camera/image/varied_camera_right_image": side_image
                }
            else:
                obs = {
                    "robot_state/cartesian_position": robot_state[6:13],
                    "robot_state/joint_position": robot_state[:6],
                    "robot_state/gripper_obs": robot_state[[13]],
                    "camera/image/hand_camera_image": hand_image,
                    "camera/image/varied_camera_right_image": side_image
                }
            fs_wrapper.add_obs(obs)
            obs_history = fs_wrapper.get_obs_history()
            action = model(obs_history)

            # clip action
            action = np.clip(action, a_min=-1, a_max=1)
            position_action = action[:3] * 0.02
            rotation_action = action[3:6] / 15
            
            action = np.append(np.append(position_action, rotation_action), action[6:])
            gripper_action_history.get()
            gripper_action_history.put(action[-2])
            gripper_last_actions = gripper_action_history.queue # a deque object that is a copy of the queue
            
            action_traj.append(action)
            action = np.clip(action, [-0.02, -0.02, -0.02, -1/15, -1/15, -1/15, -1, -1], [0.02, 0.02, 0.02, 1/15, 1/15, 1/15, 1, 1])
            print(action)

            

            if action[-1] >= 1:
                stop = True
                # action_traj.append([0,0,0,0,0,0,0,1]) # termination action
                break
            elif (self._gripper_is_closed and not self._gripper_being_blocked and np.sum(gripper_last_actions) < gripper_history_window[2]) or (not self._gripper_is_closed and not self._gripper_being_blocked and np.sum(gripper_last_actions) > gripper_history_window[1]):
                last_timestep = time.time()
                # if self._gripper_is_closed and action[-2] < 0:
                    # action_traj.append([0,0,0,0,0,0,-1,0]) # open gripper
                # elif not self._gripper_is_closed and action[-2] > 0:
                    # action_traj.append([0,0,0,0,0,0,1,0]) # close gripper
                
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

                threadList = [threading.Thread(target=trigger_gripper), threading.Thread(target=keep_recording)]
                for threads in threadList:
                    threads.start()
                for threads in threadList:
                    threads.join()
                
            else:
                
                delta_pose = RigidTransform(translation=action[:3], rotation=euler_matrix(action[3], action[4], action[5], axes="ryxz")[:3, :3], from_frame="tcp", to_frame="tcp")

                # Update Robot
                current_pose = self._robot.get_pose()
                current_pose.from_frame = "tcp"
                new_pose = current_pose * delta_pose
                # self._robot.move_pose(new_pose, vel=1, acc=10)
                self._robot.servo_pose(new_pose, 0.01, 0.2, 100)
            
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

        # Turn everything into numpy arrays
        # standard_output["robot_state"] = np.array(standard_output["robot_state"]) # (T, 15)
        # standard_output["image"] = np.stack(standard_output["image"], axis=0) # (T, 480, 640, 3)
        # standard_output["action"] = np.array(action_traj) # (T, 8)
        # standard_output["task"] = np.array(standard_output["task"]) # (T, 1)
        # standard_output["other"]["hand_image"] = np.stack(standard_output["other"]["hand_image"], axis=0) # (T, 480, 640, 3)
        # standard_output["other"]["third_person_image"] = np.stack(standard_output["other"]["third_person_image"], axis=0) # (T, 480, 640, 4)
        
        return standard_output, action_traj, state_traj, obs_traj
    
    
    
    def evaluate_lcbc_model_trajectory(self, model, language, use_hand_image=False, traj_index=0, saving_directory="/home/lawrence/robotlerf/ur5bc/data/lcbc/", gripper_history_window = (1, 0.5, 0.5)):
        """
        gripper_history_window = (window_length, prediction_sum_threshold_close, prediction_sum_threshold_open): the last `window_length` frames of gripper prediction history will be used. 
        If the sum of the last `window_length` frames is greater than `prediction_sum_threshold_close`, then the gripper will be closed. 
        If the sum of the last `window_length` frames is greater than `prediction_sum_threshold_open`, then the gripper will be opened. 
        """
        assert use_hand_image == False, "use_hand_image is not supported yet"
        gripper_is_being_blocked = False
        imsize = 128

        gripper_action_history = queue.Queue()
        # populate the queue with zeros
        for i in range(gripper_history_window[0]):
            gripper_action_history.put(0)


        def trigger_gripper():
            print("Enter into trigger gripper at time {}".format(time_ms() - self.trajectory_start_time))
            nonlocal gripper_is_being_blocked
            def update_gripper_status():
                time.sleep(1.3)
                print("Start")
                self._gripper_is_closed = not self._gripper_is_closed
                print("End")
                    
            gripper_is_closed = self._gripper_is_closed
            t = threading.Thread(target=update_gripper_status) # start a child thread
            t.daemon = True
            t.start()
            gripper_is_being_blocked = True
            print("Start blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))
            if gripper_is_closed:
                self._robot.gripper.open()
            else:
                self._robot.gripper.close()
            gripper_is_being_blocked = False
            print("End blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))


        def keep_recording():
            nonlocal state_traj, obs_traj, action_traj, standard_output, gripper_is_being_blocked, last_timestep, i
            gripper_is_being_blocked = True # otherwise it will not enter the loop before trigger_gripper set gripper_is_being_blocked to True
            
            while gripper_is_being_blocked:
                # print("Doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)

                # continue recording the state and observation at the same frequency
                state_dict, obs_dict = self.get_observation()
                state_traj["poses"].append(state_dict["robot_pose"].matrix)
                state_traj["joints"].append(state_dict["robot_joints"])
                state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
                state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
                state_traj["gripper_closed"].append(self._gripper_is_closed)
                state_traj["action_blocked"].append(True)
                obs_traj["hand_image"].append(obs_dict["hand_image"])
                obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
                obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
                obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])
                writer = AsyncWrite(None, obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
                writer.start()

                # for the standardized format:
                pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
                robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [True] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
                image = obs_dict["third_person_image"][0]
                # task = task_string
                standard_output["robot_state"].append(robot_state)
                standard_output["image"].append(image)
                # standard_output["task"].append([task])
                standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
                standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            
                action = model.predict_action(obs_dict["third_person_image"][0], language)
                gripper_action_history.get()
                gripper_action_history.put(action[-2])
                
                print(action, "Do nothing because the gripper is being blocked")
                action_traj.append(action)
                
                
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1
            print("Stop doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)
            return

        

        state_traj = {"poses": [], "joints": [], "gripper_closed": [], "action_blocked": [], "timestamp":{"read_start": [], "read_end":[]}} # "poses": [RigidTransforms], "joints": Array(T, 6); "timestamp"/ "read_start": Array(T,), "read_end": Array(T,)
        obs_traj = {"hand_image": [], "third_person_image": [], "timestamp":{"read_start": [], "read_end":[]}}
        action_traj = []
        standard_output = {"robot_state": [], "action": [], "image": [], "task": [], "other": {"hand_image": [], "third_person_image": []}}
        last_timestep = time.time()
        i = 0
        self.trajectory_start_time = time_ms()

        while True:
            state_dict, obs_dict = self.get_observation()
            state_traj["poses"].append(state_dict["robot_pose"].matrix)
            state_traj["joints"].append(state_dict["robot_joints"])
            state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
            state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
            state_traj["gripper_closed"].append(self._gripper_is_closed)
            state_traj["action_blocked"].append(False)
            obs_traj["hand_image"].append(obs_dict["hand_image"])
            obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
            obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
            obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])

            writer = AsyncWrite(None, obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
            writer.start()

            # for the standardized format:
            pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
            robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [False] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
            image = obs_dict["third_person_image"][0]
            # task = task_string
            standard_output["robot_state"].append(robot_state)
            standard_output["image"].append(image)
            # standard_output["task"].append([task])
            standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
            standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            
            
            if model.initial_obs is None:
                model.set_initial_obs(obs_dict["third_person_image"][0])
            action = model.predict_action(obs_dict["third_person_image"][0], language)
            
            gripper_action_history.get()
            gripper_action_history.put(action[-2])
            gripper_last_actions = gripper_action_history.queue # a deque object that is a copy of the queue
            
            action_traj.append(action)
            action = np.clip(action, [-0.02, -0.02, -0.02, -1/15, -1/15, -1/15, -1, -1], [0.02, 0.02, 0.02, 1/15, 1/15, 1/15, 1, 1])
            print(action)

            

            if action[-1] >= 1:
                stop = True
                # action_traj.append([0,0,0,0,0,0,0,1]) # termination action
                break
            elif (self._gripper_is_closed and not self._gripper_being_blocked and np.sum(gripper_last_actions) < gripper_history_window[2]) or (not self._gripper_is_closed and not self._gripper_being_blocked and np.sum(gripper_last_actions) > gripper_history_window[1]):
                last_timestep = time.time()
                # if self._gripper_is_closed and action[-2] < 0:
                    # action_traj.append([0,0,0,0,0,0,-1,0]) # open gripper
                # elif not self._gripper_is_closed and action[-2] > 0:
                    # action_traj.append([0,0,0,0,0,0,1,0]) # close gripper
                
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

                threadList = [threading.Thread(target=trigger_gripper), threading.Thread(target=keep_recording)]
                for threads in threadList:
                    threads.start()
                for threads in threadList:
                    threads.join()
                
            else:
                
                delta_pose = RigidTransform(translation=action[:3], rotation=euler_matrix(action[3], action[4], action[5], axes="ryxz")[:3, :3], from_frame="tcp", to_frame="tcp")

                # Update Robot
                current_pose = self._robot.get_pose()
                current_pose.from_frame = "tcp"
                new_pose = current_pose * delta_pose
                # self._robot.move_pose(new_pose, vel=1, acc=10)
                self._robot.servo_pose(new_pose, 0.01, 0.2, 100)
            
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

        # Turn everything into numpy arrays
        # standard_output["robot_state"] = np.array(standard_output["robot_state"]) # (T, 15)
        # standard_output["image"] = np.stack(standard_output["image"], axis=0) # (T, 480, 640, 3)
        # standard_output["action"] = np.array(action_traj) # (T, 8)
        # standard_output["task"] = np.array(standard_output["task"]) # (T, 1)
        # standard_output["other"]["hand_image"] = np.stack(standard_output["other"]["hand_image"], axis=0) # (T, 480, 640, 3)
        # standard_output["other"]["third_person_image"] = np.stack(standard_output["other"]["third_person_image"], axis=0) # (T, 480, 640, 4)
        
        return standard_output, action_traj, state_traj, obs_traj
    
    
    
    
    def evaluate_rt1_model_trajectory(self, model, task_embedding, traj_index=0, saving_directory="/home/lawrence/robotlerf/ur5bc/data/raw/teleop/", scale_model_output=False):
        gripper_is_being_blocked = False

        def trigger_gripper():
            print("Enter into trigger gripper at time {}".format(time_ms() - self.trajectory_start_time))
            nonlocal gripper_is_being_blocked
            def update_gripper_status():
                time.sleep(1.3)
                print("Start")
                self._gripper_is_closed = not self._gripper_is_closed
                print("End")
                    
            gripper_is_closed = self._gripper_is_closed
            t = threading.Thread(target=update_gripper_status) # start a child thread
            t.daemon = True
            t.start()
            gripper_is_being_blocked = True
            print("Start blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))
            if gripper_is_closed:
                self._robot.gripper.open()
            else:
                self._robot.gripper.close()
            gripper_is_being_blocked = False
            print("End blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))


        def keep_recording():
            nonlocal state_traj, obs_traj, action_traj, standard_output, gripper_is_being_blocked, last_timestep, i
            gripper_is_being_blocked = True # otherwise it will not enter the loop before trigger_gripper set gripper_is_being_blocked to True
            
            while gripper_is_being_blocked:
                # print("Doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)

                # continue recording the state and observation at the same frequency
                state_dict, obs_dict = self.get_observation()
                state_traj["poses"].append(state_dict["robot_pose"].matrix)
                state_traj["joints"].append(state_dict["robot_joints"])
                state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
                state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
                state_traj["gripper_closed"].append(self._gripper_is_closed)
                state_traj["action_blocked"].append(True)
                obs_traj["hand_image"].append(obs_dict["hand_image"])
                obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
                obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
                obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])
                writer = AsyncWrite(None, obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
                writer.start()

                action = model.predict_action(obs_dict["third_person_image"][0], task_embedding, scale_model_output)
                print(action, "Do nothing because the gripper is being blocked")
                action_traj.append(action)
                
                # for the standardized format:
                pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
                robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [True] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
                image = obs_dict["third_person_image"][0]
                # task = task_string
                standard_output["robot_state"].append(robot_state)
                standard_output["image"].append(image)
                # standard_output["task"].append([task])
                standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
                standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1
            print("Stop doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)
            return

        

        state_traj = {"poses": [], "joints": [], "gripper_closed": [], "action_blocked": [], "timestamp":{"read_start": [], "read_end":[]}} # "poses": [RigidTransforms], "joints": Array(T, 6); "timestamp"/ "read_start": Array(T,), "read_end": Array(T,)
        obs_traj = {"hand_image": [], "third_person_image": [], "timestamp":{"read_start": [], "read_end":[]}}
        action_traj = []
        standard_output = {"robot_state": [], "action": [], "image": [], "task": [], "other": {"hand_image": [], "third_person_image": []}}
        last_timestep = time.time()
        i = 0
        self.trajectory_start_time = time_ms()

        while True:
            state_dict, obs_dict = self.get_observation()
            state_traj["poses"].append(state_dict["robot_pose"].matrix)
            state_traj["joints"].append(state_dict["robot_joints"])
            state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
            state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
            state_traj["gripper_closed"].append(self._gripper_is_closed)
            state_traj["action_blocked"].append(False)
            obs_traj["hand_image"].append(obs_dict["hand_image"])
            obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
            obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
            obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])

            writer = AsyncWrite(None, obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
            writer.start()

            # for the standardized format:
            pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
            robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [False] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
            image = obs_dict["third_person_image"][0]
            # task = task_string
            standard_output["robot_state"].append(robot_state)
            standard_output["image"].append(image)
            # standard_output["task"].append([task])
            standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
            standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            
            action = model.predict_action(obs_dict["third_person_image"][0], task_embedding, scale_model_output)
            action_traj.append(action)
            action = np.clip(action, [-0.02, -0.02, -0.02, -1/15, -1/15, -1/15, -1, -1], [0.02, 0.02, 0.02, 1/15, 1/15, 1/15, 1, 1])
            print(action)

            

            if action[-1] == 1:
                stop = True
                # action_traj.append([0,0,0,0,0,0,0,1]) # termination action
                break
            elif (self._gripper_is_closed and action[-2] < -0.5) or (not self._gripper_is_closed and action[-2] > 0.5):
                last_timestep = time.time()
                # if self._gripper_is_closed and action[-2] < 0:
                    # action_traj.append([0,0,0,0,0,0,-1,0]) # open gripper
                # elif not self._gripper_is_closed and action[-2] > 0:
                    # action_traj.append([0,0,0,0,0,0,1,0]) # close gripper
                
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

                threadList = [threading.Thread(target=trigger_gripper), threading.Thread(target=keep_recording)]
                for threads in threadList:
                    threads.start()
                for threads in threadList:
                    threads.join()
                
            else:
                
                
                delta_pose = RigidTransform(translation=action[:3], rotation=euler_matrix(action[3], action[4], action[5], axes="ryxz")[:3, :3], from_frame="tcp", to_frame="tcp")

                # Update Robot
                current_pose = self._robot.get_pose()
                current_pose.from_frame = "tcp"
                new_pose = current_pose * delta_pose
                # self._robot.move_pose(new_pose, vel=1, acc=10)
                self._robot.servo_pose(new_pose, 0.01, 0.2, 100)
            
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

        # Turn everything into numpy arrays
        # standard_output["robot_state"] = np.array(standard_output["robot_state"]) # (T, 15)
        # standard_output["image"] = np.stack(standard_output["image"], axis=0) # (T, 480, 640, 3)
        # standard_output["action"] = np.array(action_traj) # (T, 8)
        # standard_output["task"] = np.array(standard_output["task"]) # (T, 1)
        # standard_output["other"]["hand_image"] = np.stack(standard_output["other"]["hand_image"], axis=0) # (T, 480, 640, 3)
        # standard_output["other"]["third_person_image"] = np.stack(standard_output["other"]["third_person_image"], axis=0) # (T, 480, 640, 4)
        
        return standard_output, action_traj, state_traj, obs_traj

    def evaluate_orca_model_trajectory(self, task_string, traj_index=0, saving_directory="/home/lawrence/robotlerf/ur5bc/data/raw/teleop/"):
        gripper_is_being_blocked = False

        import tensorflow as tf
        import urllib
        import requests
        import json_numpy
        json_numpy.patch()
        from json import loads
        def resize_image(image, size=128):
            image = tf.image.resize(image, size=[size, size], method="lanczos3", antialias=True)
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
            return image

        num_obs = 0
        goal_image = None
        use_goal_image = input("Use goal image? ") == "y"

        def trigger_gripper():
            print("Enter into trigger gripper at time {}".format(time_ms() - self.trajectory_start_time))
            nonlocal gripper_is_being_blocked, num_obs
            def update_gripper_status():
                time.sleep(1.3)
                print("Start")
                self._gripper_is_closed = not self._gripper_is_closed
                print("End")
                    
            gripper_is_closed = self._gripper_is_closed
            t = threading.Thread(target=update_gripper_status) # start a child thread
            t.daemon = True
            t.start()
            gripper_is_being_blocked = True
            print("Start blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))
            if gripper_is_closed:
                self._robot.gripper.open()
            else:
                self._robot.gripper.close()
            gripper_is_being_blocked = False
            print("End blocking gripper at time {}".format(time_ms() - self.trajectory_start_time))


        def keep_recording():
            nonlocal state_traj, obs_traj, action_traj, standard_output, gripper_is_being_blocked, last_timestep, i, num_obs, goal_image, task
            gripper_is_being_blocked = True # otherwise it will not enter the loop before trigger_gripper set gripper_is_being_blocked to True
            
            while gripper_is_being_blocked:
                # print("Doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)

                # continue recording the state and observation at the same frequency
                state_dict, obs_dict = self.get_observation()
                state_traj["poses"].append(state_dict["robot_pose"].matrix)
                state_traj["joints"].append(state_dict["robot_joints"])
                state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
                state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
                state_traj["gripper_closed"].append(self._gripper_is_closed)
                state_traj["action_blocked"].append(True)
                obs_traj["hand_image"].append(obs_dict["hand_image"])
                obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
                obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
                obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])
                writer = AsyncWrite(obs_dict["hand_image"], obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
                writer.start()
                
                # for the standardized format:
                pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
                robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [True] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
                image = obs_dict["third_person_image"][0]
                # task = task_string
                standard_output["robot_state"].append(robot_state)
                standard_output["image"].append(image)
                # standard_output["task"].append([task])
                standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
                standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1
            print("Stop doing things at time {} ".format(time_ms() - self.trajectory_start_time), "for iteration ", i)
            return

        

        state_traj = {"poses": [], "joints": [], "gripper_closed": [], "action_blocked": [], "timestamp":{"read_start": [], "read_end":[]}} # "poses": [RigidTransforms], "joints": Array(T, 6); "timestamp"/ "read_start": Array(T,), "read_end": Array(T,)
        obs_traj = {"hand_image": [], "third_person_image": [], "timestamp":{"read_start": [], "read_end":[]}}
        action_traj = []
        standard_output = {"robot_state": [], "action": [], "image": [], "task": [], "other": {"hand_image": [], "third_person_image": []}}
        last_timestep = time.time()
        i = 0
        self.trajectory_start_time = time_ms()

        

        while True:
            state_dict, obs_dict = self.get_observation()
            state_traj["poses"].append(state_dict["robot_pose"].matrix)
            state_traj["joints"].append(state_dict["robot_joints"])
            state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
            state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
            state_traj["gripper_closed"].append(self._gripper_is_closed)
            state_traj["action_blocked"].append(False)
            obs_traj["hand_image"].append(obs_dict["hand_image"])
            obs_traj["third_person_image"].append(obs_dict["third_person_image"]) 
            obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
            obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])

            writer = AsyncWrite(obs_dict["hand_image"], obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
            writer.start()

            # for the standardized format:
            pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
            robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [False] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
            image = obs_dict["third_person_image"][0]
            # task = task_string
            standard_output["robot_state"].append(robot_state)
            standard_output["image"].append(image)
            # standard_output["task"].append([task])
            standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
            standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))
            
            obs = {
                "image_0": resize_image(obs_dict["third_person_image"][0], size=256),
                # "image_1": resize_image(obs_dict["hand_image"], size=128)[:,:,::-1],
            }
            if goal_image is None and use_goal_image:
                goal_image = obs
                task = goal_image
                input("Press [Enter] to continue")
            else:
                task = task_string

            num_obs += 1

            action = np.copy(np.array(loads(requests.post(
                            urllib.parse.urljoin('http://128.32.175.252:8000', 'query'),
                            json={"observation": obs, "task": task},
                        ).json())))

            print(action[-1])
            action[-1] = 1 if (action[-1] < 0.25) else -1
            action = np.append(action, 0)

            action_traj.append(action)
            action = np.clip(action, [-0.02, -0.02, -0.02, -1/15, -1/15, -1/15, -1, -1], [0.02, 0.02, 0.02, 1/15, 1/15, 1/15, 1, 1])

            if action[-1] == 1:
                stop = True
                # action_traj.append([0,0,0,0,0,0,0,1]) # termination action
                break
            elif (self._gripper_is_closed and action[-2] < -0.5) or (not self._gripper_is_closed and action[-2] > 0.5):
                last_timestep = time.time()
                # if self._gripper_is_closed and action[-2] < 0:
                    # action_traj.append([0,0,0,0,0,0,-1,0]) # open gripper
                # elif not self._gripper_is_closed and action[-2] > 0:
                    # action_traj.append([0,0,0,0,0,0,1,0]) # close gripper
                
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

                threadList = [threading.Thread(target=trigger_gripper), threading.Thread(target=keep_recording)]
                for threads in threadList:
                    threads.start()
                for threads in threadList:
                    threads.join()
                
            else:
                
                
                delta_pose = RigidTransform(translation=action[:3], rotation=euler_matrix(action[3], action[4], action[5], axes="ryxz")[:3, :3], from_frame="tcp", to_frame="tcp")

                # Update Robot
                current_pose = self._robot.get_pose()
                current_pose.from_frame = "tcp"
                new_pose = current_pose * delta_pose
                # self._robot.move_pose(new_pose, vel=1, acc=10)
                self._robot.servo_pose(new_pose, 0.01, 0.2, 100)
            
                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
                i += 1

        # Turn everything into numpy arrays
        # standard_output["robot_state"] = np.array(standard_output["robot_state"]) # (T, 15)
        # standard_output["image"] = np.stack(standard_output["image"], axis=0) # (T, 480, 640, 3)
        # standard_output["action"] = np.array(action_traj) # (T, 8)
        # standard_output["task"] = np.array(standard_output["task"]) # (T, 1)
        # standard_output["other"]["hand_image"] = np.stack(standard_output["other"]["hand_image"], axis=0) # (T, 480, 640, 3)
        # standard_output["other"]["third_person_image"] = np.stack(standard_output["other"]["third_person_image"], axis=0) # (T, 480, 640, 4)
        
        return standard_output, action_traj, state_traj, obs_traj

    def play_teleop_trajectory(self, action_traj, action_blocked, starting_state=None):
        """
        Execute the teleop trajectory from the starting state
        action_traj: np.array of shape (T, 8)
        action_blocked: np.array of shape (T,)
        """
        def trigger_gripper(command):
            print("triggering gripper")
            if command == "open":
                self._robot.gripper.open()
            elif command == "close":
                self._robot.gripper.close()
            print("gripper triggered")

        input("enter to play trajectory")
        if starting_state is not None:
            self._robot.move_joint(starting_state)
        
        last_timestep = time.time()
        for i in range(len(action_traj)):
            action = action_traj[i]
            if action_blocked[i]:
                time.sleep(1 / self.control_hz)
                last_timestep = time.time()
                continue
            if action[-1] == 1:
                break
            elif action[-2] == 0:
                delta_pose = RigidTransform(translation=np.array([action[0], action[1], action[2]]), rotation=euler_matrix(action[3], action[4], action[5], axes="ryxz")[:3, :3], from_frame="tcp", to_frame="tcp")

                # Update Robot
                current_pose = self._robot.get_pose()
                current_pose.from_frame = "tcp"
                new_pose =  current_pose * delta_pose
                # self._robot.move_pose(new_pose, vel=1, acc=10)
                # self._robot.servo_pose(new_pose, 0.002, 0.1, 100) # 10 Hz
                self._robot.servo_pose(new_pose, 0.01, 0.2, 100) # 5 Hz

                if time.time() - last_timestep < 1 / self.control_hz:
                    time.sleep(last_timestep + 1 / self.control_hz - time.time())
                else:
                    print("Warning: Control Loop is running slower than desired frequency")
                    print(time.time() - last_timestep, " seconds has passed")
                last_timestep = time.time()
            else:
                
                if action[-2] == 1:
                    t = threading.Thread(target=trigger_gripper, args=("close",))
                else:
                    t = threading.Thread(target=trigger_gripper, args=("open",))
                
                t.daemon = True
                t.start()
                time.sleep(1 / self.control_hz)
                last_timestep = time.time()

    def record_free_drive_trajectory(self, task_string, traj_index=0, saving_directory="/home/lawrence/robotlerf/ur5bc/data/raw/freedrive/"):
        self._robot.start_freedrive()
        input("enter to start recording traj")

        stop = False

        def on_press(key):
            if key == keyboard.Key.esc:
                return False  # stop listener
            try:
                k = key.char  # single-char keys
            except:
                k = key.name  # other keys
            if k in ['1', '2', '0']:  # keys of interest
                # self.keys.append(k)  # store it in global-like variable
                print('Key pressed: ' + k)
                trigger_gripper()
            elif k in ['space']:
                execute_termination()
                return False  # stop listener; remove this if want more keys

        def trigger_gripper():
            def update_gripper_status():
                time.sleep(1.3)
                print("Start")
                self._gripper_is_closed = not self._gripper_is_closed
                print("End")
                    
            gripper_is_closed = self._gripper_is_closed
            t = threading.Thread(target=update_gripper_status) # start a child thread
            t.daemon = True
            t.start()
            if gripper_is_closed:
                self._robot.gripper.open()
            else:
                self._robot.gripper.close()
            self._robot.start_freedrive()

        def execute_termination():
            nonlocal stop
            stop = True

        # keyboard.add_hotkey("space", execute_termination) # this requires root
        # keyboard.add_hotkey("0", trigger_gripper)

        listener = keyboard.Listener(on_press=on_press)
        listener.start()  # start to listen on a separate thread
        # listener.join()  # remove if main thread is polling self.keys

        state_traj = {"poses": [], "joints": [], "gripper_closed": [], "timestamp":{"read_start": [], "read_end":[]}} # "poses": [RigidTransforms], "joints": Array(T, 6); "timestamp"/ "read_start": Array(T,), "read_end": Array(T,)
        obs_traj = {"hand_image": [], "third_person_image": [], "timestamp":{"read_start": [], "read_end":[]}}
        standard_output = {"robot_state": [], "action": [], "image": [], "task": [], "other": {"hand_image": [], "third_person_image": []}}
        last_timestep = time.time()
        i = 0
        self.trajectory_start_time = time_ms()
        while True:
            if stop:
                break

            state_dict, obs_dict = self.get_observation()
            state_traj["poses"].append(state_dict["robot_pose"])
            state_traj["joints"].append(state_dict["robot_joints"])
            state_traj["timestamp"]["read_start"].append(state_dict["timestamp"]["read_start"])
            state_traj["timestamp"]["read_end"].append(state_dict["timestamp"]["read_end"])
            state_traj["gripper_closed"].append(self._gripper_is_closed)
            obs_traj["hand_image"].append(obs_dict["hand_image"])
            obs_traj["third_person_image"].append(obs_dict["third_person_image"]) # this line causes the depth camera to freeze
            obs_traj["timestamp"]["read_start"].append(obs_dict["timestamp"]["read_start"])
            obs_traj["timestamp"]["read_end"].append(obs_dict["timestamp"]["read_end"])

            writer = AsyncWrite(obs_dict["hand_image"], obs_dict["third_person_image"][0], obs_dict["third_person_image"][1], traj_index, saving_directory, i)
            writer.start()
            
            # for the standardized format:
            pose = list(translation_from_matrix(state_dict["robot_pose"].matrix)) + list(quaternion_from_matrix(state_dict["robot_pose"].matrix)) # [x,y,z, qx,qy,qz,qw]
            robot_state = state_dict["robot_joints"] + pose + [self._gripper_is_closed] + [False] # [joint_angles, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
            image = obs_dict["third_person_image"][0]
            task = task_string
            standard_output["robot_state"].append(robot_state)
            standard_output["image"].append(image)
            standard_output["task"].append([task])
            standard_output["other"]["hand_image"].append(obs_dict["hand_image"])
            standard_output["other"]["third_person_image"].append(np.dstack((obs_dict["third_person_image"][0], obs_dict["third_person_image"][1])))

            if time.time() - last_timestep < 1 / self.control_hz:
                time.sleep(last_timestep + 1 / self.control_hz - time.time())
            else:
                print("Warning: Control Loop is running slower than desired frequency")
                print(time.time() - last_timestep, " seconds has passed")
            last_timestep = time.time()
            i += 1
        
        action_traj = []
        for i in range(len(state_traj["poses"])-1):
            action_traj.append(self.infer_action_from_observation(state_traj["poses"][i], state_traj["poses"][i+1], state_traj["gripper_closed"][i], state_traj["gripper_closed"][i+1]))
        action_traj.append([0,0,0,0,0,0,0,1]) # termination action
        
        # Turn everything into numpy arrays
        standard_output["robot_state"] = np.array(standard_output["robot_state"]) # (T, 18)
        standard_output["image"] = np.stack(standard_output["image"], axis=0) # (T, 480, 640, 3)
        standard_output["action"] = np.array(action_traj) # (T, 8)
        standard_output["task"] = np.array(standard_output["task"]) # (T, 1)
        standard_output["other"]["hand_image"] = np.stack(standard_output["other"]["hand_image"], axis=0) # (T, 480, 640, 3)
        standard_output["other"]["third_person_image"] = np.stack(standard_output["other"]["third_person_image"], axis=0) # (T, 480, 640, 4)
            
        return standard_output, action_traj, state_traj, obs_traj

if __name__=="__main__":
    env = RobotEnv()
    while True:
        # env.read_cameras()
        # env.get_observation()
        env.record_free_drive_trajectory(10, "/home/lawrence/robotlerf/ur5bc/data/raw/freedrive/")
        # env.record_teleop_trajectory()