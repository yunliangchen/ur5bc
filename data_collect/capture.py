from autolab_core import RigidTransform, Box, Point
import numpy as np
import dexnerf.capture.capture_utils as cu
from dexnerf.cameras.zed import ZedImageCapture
from ur5py.ur5 import UR5Robot
import multiprocessing as mp
import time
from queue import Empty

class MPWrapper(mp.Process):
    def __init__(self,cls,*args,**kwargs):
        '''
        wraps the given class type specified by cls with the args provided
        '''
        super().__init__()
        self.cls=cls
        #filter out defaults
        self.fn_names = list(filter(lambda x:'__' not in x,dir(self.cls)))
        #define a new attr of this class matching the names, but calling _run_fn
        self.cmd_q = mp.Manager().Queue()#holds commands to execute
        self.resp_q = mp.Manager().Queue()#holds results of commands
        for fn_name in self.fn_names:
            self._add_function(fn_name)
        self.args=args
        self.kwargs=kwargs
        self.daemon=True
        self.start()

    def _add_function(self, fn_name):
        def wrapper_fn(*fn_args,**fn_kwargs):
            #first, enqueue the command
            cmd_spec = (fn_name,fn_args,fn_kwargs)
            self.cmd_q.put(cmd_spec)
            #then, wait for response
            resp = self.resp_q.get(block=True)
            #then return response
            return resp
        setattr(self,fn_name,wrapper_fn)

    def run(self):
        '''
        loops and runs commands and enqueues responses
        '''
        #instantiate the object
        self.obj = self.cls(*self.args,**self.kwargs)
        while True:
            #check if the queue has something
            fn_name,fn_args,fn_kwargs = self.cmd_q.get(block=True)
            #execute the command
            resp = getattr(self.obj,fn_name)(*fn_args,**fn_kwargs)
            self.resp_q.put(resp)

class AsyncCapture(mp.Process):
    def __init__(self, use_stereo:bool, *zed_args, **zed_kwargs):
        '''
        rob can be any object that supports .get_pose() and .get_joints() (ie either UR5RObot ur YuMiArm)
        '''
        super().__init__()
        self.zed_args = zed_args
        self.zed_kwargs = zed_kwargs
        self.img_q = mp.Manager().Queue()#queue which ONLY holds img,pose pairs
        self.cmd_q = mp.Manager().Queue()#queue which holds generic input cmd data (such as last traj joint angles)
        self.res_q = mp.Manager().Queue()#queue which returns generic output data
        self.rob_q = mp.Manager().Queue()#queue for handling commands to robot
        self.trigger_capture = mp.Value('i',0)
        self.cap_threshold=.03#distance between images to save them
        self.use_stereo = use_stereo
        self.daemon = True
        self.rob = MPWrapper(UR5Robot)
        self.start()
        self.zed_intr = self.res_q.get(True)
        self.zed_translation = self.res_q.get(True)

    def run(self):
        self.zed=ZedImageCapture(*self.zed_args,**self.zed_kwargs)
        imgl, imgr = self.zed.capture_image()
        self.res_q.put(self.zed.intrinsics)
        self.res_q.put(self.zed.stereo_translation)
        left_to_right = RigidTransform(translation=self.zed.stereo_translation)
        H = RigidTransform.load('cfg/T_zed_wrist.tf')
        self.rob.set_tcp(H)
        self.rob.start_freedrive()
        while True:
            if self.trigger_capture.value>=1:
                lastpose=None
                last = self.cmd_q.get()
                while True:
                    time.sleep(.01)
                    pose = self.rob.get_pose()
                    # if the norm of the difference of last pose and current pose exceeds cap_threshold, take an image
                    if (lastpose is None or np.linalg.norm(pose.translation - lastpose.translation) > self.cap_threshold):
                        imgl, imgr = self.zed.capture_image()
                        lastpose=pose
                        self.img_q.put((imgl,pose))
                        if self.use_stereo:
                            right_pose = pose*left_to_right.as_frames(pose.from_frame,pose.from_frame)
                            self.img_q.put((imgr,right_pose))
                    #if we reached the last joint angle, end the capture
                    if last is not None and np.linalg.norm(self.rob.get_joints() - last) < 0.2:
                        with self.trigger_capture.get_lock():
                            self.trigger_capture.value = 0
                    if self.trigger_capture.value==0:
                        break
    
    @property
    def intrinsics(self):
        return self.zed_intr
    
    @property 
    def stereo_translation(self):
        return self.zed_translation

    def done(self):
        '''
        returns true if the capture process has finished
        '''
        return self.trigger_capture.value==0
    
    def end_cap(self):
        with self.trigger_capture.get_lock():
            self.trigger_capture.value = 0

    def trigger_cap(self,last_joints):
        '''
        Triggers the capture which will start populating the queue with images and poses
        If last_joints is specified, it will end when the robot reaches that pose, otherwise it will only stop when
        triggered externally
        '''
        self.cmd_q.put(last_joints)
        with self.trigger_capture.get_lock():
            self.trigger_capture.value = 1
    
    def get_imgs(self):
        """
        returns the top images on the queue and removes them
        """
        imgs,poses=[],[]
        while True:
            try:
                im,pose = self.img_q.get(True,timeout=.04)
                imgs.append(im)
                poses.append(pose)
            except Empty:
                break
        return imgs,poses
    
class URCapture:
    def __init__(self):
        self.hardware_proc = AsyncCapture(use_stereo=True,resolution="720p", exposure=60, 
                    gain=70, whitebalance_temp=3000, fps=100)
    
    def start_cap(self):
        self.hardware_proc.trigger_cap(None)
    
    def stop_cap(self):
        self.hardware_proc.end_cap()

    def freedrive_capture(self):
        self.start_cap()
    
    def get_imgs(self):
        return self.hardware_proc.get_imgs()
    
