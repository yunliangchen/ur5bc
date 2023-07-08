from tracikpy import TracIKSolver
from autolab_core import RigidTransform
import numpy as np


class UR5Kinematics():
    #these are the names of frames in RigidTransforms
    base_frame="base_link"
    #tip frame is the end of the urdf file, in this case meaning the wrist
    tip_frame="ee_link"
    #tcp is tool center point, meaning the point ik and fk will compute to
    tcp_frame="tcp"
    
    
    def __init__(self, urdf_filename=None, urdf_path=None):
        '''
        Initializes the kinematics with the urdf file
        '''
        import os
        if urdf_filename is None:
            urdf_filename = "/ur5.urdf"
        if urdf_path is None:
            urdf_path = os.path.dirname(os.path.abspath(__file__)) + urdf_filename
        
        #setup the tool center point as 0 transform
        self.set_tcp(None)
        '''taken from tracik:
        Speed: returns very quickly the first solution found
        % Distance: runs for the full timeout_in_secs, then returns the solution that minimizes SSE from the seed
        % Manipulation1: runs for full timeout, returns solution that maximizes sqrt(det(J*J^T)) (the product of the singular values of the Jacobian)
        % Manipulation2: runs for full timeout, returns solution that minimizes the ratio of min to max singular values of the Jacobian.
        '''
        self.solvers={}
        self.solvers["Distance"]=TracIKSolver(urdf_path,self.base_frame,self.tip_frame,timeout=.05,solve_type="Distance")
        self.solvers["Manipulation1"] = TracIKSolver(urdf_path,self.base_frame,self.tip_frame,timeout=.05,solve_type="Manipulation1")
        self.solvers["Speed"] = TracIKSolver(urdf_path,self.base_frame,self.tip_frame,timeout=.05,solve_type="Speed")

    def set_tcp(self,tool=None):
        if tool is None:
            self.tcp = RigidTransform(from_frame=self.tcp_frame, to_frame=self.tip_frame)
        else:
            assert tool.from_frame == self.tcp_frame,tool.to_frame == self.tip_frame
            self.tcp = tool

    def fk(self, q=None):
        '''
        computes the forward kinematics 
        q is an np array
        returns a RigidTransform
        '''
        res=None
        if q is not None:
            lpos = self.solvers["Speed"].fk(q)
            res = RigidTransform(translation=lpos[:3,3],rotation=lpos[:3,:3],from_frame=self.tip_frame,to_frame=self.base_frame)*self.tcp
        return res
        
    def ik(self, pose=None, qinit=None, solve_type="Speed",
            bs=[1e-5,1e-5,1e-5,  1e-3,1e-3,1e-3]):
        '''
        given  and/or right target poses, calculates the joint angles and returns them as a tuple
        poses are RigidTransforms, qinits are np arrays
        solve_type can be "Distance" or "Manipulation1" or "Speed" (See constructor for what these mean)
        bs is an array representing the tolerance on end pose of the gripper
        '''
        #NOTE: assumes given poses are in the tcp frames
        res=None
        if pose is not None:
            pose = pose*self.tcp.inverse()
            res=self.solvers[solve_type].ik(pose.matrix,qinit,
                    bx=bs[0],by=bs[1],bz=bs[2],  brx=bs[3],bry=bs[4],brz=bs[5])
        return res
