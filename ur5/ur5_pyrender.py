import trimesh
import numpy as np
import sys, getopt, itertools
from math import pi
from autolab_core.rigid_transformations import RigidTransform
import pyrender
from pyrender import MetallicRoughnessMaterial
import time
from utils import *
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

# Adds a constraint-based child.  In some cases it may be better to
# use `c.parent = p`.  The difference (other than syntax) seems to be
# that `.parent` changes the hierarchical view of objects in blender,
# whereas contraints are only found through navigating the contraint
# panes of blender.
def add_child(parent, child):
    child.assign_parent(parent)

# Loads a Collada (.dae) mesh and returns an object for that mesh.
# The mesh may optionally be uniformly scaled.
def load_mesh(name, filepath, scale=1):
    # Mesh files can contain multiple objects.  To simplify things, we
    # create an object without geometry (aka an "empty") to be the
    # parent of everything contained in the mesh file.
    p = Link(name, None)

    f = open(filepath, 'rb')
    kwargs = trimesh.exchange.dae.load_collada(f)
    for pose_dict in kwargs['graph']:
        name = pose_dict['frame_to']
        geom_name = pose_dict['geometry']
        pose = pose_dict['matrix']
        params = kwargs['geometry'][geom_name]
        mesh = trimesh.Trimesh(**params)
        c = Link(name, mesh, pose)
        c.assign_parent(p)

    if scale != 1:
        p.scale_uniform(scale)

    return p

class Link:
    def __init__(self, name, mesh, transform=np.eye(4)):
        self.name = name
        self.mesh = mesh

        self.location = transform[:3, 3]
        tf = RigidTransform(rotation=transform[:3, :3])
        R3_axis_angle = tf.axis_angle
        # theta, x, y, z
        theta = np.linalg.norm(R3_axis_angle)
        if theta != 0:
            self.rotation_axis_angle = np.hstack((theta, R3_axis_angle / theta))
        else:
            self.rotation_axis_angle = np.array([0, 0, 0, 1])

        self.parent = None
        self.children = []

        self.keyframes = {}
        self.keyposes = {}
        self.visual_node = None
        self.node = None

    @property
    def relative_transform(self):
        R3_axis_angle = np.array(self.rotation_axis_angle[1:]) * self.rotation_axis_angle[0] # update once r_ax_ang is always np array
        rot = RigidTransform.rotation_from_axis_angle(R3_axis_angle)
        tf = RigidTransform(rotation=rot, translation=self.location)
        return tf.matrix

    @property
    def global_transform(self):
        if self.parent is None:
            return self.relative_transform
        return self.parent.global_transform @ self.relative_transform

    def assign_parent(self, parent):
        self.parent = parent
        parent.children.append(self)

    def scale_uniform(self, scale):
        if self.mesh is not None:
            self.mesh.vertices *= scale
        self.location = np.array(self.location) * scale
        for c in self.children:
            c.scale_uniform(scale)

    def keyframe_insert(self, data_path, frame):
        # TODO: what should this do?
        self.keyframes[frame] = self.rotation_axis_angle # np.array(exec("self." + data_path + "[:]"))
        self.keyposes[frame] = self.relative_transform

    def update_visual(self, frame):
        if self.mesh is None:
            self.rotation_axis_angle = self.keyframes[frame]
            self.node.matrix = self.keyposes[frame]

    def assume_pose(self):
        self.node.matrix = self.relative_transform
        for c in self.children:
            c.assume_pose()

# Class for loading and tracking the Robotiq 2F 85 gripper mesh objects.
class Gripper:
    def __init__(self):
        meshBase = "ur5/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_"
        
        self.gripper_base = load_mesh("gripper_base", meshBase + "base_link.dae", 0.001)

        self.left_outer_knuckle = load_mesh("left_outer_knuckle", meshBase + "outer_knuckle.dae", 0.001)
        self.left_inner_knuckle = load_mesh("left_inner_knuckle", meshBase + "inner_knuckle.dae", 0.001)
        self.left_outer_finger  = load_mesh("left_outer_finger",  meshBase + "outer_finger.dae", 0.001)
        self.left_inner_finger  = load_mesh("left_inner_finger",  meshBase + "inner_finger.dae", 0.001)
        self.left_inner_finger_pad = load_mesh("left_inner_finger_pad", meshBase + "pad.dae", 0.001)
        
        self.right_outer_knuckle = load_mesh("right_outer_knuckle", meshBase + "outer_knuckle.dae", 0.001)
        self.right_inner_knuckle = load_mesh("right_inner_knuckle", meshBase + "inner_knuckle.dae", 0.001)
        self.right_outer_finger  = load_mesh("right_outer_finger",  meshBase + "outer_finger.dae", 0.001)
        self.right_inner_finger  = load_mesh("right_inner_finger",  meshBase + "inner_finger.dae", 0.001)
        self.right_inner_finger_pad = load_mesh("right_inner_finger_pad", meshBase + "pad.dae", 0.001)

        add_child(self.gripper_base, self.left_outer_knuckle)
        add_child(self.gripper_base, self.left_inner_knuckle)
        add_child(self.left_outer_knuckle, self.left_outer_finger)
        add_child(self.left_outer_finger,  self.left_inner_finger)
        add_child(self.left_inner_finger,  self.left_inner_finger_pad)

        add_child(self.gripper_base, self.right_outer_knuckle)
        add_child(self.gripper_base, self.right_inner_knuckle)
        add_child(self.right_outer_knuckle, self.right_outer_finger)
        add_child(self.right_outer_finger,  self.right_inner_finger)
        add_child(self.right_inner_finger,  self.right_inner_finger_pad)

        self.gripper_base.location = (0, 0.08230, 0)
        self.gripper_base.rotation_axis_angle = (-pi/2, 1, 0, 0)
        
        self.left_outer_knuckle.location = (0, -0.0306011, 0.054904)
        self.left_outer_knuckle.rotation_axis_angle = (pi, 0, 0, 1)
        self.left_inner_knuckle.location = (0, -0.0127, 0.06142)
        self.left_inner_knuckle.rotation_axis_angle = (pi, 0, 0, 1)
        self.left_outer_finger.location = (0, 0.0315, -0.0041)
        self.left_inner_finger.location = (0, 0.0061, 0.0471)

        self.right_outer_knuckle.location = (0, 0.0306011, 0.054904)
        self.right_inner_knuckle.location = (0, 0.0127, 0.06142)
        self.right_outer_finger.location = (0, 0.0315, -0.0041)
        self.right_inner_finger.location = (0, 0.0061, 0.0471)
        
        self.ee = Link("ee", None)
        self.ee.location = (0, 0, 0.129459)
        add_child(self.gripper_base, self.ee)

class SuctionGripper:
    def __init__(self, graspedInfo):
        f = open(graspedInfo);
        lines = f.readlines()
        i = 0
        while lines[i] not in ("capsule\n", "sphere\n"):
            i += 1

        i += 1
        endpt1 = np.array(lines[i].split(), dtype=np.float64)
        if lines[i-1] == "capsule\n":
            i += 1
            endpt2 = np.array(lines[i].split(), dtype=np.float64)
        else:
            endpt2 = endpt1 + np.array((0, 0, 1e-6))
        i += 1
        grasped_radius = float(lines[i])

        i += 1
        tube_length = float(lines[i])
        i += 1
        tube_radius = float(lines[i])

        self.gripper_base = Link("gripper_base", None)
        mesh = trimesh.creation.cylinder(tube_radius, tube_length)
        # grasped = trimesh.creation.icosphere(radius=grasped_radius)

        # endpt1 = np.array((-0.14, 0, grasped_radius))
        # endpt2 = np.array((0.01, 0, grasped_radius))
        midpt = (endpt1 + endpt2) / 2.0
        dir = endpt2 - endpt1
        length = np.linalg.norm(dir)
        if length != 0:
            dir /= length
        z = np.array([0, 0, 1])
        x = np.array([1, 0, 0])
        y = np.cross(dir, x)
        if np.linalg.norm(y) != 0:
            y /= np.linalg.norm(y)
        else:
            # dir = np.array([1, 0, 0])
            y = np.array([0, 0, 1])
        x = np.cross(y, dir)
        x /= np.linalg.norm(x)
        rot = np.vstack((x, y, dir)).T
        if length == 0:
            rot = np.eye(3)
        tf = RigidTransform(rotation=rot, translation=endpt1)
        # print(tf.matrix)
        grasped = trimesh.creation.capsule(length, grasped_radius)

        move_to_tip = tf.matrix # np.eye(4)
        move_to_tip[2, 3] += tube_length/2 #  + grasped_radius
        grasped.apply_transform(move_to_tip)
        mesh += grasped
        pose = np.eye(4)
        pose[2, 3] = tube_length/2
        self.tube = Link("suction_tube", mesh, pose)
        self.tube.assign_parent(self.gripper_base)
        self.gripper_base.location = (0, 0.08230, 0)
        self.gripper_base.rotation_axis_angle = (-pi/2, 1, 0, 0)

        self.ee = Link("ee", None)
        self.ee.location = (0, 0, tube_length)
        add_child(self.gripper_base, self.ee)

# Class for loading and tracking the UR5 robot meshes.  After loading,
# `set_config` can be used to change the robot's transforms, and
# `keyframe_insert` can be used to insert a keyframe for the
# configuration.
class UR5:
    ssh = False # account for additional latency
    suction = False
    def __init__(self, graspedInfo):
        meshBase="ur5/mesh/visual/"
         
        self.base = load_mesh("Base", meshBase + "Base.dae")

        self.shoulder = load_mesh("Shoulder", meshBase + "Shoulder.dae")
        self.upper_arm = load_mesh("UpperArm", meshBase + "UpperArm.dae")
        self.forearm = load_mesh("Forearm", meshBase + "Forearm.dae")
        self.wrist_1 = load_mesh("Wrist1", meshBase + "Wrist1.dae")
        self.wrist_2 = load_mesh("Wrist2", meshBase + "Wrist2.dae")
        self.wrist_3 = load_mesh("Wrist3", meshBase + "Wrist3.dae")
        
        if self.suction:
             self.gripper = SuctionGripper(graspedInfo)
        else:
             self.gripper = Gripper()
        
        add_child(self.base, self.shoulder)
        add_child(self.shoulder, self.upper_arm)
        add_child(self.upper_arm, self.forearm)
        add_child(self.forearm, self.wrist_1)
        add_child(self.wrist_1, self.wrist_2)
        add_child(self.wrist_2, self.wrist_3)

        add_child(self.wrist_3, self.gripper.gripper_base)

        self.shoulder.location = (0, 0, 0.089159)
        self.upper_arm.location = (0, 0.13585, 0)
        self.forearm.location = (0, -0.1197, 0.42500)
        self.wrist_1.location = (0, 0, 0.39225)
        self.wrist_2.location = (0, 0.09465, 0)
        self.wrist_3.location = (0, 0, 0.09465)

        self.scene_graph = pyrender.Scene()
        self.populate_scene_graph(self.base)
        self.gripper.gripper_base.assume_pose()
        self.links = [ self.shoulder, self.upper_arm, self.forearm, self.wrist_1, self.wrist_2, self.wrist_3 ]
        self.path_node = pyrender.Node(name="path")
        self.depth_map_node = pyrender.Node(name="depth_map")

    def populate_scene_graph(self, current):
        if current.mesh is not None:
            pymesh = pyrender.Mesh.from_trimesh(current.mesh)
            current.node = pyrender.Node(name=current.name, mesh=pymesh, matrix=current.relative_transform)
        else:
            current.node = pyrender.Node(name=current.name)

        if current.parent is not None:
            self.scene_graph.add_node(current.node, parent_node=current.parent.node)
        else:
            self.scene_graph.add_node(current.node)

        for c in current.children:
            self.populate_scene_graph(c)

    def set_config(self, config):
        # TODO: according to visualizing zero position and https://www.mdpi.com/machines/machines-09-00113/article_deploy/html/images/machines-09-00113-g001-550.jpg
        # these angle offsets are correct
        self.shoulder.rotation_axis_angle = (config[0], 0, 0, 1)
        self.upper_arm.rotation_axis_angle = (config[1] + pi/2, 0, 1, 0)
        self.forearm.rotation_axis_angle = (config[2], 0, 1, 0)
        self.wrist_1.rotation_axis_angle = (config[3] + pi/2, 0, 1, 0)
        self.wrist_2.rotation_axis_angle = (config[4], 0, 0, 1)
        self.wrist_3.rotation_axis_angle = (config[5], 0, 1, 0)

    def keyframe_insert(self, frame):
        for link in self.links:
            link.keyframe_insert(data_path='rotation_axis_angle', frame=frame)

    def render(self, current_frame, end_frame, frame_step, close=3, offscreen=True):
        num_frames = ((end_frame - current_frame) / frame_step) + 1
        if offscreen:
            print("Rendering...")
            images = []
            renderer = pyrender.OffscreenRenderer(2000, 2000) #, point_size=0.25)

            mcn = self.scene_graph.main_camera_node
            if mcn is not None:
                self.scene_graph.remove_node(mcn)
            camera_node = set_default_camera(self.scene_graph)

            if len(self.scene_graph.light_nodes) == 0:
                light = pyrender.light.DirectionalLight(color=np.ones(3), intensity=1.0)
                self.scene_graph.add(light, pose=np.eye(4))
                create_raymond_lights(self.scene_graph)

            for l in self.links:
                l.update_visual(current_frame)

            for frame in tqdm(range(current_frame + frame_step, end_frame + frame_step, frame_step)):
                image, _ = renderer.render(self.scene_graph)
                images.append(image)
                for l in self.links:
                    l.update_visual(frame)

            # fencepost!
            image, _ = renderer.render(self.scene_graph)
            images += [image]*10 # want a pause at the end of the gif loop so it isn't dizzying

            self.scene_graph.remove_node(camera_node)
            if mcn is not None:
                self.scene_graph.add_node(mcn)
                self.scene_graph.main_camera_node = mcn

            renderer.delete()

            for l in self.scene_graph.light_nodes:
                self.scene_graph.remove_node(l)

            clip = ImageSequenceClip(images, fps=7.8125) # 1/4 speed from 0.032 sec per frame
            clip.write_gif('rendered_motion.gif', fps=7.8125)
        else:
            viewer = pyrender.Viewer(self.scene_graph,
                                     run_in_thread=True,
                                     use_raymond_lighting=True,
                                     viewport_size=(2000, 2000))
            viewer._trackball.scroll(-2) # zoom out; TODO: this is terrible
            for l in self.links:
                l.update_visual(current_frame)
            time.sleep(3) # wait for window to appear
            if self.ssh:
                time.sleep(2)
            for frame in range(current_frame + frame_step, end_frame + frame_step, frame_step):
                time.sleep(0.05) # takes time for image to update
                if self.ssh:
                    time.sleep(0.7)
                viewer.render_lock.acquire()
                for l in self.links:
                    l.update_visual(frame)
                viewer.render_lock.release()
            time.sleep(close) # wait before closing window
            viewer.close_external()

    def add_path(self, current_frame, end_frame, frame_step, width=0.01, color=np.array((0, 0, 255))):
        prev_frame = current_frame
        curr_frame = current_frame + frame_step
        self.scene_graph.add_node(self.path_node)
        for l in self.links:
            l.update_visual(prev_frame)
        # TODO: gripper maw instead of gripper base
        colors = np.array(((255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)))
        prev_pos = self.gripper.ee.global_transform[:3, 3]
        while prev_frame != end_frame:
            color = colors[0] # np.roll(color, 1, axis=0)
            colors = np.roll(colors, -1, axis=0)
            for l in self.links:
                l.update_visual(curr_frame)
            curr_pos = self.gripper.ee.global_transform[:3, 3]

            # construct a cylinder
            midpt = (prev_pos + curr_pos) / 2.0
            dir = curr_pos - prev_pos
            length = np.linalg.norm(dir)
            if length != 0:
                dir /= length
            z = np.array([0, 0, 1])
            x = np.array([1, 0, 0])
            y = np.cross(dir, x)
            if np.linalg.norm(y) < 1e-6:
                z = np.array([1, 0, 0])
                x = np.array([0, 1, 0])
                y = np.cross(dir, x)
            y /= np.linalg.norm(y)
            x = np.cross(y, dir)
            x /= np.linalg.norm(x)
            rot = np.vstack((x, y, dir)).T
            if length == 0:
                rot = np.eye(3)
            if length < 10.0:
                tf = RigidTransform(rotation=rot, translation=prev_pos)
                mesh = trimesh.creation.capsule(length, width/2.0)
                rgba = np.append(color / 255.0, 1)
                pymesh = pyrender.Mesh.from_trimesh(mesh, material=MetallicRoughnessMaterial(baseColorFactor=rgba))
                node = pyrender.Node(name="path"+str(curr_frame), mesh=pymesh, matrix=tf.matrix)
                self.scene_graph.add_node(node, parent_node=self.path_node)

            prev_frame = curr_frame
            curr_frame += frame_step
            prev_pos = curr_pos

    def remove_path(self):
        self.scene_graph.remove_node(self.path_node)

    def add_depth_map(self, depth_map_file, capsules=True):
        self.scene_graph.add_node(self.depth_map_node)

        f = open(depth_map_file, 'r')
        lines = f.readlines()
        min_x, min_y = [float(i) for i in lines.pop(0).split()]
        max_x, max_y = [float(i) for i in lines.pop(0).split()]
        height = int(lines.pop(0))
        width = int(lines.pop(0))
        xres = (max_x - min_x) / width
        yres = (max_y - min_y) / height

        # row = np.array((height, width))
        bottom = -0.6
        mesh = None
        for i in range(height):
            row = np.array(lines[i].split()).astype(float)
            for j in range(width):
                pillar_ctr_x = min_x + j*xres + xres/2
                pillar_ctr_y = max_y - i*yres - yres/2
                pillar_height = row[j] - bottom
                rad = np.sqrt(xres**2 + yres**2) / 2

                tf = np.eye(4)
                tf[:3, 3] = [pillar_ctr_x, pillar_ctr_y, pillar_height/2 + (1-capsules)*rad/2 + bottom]
                if capsules:
                    tf[2, 3] = bottom

                if mesh is None:
                    if capsules:
                        mesh = trimesh.creation.capsule(height=pillar_height, radius=rad)
                        mesh.apply_transform(tf)
                    else:
                        mesh = trimesh.creation.box((xres, yres, pillar_height + rad), transform=tf)
                else:
                    if capsules:
                        pillar = trimesh.creation.capsule(height=pillar_height, radius=rad)
                        pillar.apply_transform(tf)
                        mesh += pillar
                    else:
                        mesh += trimesh.creation.box((xres, yres, pillar_height + rad), transform=tf)

        pymesh = pyrender.Mesh.from_trimesh(mesh, material=MetallicRoughnessMaterial(baseColorFactor=(0.3, 0.8, 0.3, 1.0)))
        node = pyrender.Node(name="pillars", mesh=pymesh)
        self.scene_graph.add_node(node, parent_node=self.depth_map_node)

    def remove_depth_map(self):
        self.scene_graph.remove_node(self.depth_map_node)

    def add_workspace(self, workspace):
        self.scene_graph.add_node(workspace.workspace_node)

    def remove_workspace(self, workspace):
        self.scene_graph.remove_node(workspace.workspace_node)

def usage(code=0):
    print("Usage: ./ur5_viz.py -- [options] FILE")
    print("Options:")
    print("  -h, --help            print this message and exit")
    print("  -s, --frame-step=INT  the number for frames between each waypoint")
    print("  -v                    enable verbose output")
    print()
    print("The '--' is required since Blender expects its own options, and")
    print("only passes arguments to the script after '--'.");

    sys.exit(code)

def load_traj(fileName):
    with open(fileName) as fp:
        # If we just have a sequence of waypoints, the following 1-liner does the job:
        # return [[float(s) for s in line.split()] for line in fp if line]
        #
        # However, now we have a file with multiple "data sets" which
        # are separated by empty lines.  We use 'takewhile' to get the
        # list up to the empty line.
        lines = itertools.takewhile(lambda x: x.strip(), fp)
        return [[float(s) for s in line.split()] for line in lines]
    
def main():
    verbose = False
    frame_step = 5
    
    try:
        args = sys.argv[1:]
        if "ssh" in args:
            UR5.ssh = True
            args.remove("ssh")
        print(args)
        opts, args = getopt.gnu_getopt(args, 'hvs:', ["help", "frame-step="])
    except ValueError:
        # "--" was not found
        print("no args")
        opts, args = [], []
    except getopt.GetoptError:
        usage(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-s", "--frame-step"):
            frame_step = int(arg)
            if frame_step < 1:
                sys.exit("--frame-step must be at least 1")
        else:
            assert False, "unhandled option"

    if not args:
        sys.exit("FILE must be specified.  Use '-- --help' for help.")

    trajs = [load_traj(f) for f in args]

    # Move the camera to a reasonable position (though this position
    # may need refinement...)
    # camera = bpy.data.objects['Camera']
    # camera.location = (0.274, 3.210, 1.544)
    # camera.rotation_euler = (1.129, 0.0, 3.110)

    # Change view to use camera (which requires finding which area of
    # the screen is the 3D view, and then changing it)
    # for area in bpy.context.screen.areas:
    #      if area.type == 'VIEW_3D':
    #         area.spaces[0].region_3d.view_perspective = 'CAMERA'

    # Create the meshes for the ur5
    ur5 = UR5()

    # Add in keyframes.  Blender seems to start at keyframe 1, though
    # it also seems capable of having negative keyframes.
    kf = 1
    for traj in trajs:
        for wp in traj:
            print(wp[0:7]) # time stamp is first element in blender
            ur5.set_config(wp[1:7])
            ur5.keyframe_insert(kf)
            kf = kf + frame_step

    # Set the end frame of the animation.
    end_frame = kf - frame_step

    # After inserting keyframes, set the current frame to the first frame.
    current_frame = 1

    # create pyrender environment
    ur5.render(current_frame, end_frame, frame_step)

if __name__ == "__main__":
    main()
    # argv = []
    # for i in range(len(sys.argv)):
    #     if sys.argv[i] == '--':
    #         argv = sys.argv[i+1:]
    #         break

