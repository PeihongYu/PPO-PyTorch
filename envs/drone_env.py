import airsim
import time
import numpy as np
from PIL import Image
import gym
from gym.spaces import Box
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
import math

np.set_printoptions(precision=3, suppress=True)


class SubMemory():
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []

    def add_data(self, data):
        self.x.append(data[0])
        self.y.append(data[1])
        self.z.append(data[2])

    def clear_memory(self):
        del self.x[:]
        del self.y[:]
        del self.z[:]


class TrajMemory():
    def __init__(self):
        self.target_loc = SubMemory()
        self.camera_loc = SubMemory()
        self.camera_rot = SubMemory()
        self.rel_loc = SubMemory()
        self.rel_rot = SubMemory()
        self.reward_history = [0]

    def add_loc(self, target_loc, camera_loc, camera_rot, rel_loc, rel_rot):
        self.target_loc.add_data(target_loc)
        self.camera_loc.add_data(camera_loc)
        self.camera_rot.add_data(camera_rot)
        self.rel_loc.add_data(rel_loc)
        self.rel_rot.add_data(rel_rot)

    def add_reward(self, reward):
        self.reward_history.append(reward)

    def clear_memory(self):
        self.target_loc.clear_memory()
        self.camera_loc.clear_memory()
        self.camera_rot.clear_memory()
        self.rel_loc.clear_memory()
        self.rel_rot.clear_memory()
        self.reward_history = [0]


class DroneEnv(gym.Env):
    def __init__(self):
        self.cur_step = 0
        self.camera_id = '0'

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        orientation = airsim.to_quaternion(-np.pi / 6, 0, 0)
        self.client.simSetCameraOrientation('0', orientation)

        # obtain human id
        # -- Method 1
        obj_list = self.client.simListSceneObjects('^(cart|Cart)[\w]*')
        assert len(obj_list) == 1  # making sure there is only one human
        self.HUMAN_ID = obj_list[0]
        # -- Method 2
        # self.HUMAN_ID = "carla"

        self.trajectory = TrajMemory()

    def reset(self):
        self.cur_step = 0
        self.trajectory.clear_memory()
        
        cur_pos = self.get_cur_position()
        if math.isnan(cur_pos[0]) or math.isnan(cur_pos[1]) or math.isnan(cur_pos[2]) or self.client.simGetCollisionInfo().has_collided:
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            orientation = airsim.to_quaternion(-np.pi / 6, 0 ,0)
            self.client.simSetCameraOrientation('0', orientation)
        
        # set the starting position of the drone to be at 4 meters away from the human
        rel_pos = self.local_to_world(np.array([0, -2, -4]), 1) 
        rel_pos += np.random.rand(*rel_pos.shape)
        position = self.client.simGetObjectPose(self.HUMAN_ID).position
        position.z_val -= 1.7 / 2
        position.x_val += rel_pos[0]
        position.y_val += rel_pos[1]
        position.z_val += rel_pos[2]
        heading = self.client.simGetObjectPose(self.HUMAN_ID).orientation
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)

        # use songxiaocheng's airsim (https://github.com/songxiaocheng/AirSim)
        self.client.moveToPositionAsync(position.x_val, position.y_val, position.z_val, 1).join()

        # use official airsim
        # self.client.takeoffAsync().join()

        print("===== start position: ", self.v2t(position))

    def render(self, mode):
        if mode == 'rgb_array':
            return self.get_image(type='rgb')

    def move_by_dist(self, diff, ForwardOnly=False):
        duration = 1.0 #0.1
        if ForwardOnly:
            # 3D control, vehicle's front always points in the direction of travel
            self.client.moveByVelocityAsync(float(diff[0]), float(diff[1]), float(diff[2]), duration,
                                            drivetrain=airsim.DrivetrainType.ForwardOnly,
                                            yaw_mode=airsim.YawMode(False, 0))
        else:
            # 4D control, vehicle's front direction is controlled by diff[3]
            self.client.moveByVelocityAsync(float(diff[0]), float(diff[1]), float(diff[2]), duration,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(True, diff[3]))
        return 0

    def get_state(self):
        return OrderedDict()

    def get_cur_position(self):
        return self.v2t(self.client.getMultirotorState().kinematics_estimated.position)

    def get_cur_velocity(self):
        return self.v2t(self.client.getMultirotorState().kinematics_estimated.linear_velocity)

    def get_object_pose(self, object_id):
        pose = self.client.simGetObjectPose(object_id)
        # sometimes simGetObjectPose returns NaN values, we need to retry the call a number of
        # times until valid data is returned
        # reference: https://github.com/microsoft/AirSim/issues/2695
        while (math.isnan(pose.position.x_val) or math.isnan(pose.position.y_val) or
               math.isnan(pose.position.z_val) or math.isnan(pose.orientation.x_val) or
               math.isnan(pose.orientation.y_val) or math.isnan(pose.orientation.z_val) or
               math.isnan(pose.orientation.w_val)):
            pose = self.client.simGetObjectPose(object_id)
        return pose

    def get_relloc_camera(self):
        """
        Function to get position of human relative to the camera

        Returns
        -------
            rel_pos: Relative position (x, y, z). Numpy array
            rel_orient: if mode == angle: Relative orientation (roll, pitch, yaw). Numpy array
                        if mode == rot_mat: Rotation matix is returned
        """
        # Get human's pose
        human_pose = self.get_object_pose(self.HUMAN_ID)
        # Adjust human_pose to be at the human center, 1.7 is the human height
        human_pose.position.z_val -= 1.7 / 2
        # Get camera's pose
        camera_pose = self.client.simGetCameraInfo(self.camera_id).pose
        # drone_pose = self.client.simGetVehiclePose()

        # Get relative position
        rel_pos = (human_pose.position - camera_pose.position).to_numpy_array()

        # Get rotation matrix
        human_rot = R.from_quat(human_pose.orientation.to_numpy_array()).as_matrix()
        camera_rot = R.from_quat(camera_pose.orientation.to_numpy_array()).as_matrix()

        # Calculate transformation/rotation matrix
        rel_orient = (camera_rot.T).dot(human_rot)
        rel_pos = (camera_rot.T).dot(rel_pos)

        comp_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        rel_orient = (comp_rot.T).dot(rel_orient.dot(comp_rot))
        rel_pos = (comp_rot.T).dot(rel_pos)

        rot = R.from_matrix(rel_orient)
        rel_orient = rot.as_euler('zyx', degrees=True)

        camera_rot_euler = R.from_quat(camera_pose.orientation.to_numpy_array()).as_euler('zyx', degrees=True)
        # log trajectory
        self.trajectory.add_loc(self.v2t(human_pose.position), self.v2t(camera_pose.position),
                                self.v2t(camera_rot_euler),
                                self.v2t(rel_pos), self.v2t(rel_orient))

        return rel_pos, rel_orient

    def local_to_world(self, vec, flag):
        """
        Function to transform a vector from a local coordinate framework to the world framework

        :param vec: the input vector
        :param flag: 0: use camera as the reference local framework; 1: use the target human as reference
        """
        #
        # If vector has angle as well, split it
        angle = None
        if len(vec) == 4:
            # Convert angle to degrees
            angle = np.array(180. * vec[3] / np.pi)
            vec = vec[:3]

        if flag == 0:  # using camera
            pose = self.client.simGetCameraInfo(self.camera_id).pose
        else:  # using human
            pose = self.get_object_pose(self.HUMAN_ID)
        rot = R.from_quat(pose.orientation.to_numpy_array()).as_matrix()
        comp_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        res = rot.dot(comp_rot.dot(vec))

        if angle is not None:
            # If angle was input, append to the transformed vector
            res = np.append(res, angle)

        return res

    def get_image(self, type):
        image_size = 84

        if type == 'depth':
            imageType = airsim.ImageType.DepthPerspective
            pixels_as_float = True
        elif type == 'rgb':
            imageType = airsim.ImageType.Scene
            pixels_as_float = False
        else:
            imageType = airsim.ImageType.Segmentation
            pixels_as_float = False

        responses = self.client.simGetImages([airsim.ImageRequest(self.camera_id, imageType, pixels_as_float, False)])
        while responses[0].height == 0:
            responses = self.client.simGetImages(
                [airsim.ImageRequest(self.camera_id, imageType, pixels_as_float, False)])

        if pixels_as_float:
            img1d = np.array(responses[0].image_data_float, dtype=np.float)
        else:
            img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)

        if type == 'rgb':
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
        else:
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        if type == 'depth':
            image = Image.fromarray(img2d)
            im_final = np.array(image.resize((image_size, image_size)).convert('L'), dtype=np.float) / 255
            im_final.resize((image_size, image_size, 1))
        else:
            im_final = img2d

        return im_final

    def v2t(self, vect):
        if isinstance(vect, airsim.Vector3r):
            res = np.array([vect.x_val, vect.y_val, vect.z_val])
        else:
            res = np.array(vect)
        return res

    def distance(self, pos1, pos2):
        pos1 = self.v2t(pos1)
        pos2 = self.v2t(pos2)
        dist = np.linalg.norm(pos1 - pos2)
        return dist
