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


class traj_memory():
    def __init__(self):
        self.target_loc_x = []
        self.target_loc_y = []
        self.target_loc_z = []
        self.camera_loc_x = []
        self.camera_loc_y = []
        self.camera_loc_z = []

    def add_loc(self, target, camera):
        self.target_loc_x.append(target[0])
        self.target_loc_y.append(target[1])
        self.target_loc_z.append(target[2])
        self.camera_loc_x.append(camera[0])
        self.camera_loc_y.append(camera[1])
        self.camera_loc_z.append(camera[2])

    def clear_memory(self):
        del self.target_loc_x[:]
        del self.target_loc_y[:]
        del self.target_loc_z[:]
        del self.camera_loc_x[:]
        del self.camera_loc_y[:]
        del self.camera_loc_z[:]

class drone_env(gym.Env):
    def __init__(self):
        self.cur_step = 0
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # obtain human id
        # -- Method 1
        obj_list = self.client.simListSceneObjects('^(cart|Cart)[\w]*')
        assert len(obj_list) == 1  # making sure there is only one human
        self.HUMAN_ID = obj_list[0]
        # -- Method 2
        # self.HUMAN_ID = "carla"

        self.trajectory = traj_memory()

        self.observation_space = Box(low=np.array([-100, -100, -100, -1, -1, -1]),
                                     high=np.array([100, 100, 100, 1, 1, 1]))
        self.action_space = Box(low=-1, high=1, shape=(3,))

    def reset(self):
        self.client.reset()
        self.cur_step = 0
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.trajectory.clear_memory()

        # set the starting position of the drone to be at 4 meters away from the human
        rel_pos = self.local_to_world(np.array([0, -1, -4]), 1)
        position = self.client.simGetObjectPose(self.HUMAN_ID).position
        position.x_val += rel_pos[0]
        position.y_val += rel_pos[1]
        position.z_val += rel_pos[2]
        heading = self.client.simGetObjectPose(self.HUMAN_ID).orientation
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)
        self.client.moveToPositionAsync(position.x_val, position.y_val, position.z_val, 1)

        print("start position: ", [position.x_val, position.y_val, position.z_val])

        time.sleep(2)

    def moveByDist(self, diff, ForwardOnly=True):
        if ForwardOnly:
            # vehicle's front always points in the direction of travel
            self.client.moveByVelocityAsync(float(diff[0]), float(diff[1]), float(diff[2]), 1,
                                            drivetrain=airsim.DrivetrainType.ForwardOnly,
                                            yaw_mode=airsim.YawMode(False, 0)).join()
        else:
            # vehicle's front direction is controlled by diff[3]
            self.client.moveByVelocityAsync(float(diff[0]), float(diff[1]), float(diff[2]), 1,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(False, diff[3])).join()
        return 0

    def getState(self):
        return OrderedDict()

    def getObjectPosition(self, name):
        return self.v2t(self.client.simGetObjectPose(name).position)

    def getCurPosition(self):
        return self.v2t(self.client.getMultirotorState().kinematics_estimated.position)

    def getCurVelocity(self):
        return self.v2t(self.client.getMultirotorState().kinematics_estimated.linear_velocity)

    def get_relloc_camera(self, camera_id='0'):
        """
        Function to get position of human relative to the camera

        Parameters
        ----------
            camera_id: ID of the camera from which we want to observe the environment

        Returns
        -------
            rel_pos: Relative position (x, y, z). Numpy array
            rel_orient: if mode == angle: Relative orientation (roll, pitch, yaw). Numpy array
                        if mode == rot_mat: Rotation matix is returned
        """
        # Get human's pose
        human_pose = self.get_object_pose(self.HUMAN_ID)
        # Get camera's pose
        camera_pose = self.client.simGetCameraInfo(camera_id).pose
        # drone_pose = self.client.simGetVehiclePose()

        #log trajectory
        self.trajectory.add_loc(self.v2t(human_pose.position), self.v2t(camera_pose.position))

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

        return rel_pos, rel_orient

    def local_to_world(self, vec, flag, camera_id='0'):
        """
        Function to transform a vector from a local coordinate framework to the world framework

        :param vec: the input vector
        :param flag: 0: use camera as the reference local framework; 1: use the target human as reference
        """
        #
        if flag == 0:  # using camera
            pose = self.client.simGetCameraInfo(camera_id).pose
        else:  # using human
            pose = self.get_object_pose(self.HUMAN_ID)
        rot = R.from_quat(pose.orientation.to_numpy_array()).as_matrix()
        comp_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        res = rot.dot(comp_rot.dot(vec))
        return res

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

    def getImg(self, type):
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

        responses = self.client.simGetImages([airsim.ImageRequest(0, imageType, pixels_as_float, False)])
        while responses[0].height == 0:
            responses = self.client.simGetImages([airsim.ImageRequest(0, imageType, pixels_as_float, False)])

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
