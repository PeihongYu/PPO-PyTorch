import airsim
import time
import copy
import numpy as np
from PIL import Image
import cv2
import gym
from gym.spaces import Box
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=3, suppress=True)
height_control_version = True
fix_start = False

class drone_env(gym.Env):
    def __init__(self, start=[0, 0, -5]):
        self.start = np.array(start)
        self.cur_step = 0
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync(1).join()
        
        obj_list = self.client.simListSceneObjects('^(cart|Cart)[\w]*')
        assert len(obj_list) == 1  # making sure there is only one human
        self.HUMAN_ID = obj_list[0]
        # self.HUMAN_ID = "carla"
        
        self.observation_space = Box(low=np.array([-100, -100, -100, -1, -1, -1]),# -1, -1, -1]),
                                     high=np.array([100, 100, 100, 1, 1, 1]))# 1, 1, 1]))
        self.action_space = Box(low=-1, high=1, shape=(4,))#shape=(3,))
    
    @property
    def dof(self):
        return 4 #3
        # if height_control_version == True:
        #     return 1
        # else:
        #     return 3
    
    def observation_spec(self):
        observation = self.getState()
        return observation
    
    def action_spec(self):
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
        return low, high
   
    def get_action_space(self):
        return self.action_space
 
    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync(1).join()


        self.cur_step = 0
        # self.client.enableApiControl(True)
        # self.client.armDisarm(True)
        start_pos = (self.client.simGetObjectPose(self.HUMAN_ID).position).to_numpy_array().astype(np.float64) #self.getObjectPosition(self.HUMAN_ID)
        
        start_pos[0] -= (np.random.random() - 0.5)*10.
        start_pos[1] -= (np.random.random() - 0.5)*10.
        start_pos[2] -= 1
        
        # self.client.moveToPositionAsync(start[0], start[1], start[2], 5).join()
        # self.client.hoverAsync().join()
        time.sleep(1) 
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(start_pos[0], start_pos[1], start_pos[2]), airsim.to_quaternion(0, 0, 0)), True)
        
        
        self.client.takeoffAsync(1).join()
    
        # position = self.client.simGetObjectPose(self.HUMAN_ID).position
        # # position.z_val += 2
        # heading = self.client.simGetObjectPose(self.HUMAN_ID).orientation
        # pose = airsim.Pose(position, heading)
        # self.client.simSetVehiclePose(pose, True)
    
        time.sleep(0.5)
    
    def moveByDist(self, diff, forward=False):
        self.client.moveByVelocityAsync(float(diff[0]), float(diff[1]), float(diff[2]), 1,
                                        drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode = airsim.YawMode()).join()
                                        # drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False,0)).join()
        
        
        # quad_vel = self.getCurVelocity()
        # self.client.moveByVelocityAsync(quad_vel[0] + diff[0], quad_vel[1] + diff[1],
        #                                 quad_vel[2] + diff[2], 1).join()
        # time.sleep(0.5)
        return 0
    
    def rotateByAngle(self, angle):
        self.client.rotateByYawRateAsync(int(180. * angle/np.pi) ,1).join()
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
        human_pose = self.client.simGetObjectPose(self.HUMAN_ID)
        # Get camera's pose
        camera_pose = self.client.simGetCameraInfo(camera_id).pose
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
    
        return rel_pos, rel_orient
    
    def local_to_world(self, action, camera_id='0'):
        camera_pose = self.client.simGetCameraInfo(camera_id).pose
        camera_rot = R.from_quat(camera_pose.orientation.to_numpy_array()).as_matrix()
        comp_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        res = camera_rot.dot(comp_rot.dot(action))
        return res
    
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

