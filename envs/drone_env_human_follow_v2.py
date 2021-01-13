import airsim
import time
import copy
import numpy as np
from PIL import Image
import cv2
import gym
from gym.spaces import Box
from collections import OrderedDict

# from env.drone_env import drone_env
from env.drone_env_v2 import drone_env

np.set_printoptions(precision=3, suppress=True)
# -------------------------------------------------------
# height control
# continuous control

class drone_env_heightcontrol(drone_env):
    def __init__(self, start=[-23, 0, -10], aim=[-23, 60, -10], scaling_factor=2):
        super().__init__(start)
        
        self.scaling_factor = scaling_factor
        self.threshold = 3
        self.rand = True
        self.aim = np.array(aim)
        self.state = self.getState()
        if aim == None:
            self.rand = True
            self.start = np.array([0, 0, -10])
        
        self.lost_count = 0
        self.cur_pos = self.getCurPosition()
        
        # id = self.client.simSetSegmentationObjectID("Cart", 0)
        # print(id)
    
    def reset_aim(self):
        self.aim = (np.random.rand(3) * 100).astype("int") - 50
        self.aim[2] = -np.random.randint(10) - 5
        print("Our aim is: {}".format(self.aim).ljust(80, " "), end='\r')
    
    def reset(self):
        super().reset()
        self.state = self.getState()
        # print('State:', self.state)
        self.cur_pos = self.getCurPosition()
        self.lost_count = 0
        return self.state
    
    def getState(self):
        rel_pos, rel_orient = self.get_relloc_camera()
        state = np.concatenate((rel_pos, rel_orient/180.))
    
        return state
    
    def step(self, action):
        #angle = action[3]
        #action = self.local_to_world(action[:3])
        
        self.moveByDist(action, forward=False)
        
        # self.rotateByAngle(angle)
        
        state_ = self.getState()
        cur_pos = self.getCurPosition()
        
        info = None
        done = False
        
        best = np.array([0,0,4])
        cur_dis = self.distance(state_[0:3], best)
        if cur_dis < 2:
            reward = 2 - self.distance(state_[0:3], best) + 1
            info = "good range"
        else:
            if self.distance(best, state_[0:3]) < self.distance(best, self.state[0:3]):
                reward = 1
                info = "getting close"
            else:
                reward = -1
                info = "getting away"
       
        reward -= np.abs(state_[0])/10.
 
        h = 144
        w = 256
        K = np.array([[w / 2, 0, w / 2], [0, h / 2, h / 2], [0, 0, 1]])
        img_coord = K.dot(state_[0:3])
        img_coord = img_coord / img_coord[2]
        
        if img_coord[0] > w or img_coord[0] < 0 or img_coord[1] > h or img_coord[1] < 0 or state_[2] > 8 or state_[2] < 0:
            self.lost_count += 1
        else:
            if self.lost_count != 0:
                self.lost_count = 0
        
        # if self.isDone():
        #     done = True
        #     reward = 50
        #     info = "success"
        
        if self.lost_count >= 10:
            reward = -50
            done = True
            info = "lost"
        
        if self.client.simGetCollisionInfo().has_collided:
            reward = -50
            done = True
            info = "collision"
        
        if self.distance(self.cur_pos, cur_pos) < 1e-3:
            done = True
            info = "freeze"
            reward = -50
        
        self.cur_pos = cur_pos
        
        self.cur_step += 1
        self.state = state_
        infor = {}
        infor['info'] = info
        
        print("cur_step:", self.cur_step, "pos:", self.state, "action (w/ angle):", action, angle, "reward: ", reward, "distance", round(cur_dis,3), "info: ", info, "lost_count: ", self.lost_count)
        
        return state_, reward, done, infor
    
    def isDone(self):
        pos = self.getCurPosition()
        if self.distance(self.aim, pos) < self.threshold:
            return True
        return False
    
    def rewardf(self, state_old, state_cur):
        dis_old = self.distance(state_old["robot-position"], self.aim)
        dis_cur = self.distance(state_cur["robot-position"], self.aim)
        reward = dis_old - dis_cur
        return reward

