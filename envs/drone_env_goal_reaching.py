import airsim
import time
import copy
import numpy as np
from PIL import Image
import cv2
import gym
from gym.spaces import Box
from collections import OrderedDict

from envs.drone_env import drone_env

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
        self.action_space = [-1, 0, 1]

        self.lost_count = 0

        # id = self.client.simSetSegmentationObjectID("Cart", 0)
        # print(id)

    def reset_aim(self):
        self.aim = (np.random.rand(3) * 100).astype("int") - 50
        self.aim[2] = -np.random.randint(10) - 5
        print("Our aim is: {}".format(self.aim).ljust(80, " "), end='\r')

    def reset(self):
        if self.rand:
            self.reset_aim()
        super().reset()
        self.state = self.getState()
        self.lost_count = 0
        return self.state

    def getState(self):
        state = super().getState()

        # state["depth"] = self.getImg('depth')
        # state["image"] = self.getImg('rgb')
        state["image"] = self.getImg('depth')
        state["robot-state"] = self.getCurPosition() - self.getObjectPosition("carla")
        state["robot-position"] = self.getCurPosition()
        state["robot-velocity"] = self.getCurVelocity()

        return state

    def step(self, action):

        # if self.dof == 1:
        dpos = self.state["robot-state"]
        temp = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
        dx = - dpos[0] / temp * self.scaling_factor
        dy = - dpos[1] / temp * self.scaling_factor
        dz = - self.action_space[action]
        self.moveByDist([dx, dy, dz], forward=True)

        # else:
        #     self.moveByDist(action, forward=True)

        state_ = self.getState()

        info = None
        done = False
        reward = self.rewardf(self.state, state_)

        reward = max(-1, 1 - abs(self.distance([0,0,0], state_["robot-state"]) - 3)/3)


        if abs(self.distance([0,0,0], state_["robot-state"])) > 5:
            self.lost_count += 1
        else:
            if self.lost_count != 0:
                self.lost_count = 0


        if state_["robot-position"][2] > -1:
            reward -= (state_["robot-position"][2] + 1)

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

        if self.distance(self.state["robot-position"], state_["robot-position"]) < 1e-3:
            done = True
            info = "freeze"
            reward = -50

        self.cur_step += 1
        self.state = state_
        infor = {}
        infor['info'] = info

        print("cur_step:", self.cur_step, "aim:", self.aim, "pos:", self.state["robot-position"], "action:", action, "  reward: ", reward, "info: ", info)

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
