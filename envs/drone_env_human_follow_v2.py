import numpy as np
from env.drone_env_v2 import drone_env

np.set_printoptions(precision=3, suppress=True)

class drone_env_human_follow_v2(drone_env):
    def __init__(self):
        super().__init__()
        
        self.state = self.getState()
         
        self.lost_count = 0
        self.total_track = 0
        self.cur_pos = self.getCurPosition()
    
    
    def reset(self):
        super().reset()
        self.state = self.getState()
        self.cur_pos = self.getCurPosition()
        self.lost_count = 0
        self.total_track = 0
        return self.state
    
    def getState(self):
        rel_pos, rel_orient = self.get_relloc_camera()
        state = np.concatenate((rel_pos, rel_orient/180.))
    
        return state
    
    def step(self, action):
        action = self.local_to_world(action, 0)
        self.moveByDist(action, forward=False)
        
        state_ = self.getState()
        cur_pos = self.getCurPosition()
        
        info = None
        done = False
        
        best = np.array([0,1,4])
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
            self.total_track += 1
            if self.lost_count != 0:
                self.lost_count = 0
        
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
        infor['rel_dis'] = self.distance(state_[0:3], np.zeros(3))
        infor['totalTrack'] = self.total_track
        infor['traj'] = self.trajectory
        
        print("cur_step:", self.cur_step, "pos:", self.state, "action (w/ angle):", action, "reward: ", reward, "distance", round(cur_dis,3), "info: ", info, "lost_count: ", self.lost_count)
        
        return state_, reward, done, infor 
