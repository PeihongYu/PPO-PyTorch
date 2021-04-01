import numpy as np
from envs.drone_env import *
from algos.FCN16s import *
import torch

np.set_printoptions(precision=3, suppress=True)


class DroneEnvHumanFollowV1(DroneEnv):
    def __init__(self, reward_version='v1'):
        super().__init__()

        self.reward_version = reward_version

        self.observation_space = Box(low=np.array( [-100, -100, -100, -1, -1, -1] + [-100]*4032),
                                     high=np.array([100, 100, 100, 1, 1, 1] + [100]*4032))
        self.action_space = Box(low=-1, high=1, shape=(4,))
        
        self.img_model = FCN16s()
        self.img_model.load_state_dict(torch.load('/scratch1/vishnuds/fcn16s_from_caffe.pth'))
        self.img_model.cuda();
        self.img_model.eval();
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        print('FCN model loaded')
        

        self.state = self.get_state()
        self.cur_pos = self.get_cur_position()

        self.lost_count = 0
        self.freeze_count = 0
        self.total_track = 0
        

    def reset(self):
        super().reset()
        self.state = self.get_state()
        self.cur_pos = self.get_cur_position()

        self.lost_count = 0
        self.freeze_count = 0
        self.total_track = 0
        return self.state

    def get_state(self):
        rel_pos, rel_orient = self.get_relloc_camera()
        img_feat = self.get_img_feat()
        
        state = np.concatenate((rel_pos, rel_orient / 180., img_feat))

        return state

    def step(self, action):
        action = self.local_to_world(action, 0)
        self.move_by_dist(action)

        cur_state = self.get_state()
        cur_pos = self.get_cur_position()

        reward, info, done = self.get_reward(cur_state, cur_pos)

        self.cur_pos = cur_pos
        self.cur_step += 1
        self.state = cur_state
        information = {}
        information['info'] = info
        information['rel_dis'] = self.distance(cur_state[0:3], np.array([0, 0, 4]))
        information['totalTrack'] = self.total_track
        information['traj'] = self.trajectory

        self.trajectory.add_reward(reward)

        print("cur_step:", self.cur_step, "pos:", self.state[:6], "action:", action, "reward: ", round(reward, 3),
              ", distance", round(0, 3), ", info: ", info, ", lost_count: ", self.lost_count, "freeze_count: ", self.freeze_count)

        return cur_state, reward, done, information

    def get_reward(self, cur_state, cur_pos):
        info = None
        done = False

        best = np.array([0, 0, 4])
        cur_dis = self.distance(cur_state[0:3], best)
        if cur_dis < 2:
            reward = 2 - self.distance(cur_state[0:3], best) + 1
            reward_v1 = reward
            info = "good range"
        else:
            reward_v1 = 0
            if self.distance(best, cur_state[0:3]) < self.distance(best, self.state[0:3]):
                reward = 1
                info = "getting close"
            else:
                reward = -1
                info = "getting away"

        h = 144
        w = 256
        K = np.array([[w / 2, 0, w / 2], [0, w / 2, h / 2], [0, 0, 1]])
        img_coord = K.dot(cur_state[0:3])
        img_coord = img_coord / img_coord[2]

        if img_coord[0] > w or img_coord[0] < 0 or img_coord[1] > h or img_coord[1] < 0 or cur_state[2] > 8 or \
                cur_state[2] < 0:
            self.lost_count += 1
            reward -= 0.5
        else:
            self.total_track += 1
            if self.lost_count != 0:
                self.lost_count = 0

        if self.distance(self.cur_pos, cur_pos) < 1e-3:
            self.freeze_count += 1
        else:
            if self.freeze_count != 0:
                self.freeze_count = 0

        if self.lost_count >= 10:
            reward = -50
            done = True
            info = "lost"
        elif self.freeze_count >= 10:
            reward = -50
            done = True
            info = "freeze"
        if self.client.simGetCollisionInfo().has_collided:
            reward = -50
            done = True
            info = "collision"

        if self.reward_version == 'v1':
            reward = reward_v1

        return reward, info, done

    def render(self, mode):
        return super().render(mode)

    def get_img_feat(self):
        img = self.get_image(type='rgb')[:,:,::-1]
        img = img.astype(np.float64) - self.mean_bgr
        pimg = img.transpose(2,0,1)[np.newaxis,:,:,:]
        pimg = torch.from_numpy(pimg).float()
        
        with torch.no_grad():
            img_feat = self.img_model(pimg.cuda())
        
        img_feat_np = img_feat.cpu().numpy()[0]
        
        return img_feat_np
