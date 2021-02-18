import numpy as np
from envs.drone_env import *

np.set_printoptions(precision=3, suppress=True)


class DroneEnvHumanFollowV1(DroneEnv):
    def __init__(self, env_args):
        super().__init__()

        self.reward_version = env_args['reward_version']
        self.policy_model = env_args['policy_model']
        self.use_temporal = env_args['use_temporal']

        if self.use_temporal:
            self.observation_space = Box(low=-100, high=100, shape=(24,))
        else:
            self.observation_space = Box(low=np.array([-100, -100, -100, -1, -1, -1]),
                                         high=np.array([100, 100, 100, 1, 1, 1]))
        # self.action_space = Box(low=-1, high=1, shape=(4,))
        self.action_space = Box(low=np.array([-5, -5, -5, -1]),
                                high=np.array([5, 5, 5, 1]))

        self.state_queue = []
        self.init_state_queue()
        self.state = self.get_state()
        self.cur_pos = self.get_cur_position()

        self.lost_count = 0
        self.freeze_count = 0
        self.total_track = 0

    def reset(self):
        super().reset()
        self.init_state_queue()
        self.state = self.get_state()
        self.cur_pos = self.get_cur_position()

        self.lost_count = 0
        self.freeze_count = 0
        self.total_track = 0
        return self.state

    def init_state_queue(self):
        self.state_queue.clear()
        for i in range(4):
            self.state_queue.append(np.zeros(6))

    def get_state(self):
        rel_pos, rel_orient = self.get_relloc_camera()
        cur_state = np.concatenate((rel_pos, rel_orient / 180.))

        if self.use_temporal:
            self.state_queue.pop()
            self.state_queue.append(cur_state)
            state = self.state_queue[0]
            for i in range(3):
                state = np.concatenate((state, self.state_queue[i+1]))
        else:
            state = cur_state

        return state

    def step(self, action):
        if self.policy_model == 'Beta':
            action = action * (self.action_space.high - self.action_space.low) + self.action_space.low
        action = self.local_to_world(action, 0)
        self.move_by_dist(action)

        cur_state = self.get_state()
        cur_pos = self.get_cur_position()

        reward, info, done = self.get_reward(cur_state[-6:], cur_pos)

        self.cur_pos = cur_pos
        self.cur_step += 1
        self.state = cur_state
        information = {}
        information['info'] = info
        information['rel_dis'] = self.distance(cur_state[-6:-3], np.array([0, 0, 4]))
        information['totalTrack'] = self.total_track
        information['traj'] = self.trajectory

        self.trajectory.add_reward(reward)

        print("cur_step:", self.cur_step, "pos:", self.state[-6:], "action:", action, "reward: ", round(reward, 3),
              ", distance", round(0, 3), ", info: ", info, ", lost_count: ", self.lost_count, "freeze_count: ",
              self.freeze_count)

        return cur_state, reward, done, information

    def get_reward(self, cur_state, cur_pos):
        info = None
        done = False

        best = np.array([0, 0, 4])
        cur_dis = self.distance(cur_state[0:3], best)
        if cur_dis < 2:
            reward = 2 - self.distance(cur_state[0:3], best) + 1
            reward_v1 = reward - 1
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
        elif self.client.simGetCollisionInfo().has_collided:
            reward = -50
            done = True
            info = "collision"

        if self.reward_version == 'v1':
            reward = reward_v1

        return reward, info, done

    def render(self, mode):
        return super().render(mode)
