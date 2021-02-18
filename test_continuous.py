from algos.PPO_continuous import PPO, Memory
from envs.drone_env_human_follow_v1 import *
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
import time
import argparse

parser = argparse.ArgumentParser()
# env
parser.add_argument('--reward_version', type=str, default='v1')
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--save_gif', type=bool, default=False)
parser.add_argument('--policy_model', type=str, default='Gaussian')  # Beta
parser.add_argument('--use_temporal', type=bool, default=False)
# logging
parser.add_argument('--test_mode', type=bool, default=False)  # Disable all the logging if in test mode
# trajectory logging
parser.add_argument('--save_traj', type=bool, default=True)
parser.add_argument('--traj_log_interval', type=int, default=1)
parser.add_argument('--logging_folder', type=str, default='log_test_blocks_traj2_slow')
# model path
parser.add_argument('--model_dir', type=str, default='vishnu')
parser.add_argument('--model_id', type=int, default=2250)

args = parser.parse_args()

if args.test_mode:
    args.save_traj = False

if args.save_traj:
    folder_name = args.model_dir + '_' + str(args.model_id)
    if not os.path.isdir(args.logging_folder + '/' + folder_name):
        os.mkdir(args.logging_folder + '/' + folder_name)
        if not os.path.isdir(args.logging_folder + '/' + folder_name + '/figs/'):
            os.mkdir(args.logging_folder + '/' + folder_name + '/figs/')
        if not os.path.isdir(args.logging_folder + '/' + folder_name + '/traj/'):
            os.mkdir(args.logging_folder + '/' + folder_name + '/traj/')

def save_trajectory(i_episode, folder_name, trajectory, eplen, totaltrack, reward):
    # save figure
    plt.clf()
    plt.plot(trajectory.target_loc.x, trajectory.target_loc.y, 'ro-')
    plt.plot(trajectory.camera_loc.x, trajectory.camera_loc.y, 'bo-')
    plt.title('TotalTrack: '+ str(totaltrack) + '/' + str(eplen) + '; Reward: ' + str(round(reward, 2)))
    plt.savefig(args.logging_folder + '/' + folder_name + '/figs/traj_ep' + str(i_episode) + '.png')
    # save file
    traj = np.array([trajectory.target_loc.x, trajectory.target_loc.y, trajectory.target_loc.z,
                     trajectory.camera_loc.x, trajectory.camera_loc.y, trajectory.camera_loc.z,
                     trajectory.camera_rot.x, trajectory.camera_rot.y, trajectory.camera_rot.z,
                     trajectory.rel_loc.x, trajectory.rel_loc.y, trajectory.rel_loc.z,
                     trajectory.rel_rot.x, trajectory.rel_rot.y, trajectory.rel_rot.z,
                     trajectory.reward_history])
    np.savetxt(args.logging_folder + '/' + folder_name + '/traj/traj_ep' + str(i_episode) + '.txt', traj, fmt='%f', delimiter=',')


if __name__ == '__main__':

    # creating environment
    env_args = {'reward_version': args.reward_version,
                'policy_model': args.policy_model,
                'use_temporal': args.use_temporal}
    env = DroneEnvHumanFollowV1(env_args)  # gym.make(env_name)

    ############## Hyperparameters ##############
    env_name = "drone_env_human_follow_v1"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    n_episodes = 50  # num of episodes to run
    max_timesteps = 2000  # max timesteps in one episode

    # filename and directory to load model from
    model_name = './model/' + args.model_dir + '/' + 'PPO_continuous_drone_env_human_follow_v1_' + str(args.model_id) + '.pth'

    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(model_name))

    reward_history = []
    eplen_history = []
    track_history = []

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, done, info = env.step(action)
            print(f'Reward:{reward}, Action: {action}, Done: {done}')
            ep_reward += reward
            if args.render:
                env.render()
            if args.save_gif:
                img = env.render(mode='rgb_array')
                img = Image.fromarray(img)
                img.save('./gif/{}.jpg'.format(t))
            if done:
                break

        reward_history.append(ep_reward)
        eplen_history.append(t+1)
        track_history.append(info['totalTrack'])

        if args.save_traj and (ep % args.traj_log_interval == 0):
            save_trajectory(ep, folder_name, info['traj'], t+1, info['totalTrack'], ep_reward)

        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()

    # save file
    data = np.array([reward_history, eplen_history, track_history])
    np.savetxt(args.logging_folder + '/' + folder_name + '/res.txt', data, fmt='%f', delimiter=',')
