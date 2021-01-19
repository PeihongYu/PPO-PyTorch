output_is_3d = False

import gym
from algos.PPO_continuous import PPO, Memory
from PIL import Image
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# creating environment
if output_is_3d:
    from envs.drone_env_human_follow_v1 import *
    env = drone_env_human_follow_v1()  # gym.make(env_name)
else:
    from envs.drone_env_human_follow_v2 import *
    env = drone_env_human_follow_v2()  # gym.make(env_name)

def test():
    ############## Hyperparameters ##############
    env_name = "drone_env_human_follow_v1"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    n_episodes = 5          # num of episodes to run
    max_timesteps = 1500    # max timesteps in one episode
    render = False           # render the environment
    save_gif = True        # png images are saved in gif folder
    
    # filename and directory to load model from
    filename = "PPO_continuous_" +env_name+ ".pth"
    directory = "./preTrained/"

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    
    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory+filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            print(f'Reward:{reward}, Action: {action}, Done: {done}')
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()
    
if __name__ == '__main__':
    test()
    
    
