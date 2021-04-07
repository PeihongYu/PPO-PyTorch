'''
Contonuing experient checklist:
Comet experiment: L47-49
Folder name: L61-67
Model loading: L125-134
Start epoch: L148

'''
import matplotlib
matplotlib.use('Agg')

from comet_ml import Experiment, ExistingExperiment
from algos.PPO_continuous_v2 import *
from envs.drone_env_human_follow_v3 import *
import matplotlib.pyplot as plt
import os
import time
import argparse

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

parser = argparse.ArgumentParser()
# env
parser.add_argument('--reward_version', type=str, default="v1")
parser.add_argument('--render', type=bool, default=False)
# logging
parser.add_argument('--test_mode', type=bool, default=False)  # Disable all the logging if in test mode
# trajectory logging
parser.add_argument('--save_traj', type=bool, default=True)
parser.add_argument('--traj_log_interval', type=int, default=1)
# comet logging
parser.add_argument('--log_comet', type=bool, default=True)
parser.add_argument('--user_name', type=str, default="peihong")
# model saving
parser.add_argument('--save_model', type=bool, default=True)

args = parser.parse_args()

if args.test_mode:
    args.save_traj = False
    args.log_comet = False
    args.save_model = False

# setup comet
if args.log_comet:
    if args.user_name == 'peihong':
        experiment = Experiment(api_key="CC3qOVi4obAD5yimHHXIZ24HA", project_name="human-following",
                                workspace="peihongyu")
    elif args.user_name == 'vishnu':
        experiment = Experiment(api_key="NaB7y40lAj6qp3SyYRONzLiuJ", project_name="human-following-image",
                                workspace="vishnuds")
        # experiment = ExistingExperiment(api_key="NaB7y40lAj6qp3SyYRONzLiuJ", project_name="human-following", previous_experiment="9358b91f5ddc48d8b83db2a1105be9b8", workspace="vishnuds")
    else:
        print('User not recongnized. Raising Error')
        raise NameError

    print(f'Using Comet for logging for user {args.user_name}')

if args.save_traj:
    folder_name = str(int(time.time()))+'_image_v2'
    os.mkdir('model/' + folder_name)
    os.mkdir('log/' + folder_name)
    os.mkdir('log/' + folder_name + '/figs/')
    os.mkdir('log/' + folder_name + '/traj/')
    


def save_trajectory(i_episode, folder_name, trajectory):
    # save figure
    plt.clf()
    plt.plot(trajectory.target_loc.x, trajectory.target_loc.y, 'ro-')
    plt.plot(trajectory.camera_loc.x, trajectory.camera_loc.y, 'bo-')
    plt.savefig('log/' + folder_name + '/figs/traj_ep' + str(i_episode) + '.png')
    # save file
    traj = np.array([trajectory.target_loc.x, trajectory.target_loc.y, trajectory.target_loc.z,
                     trajectory.camera_loc.x, trajectory.camera_loc.y, trajectory.camera_loc.z,
                     trajectory.camera_rot.x, trajectory.camera_rot.y, trajectory.camera_rot.z,
                     trajectory.rel_loc.x, trajectory.rel_loc.y, trajectory.rel_loc.z,
                     trajectory.rel_rot.x, trajectory.rel_rot.y, trajectory.rel_rot.z,
                     trajectory.reward_history])
    np.savetxt('log/' + folder_name + '/traj/traj_ep' + str(i_episode) + '.txt', traj, fmt='%f', delimiter=',')
    

if __name__ == '__main__':

    env_name = "drone_env_human_follow_v1"
    render = args.render

    ############## Hyperparameters ##############
    solved_reward = 30000  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval

    max_episodes = 100000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 20 #500#100  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    env = DroneEnvHumanFollowV1(args.reward_version)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    '''
    print('Loading model:')
    
    # print('model/1616006199_image/PPO_continuous_{}.pth'.format(env_name))
    # ppo.policy.load_state_dict(torch.load('model/1616006199_image/PPO_continuous_{}.pth'.format(env_name), map_location=device))
    
    print('model/1616904531_image/PPO_continuous_drone_env_human_follow_v1_3000.pth')
    ppo.policy.load_state_dict(torch.load('model/1616904531_image/PPO_continuous_drone_env_human_follow_v1_2500.pth', map_location=device))
    
    print('Loading model:')
    print('preTrained/stationary_v2_human.pth')
    ppo.policy.load_state_dict(torch.load('preTrained/stationary_v2_human.pth', map_location=device)) 
    '''
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    reward_history = []
    rel_dis_history = []
    eplen_history = []
    loss = None

    # training loop
    start_ep = 1 #3001 #1501
    #for i_episode in range(1, max_episodes + 1):
    for i_episode in range(start_ep, max_episodes + 1):

        print("===== Episode: ", i_episode)
        state = env.reset()

        episode_reward = 0
        episode_rel_dis = 0

        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)

            action[0:3] = 1.2*action[0:3]

            state, reward, done, info = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                loss = ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward

            episode_reward += reward
            episode_rel_dis += info['rel_dis']

            if done:
                break

        avg_length += (t + 1)

        reward_history.append(episode_reward)
        rel_dis_history.append(episode_rel_dis / (t + 1))
        eplen_history.append(t + 1)

        if args.log_comet:
            experiment.log_metric("Reward", episode_reward, i_episode)
            experiment.log_metric("Loss", loss, i_episode)
            experiment.log_metric("EpLen", t + 1, i_episode)
            experiment.log_metric("TotalTrack", info['totalTrack'], i_episode)
            experiment.log_metric("RelDis", episode_rel_dis / (t + 1), i_episode)
            if (len(reward_history) > 100):
                experiment.log_metric("average reward of the last 100 episodes",
                                      sum(reward_history[-100:]) / 100, i_episode)
                experiment.log_metric("average relative distance of the last 100 episodes",
                                      sum(rel_dis_history[-100:]) / 100, i_episode)
                experiment.log_metric("average episode length of the last 100 episodes",
                                      sum(eplen_history[-100:]) / 100, i_episode)

        if args.save_traj and (i_episode % args.traj_log_interval == 0):
            #save_trajectory(i_episode, folder_name, info['traj'])
            pass

        if args.save_model:
            
            # save every 100 episodes
            if i_episode % 500 == 0:
                torch.save(ppo.policy.state_dict(),
                           'model/' + folder_name + '/PPO_continuous_{}_{}.pth'.format(env_name, i_episode))
            
            # save every 500 episodes
            if i_episode % 500 == 0:
                torch.save(ppo.policy.state_dict(), 'model/' + folder_name + '/PPO_continuous_{}.pth'.format(env_name))

        # stop training if avg_reward > solved_reward
        if args.save_model and (running_reward > (log_interval * solved_reward)):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(),
                       'model/' + folder_name + '/PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
