log_comet = True
output_is_3d = True

if output_is_3d:
    from envs.drone_env_human_follow_v1 import *
    print('Using 3D output environment')    

    if log_comet:
            # import comet_ml in the top of your file
            from comet_ml import Experiment, ExistingExperiment
            # Add the following code anywhere in your machine learning file
            experiment = Experiment(api_key="CC3qOVi4obAD5yimHHXIZ24HA", project_name="human-following", workspace="peihongyu")

            print('Using Comet for logging')
            
else:
    from envs.drone_env_human_follow_v2 import *
    print('Using 4D output environment')
    
    if log_comet:
            # import comet_ml in the top of your file
            from comet_ml import Experiment, ExistingExperiment
            # Add the following code anywhere in your machine learning file
            experiment = Experiment(api_key="NaB7y40lAj6qp3SyYRONzLiuJ", project_name="human-following", workspace="vishnuds")
            
            print('Using Comet for logging')

from algos.PPO_continuous import *
import matplotlib.pyplot as plt
import os
import time

############## Hyperparameters ##############
env_name = "drone_env_human_follow_v1"
save_traj = True
render = False
solved_reward = 300         # stop training if avg_reward > solved_reward
log_interval = 20           # print avg reward in the interval
traj_log_interval = 1
max_episodes = 100000        # max training episodes
max_timesteps = 1500        # max timesteps in one episode

update_timestep = 100      # update policy every n timesteps
action_std = 0.5            # constant std for action distribution (Multivariate Normal)
K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr = 0.0003                 # parameters for Adam optimizer
betas = (0.9, 0.999)

random_seed = None
#############################################


# creating environment
if output_is_3d:
    env = drone_env_human_follow_v1()  # gym.make(env_name)
else:
    env = drone_env_human_follow_v2()  # gym.make(env_name)


if save_traj:
    folder_name = str(int(time.time()))
    os.mkdir('model/' + folder_name)
    os.mkdir('log/' + folder_name)
    os.mkdir('log/' + folder_name + '/figs/')
    os.mkdir('log/' + folder_name + '/traj/')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
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
for i_episode in range(1, max_episodes + 1):
    state = env.reset()

    episode_reward = 0
    episode_rel_dis = 0

    for t in range(max_timesteps):
        time_step += 1
        # Running policy_old:
        action = ppo.select_action(state, memory)
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

    avg_length += (t+1)

    reward_history.append(episode_reward)
    rel_dis_history.append(episode_rel_dis/(t+1))
    eplen_history.append(t+1)

    if log_comet:
        experiment.log_metric("Reward", episode_reward, i_episode)
        experiment.log_metric("Loss", loss, i_episode)
        experiment.log_metric("EpLen", t+1, i_episode)
        experiment.log_metric("TotalTrack", info['totalTrack'], i_episode)
        experiment.log_metric("RelDis", episode_rel_dis/(t+1), i_episode)
        if (len(reward_history) > 100):
            experiment.log_metric("average reward of the last 100 episodes", sum(reward_history[-100:]) / 100, i_episode)
            experiment.log_metric("average relative distance of the last 100 episodes", sum(rel_dis_history[-100:]) / 100, i_episode)
            experiment.log_metric("average episode length of the last 100 episodes", sum(eplen_history[-100:]) / 100, i_episode)

    if save_traj and i_episode % traj_log_interval == 0:
        # save figure
        plt.clf()
        plt.plot(info['traj'].target_loc.x, info['traj'].target_loc.y, 'ro-')
        plt.plot(info['traj'].camera_loc.x, info['traj'].camera_loc.y, 'bo-')
        plt.savefig('log/' + folder_name + '/figs/traj_ep' + str(i_episode) + '.png')
        # save file
        traj = np.array([info['traj'].target_loc.x, info['traj'].target_loc.y, info['traj'].target_loc.z,
                         info['traj'].camera_loc.x, info['traj'].camera_loc.y, info['traj'].camera_loc.z,
                         info['traj'].camera_rot.x, info['traj'].camera_rot.y, info['traj'].camera_rot.z,
                         info['traj'].rel_loc.x, info['traj'].rel_loc.y, info['traj'].rel_loc.z,
                         info['traj'].rel_rot.x, info['traj'].rel_rot.y, info['traj'].rel_rot.z,
                         info['traj'].reward_history])
        np.savetxt('log/' + folder_name + '/traj/traj_ep' + str(i_episode) + '.txt', traj, fmt='%f', delimiter=',')

    # stop training if avg_reward > solved_reward
    if running_reward > (log_interval * solved_reward):
        print("########## Solved! ##########")
        torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
        break

    # save every 500 episodes
    if i_episode % 100 == 0:
        torch.save(ppo.policy.state_dict(), 'model/PPO_continuous_{}_{}.pth'.format(env_name, i_episode))

    if i_episode % 500 == 0:
        torch.save(ppo.policy.state_dict(), 'model/PPO_continuous_{}.pth'.format(env_name))

    # logging
    if i_episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        running_reward = int((running_reward / log_interval))

        print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
