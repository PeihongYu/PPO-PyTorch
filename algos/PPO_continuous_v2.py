import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from algos.FCN16s import FCN16s

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.img_states = []
        self.pos_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.img_states[:]
        del self.pos_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class FeatureFusion(nn.Module):
    def __init__(self, state_dim, out_dim):
        super(FeatureFusion, self).__init__()
        
        self.fcn = FCN16s()
        self.img_nn =  nn.Linear(4032//16, 512)
        self.img_tanh = nn.Tanh()

        self.pos_nn = nn.Linear(6,16)
        self.pos_tanh = nn.Tanh()

        self.joint_nn  = nn.Linear(512+16, out_dim)
        self.joint_tanh = nn.Tanh()

    def forward(self, x):
        x1, x2 = x
        #print(f'x1 shape: {x1.shape}')
        #print(f'x2 shape: {x2.shape}')        

        x1 = self.fcn(x1)
        x1 = self.img_tanh(self.img_nn(x1))
        
        x2 = self.pos_tanh(self.pos_nn(x2))
        
        # print(x1.shape)
        # print(x2.shape)
        x = torch.cat((x1, x2), dim=1)

        x = self.joint_tanh(self.joint_nn(x))

        return x 
'''
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                FeatureFusion(state_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                FeatureFusion(state_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )

        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
'''
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                FeatureFusion(state_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 64),
                nn.Tanh(),
                nn.Linear(64, 16),
                nn.Tanh(),
                nn.Linear(16, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                FeatureFusion(state_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 64),
                nn.Tanh(),
                nn.Linear(64, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
                )

        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor((state[0], state[1]))
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.img_states.append(state[0])
        memory.pos_states.append(state[1])
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor((state[0], state[1]))

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic((state[0], state[1]))

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        fcn_pretrained_filename = '/scratch1/vishnuds/fcn16s_from_caffe.pth' 
        print('Loading pre-trained FCN models')
        self.policy.actor[0].fcn.load_state_dict(torch.load(fcn_pretrained_filename))
        self.policy.critic[0].fcn.load_state_dict(torch.load(fcn_pretrained_filename))
        self.policy_old.actor[0].fcn.load_state_dict(torch.load(fcn_pretrained_filename))
        self.policy_old.critic[0].fcn.load_state_dict(torch.load(fcn_pretrained_filename))
        
        print('Freezing FCN layers')
        for param in self.policy.actor[0].fcn.parameters():
            param.requires_grad = False
        
        for param in self.policy.critic[0].fcn.parameters():
            param.requires_grad = False
        
        for param in self.policy_old.actor[0].fcn.parameters():
            param.requires_grad = False

        for param in self.policy_old.critic[0].fcn.parameters():
            param.requires_grad = False

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        img_state = torch.FloatTensor(state[0].reshape((1,)+state[0].shape)).to(device)
        pos_state = torch.FloatTensor(state[1].reshape(1,-1)).to(device)
        return self.policy_old.act((img_state, pos_state), memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_img_states = torch.squeeze(torch.stack(memory.img_states).to(device), 1).detach()
        old_pos_states = torch.squeeze(torch.stack(memory.pos_states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate((old_img_states, old_pos_states), old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values.float(), rewards.float()) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return advantages.mean().item()

def main():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v2"
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode

    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()

