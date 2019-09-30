import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import random
from collections import deque, namedtuple
from pearl.models.dqn_linear_model import DQN_Linear

class DQN():
    """Interacts with and learns from the environment."""

    def __init__(self, env_dims, lr, eps_initial, eps_final, eps_decay, device, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.device = device
        self.state_dim =  env_dims['env_state_dim']
        self.action_dim = env_dims['env_action_dim']
        self.lr = lr
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.seed = seed
        self.noise = None
        self.eps = eps_initial

        # Q-Networks --> local and target
        self.qnet = DQN_Linear(self.state_dim, self.action_dim, self.seed).to(self.device)
        self.qnet_target = DQN_Linear(self.state_dim, self.action_dim, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def select_action(self, obs, noise = None): 
        # Select action according to policy
        obs = torch.from_numpy(obs).float().to(self.device)#.reshape(1,-1)
        self.qnet.eval()
        with torch.no_grad():
            action = self.qnet(obs).cpu().data.numpy()
        self.qnet.train()
        # Epsilon-greedy action selection
        if random.random() > self.eps :
            return np.argmax(action)
        else:
            return random.choice(np.arange(self.action_dim))
        # update epsilon
        self.eps = max(self.eps_final, self.eps_decay*self.eps)
        return action

    def train_step(self, experiences, training_steps, batch_size, discount=0.99, tau=0.001):
        for _ in range(training_steps):
            # Sample replay buffer 
            x, u, r, y, d = experiences
            obs_old = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            obs_new = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            targets = self.qnet_target(obs_new)
            
            max_targets, _ = torch.max(targets, 1)
            
            max_targets = max_targets.unsqueeze(1)
            
            q_target = reward + discount*max_targets*done
            q_expected = self.qnet(obs_old).gather(1, action.long())           
            
            loss = F.mse_loss(q_expected, q_target)
            
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()        

            # Update target net       
            self.update_target_networks(tau)        

        return loss             

    def update_target_networks(self, tau):

        for param, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)