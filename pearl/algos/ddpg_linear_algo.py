# version adaptada al nuevo framework

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from pearl.noise.noises import *
from pearl.models.ddpg_linear_model import Actor, Critic

class DDPG(object):
    def __init__(self, env_dims, lr_actor, lr_critic, device, noise_type, noise_seed):
              
        self.device = device
        self.state_dim =  env_dims['env_state_dim']
        self.action_dim = env_dims['env_action_dim']
        self.max_action = env_dims['env_max_action']
        self.min_action = env_dims['env_min_action']

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr =  lr_actor)#, weight_decay = 1e-4)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = lr_critic)#, weight_decay = 1e-4)
        
        self.noise = self._noise(noise_type, noise_seed)

    def _noise(self, noise_type, noise_seed):
        if noise_type == 'OUNoise':
            noise_process = OUNoise(self.action_dim, noise_seed)
        else:
            noise_process = None
        return noise_process

    def select_action(self, obs, noise_epsilon = 1.): 
        # Select action according to policy
        obs = torch.from_numpy(obs).float().to(self.device)#.reshape(1,-1)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy()
        if self.noise is not None:
            action += self.noise.sample(noise_epsilon)
            action = action.clip(self.min_action, self.max_action)
        self.actor.train()
        return action

    def train_step(self, experiences, training_steps, batch_size, discount=0.99, tau=0.005):
        for _ in range(training_steps):
            # Sample replay buffer 
            x, u, r, y, d = experiences
            obs_old = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            obs_new = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value
            target_Q = self.critic_target(obs_new, self.actor_target(obs_new))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(obs_old, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(obs_old, self.actor(obs_old)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            self.update_target_networks(tau)

        return (actor_loss, critic_loss)

    # Update the frozen target models
    def update_target_networks(self, tau):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        
    # Save and load models    
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
