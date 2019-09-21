import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.networks_body import *

class Actor(nn.Module):
    def __init__(self,
                    state_dim,
                    action_dim,
                    max_action = 1.,
                    body='conv', 
                    return_probs = False):
        super(Actor, self).__init__()
        if body is 'conv': body = ConvBody(state_dim)
        if body is 'linear': body = LinearBody(state_dim)
        self.body = body
        self.fc1_actor = nn.Linear(body.feature_dim, action_dim)
        self.actor_params = list(self.body.parameters()) + list(self.fc1_actor.parameters())
        self.max_action = max_action
        self.return_probs = return_probs
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1_actor.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        y = self.body(x)
        logits = self.fc1_actor(y)
        if self.return_probs:
            return F.sigmoid(logits)
        else:
            return self.max_action*torch.tanh(logits)

class Critic(nn.Module):
    def __init__(self,
                    state_dim,
                    action_dim,
                    body='conv'):
        super(Critic, self).__init__()
        if body is 'conv': body = CriticConvBody(state_dim)
        if body is 'linear': body = CriticLinearBody(state_dim, action_dim)
        self.body = body
        self.fc1_critic = nn.Linear(body.feature_dim, 1)
        self.critic_params = list(self.body.parameters()) + list(self.fc1_critic.parameters())
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1_critic.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state, action):
        y = self.body(state, action)
        return self.fc1_critic(y)

class Actor2(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, body):
		super(Actor2, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		self.max_action = max_action
	
	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x)) 
		return x 


class Critic2(nn.Module):
	def __init__(self, state_dim, action_dim, body):
		super(Critic2, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

	def forward(self, x, u):
		x = F.relu(self.l1(torch.cat([x, u], 1)))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x 
