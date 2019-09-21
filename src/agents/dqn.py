import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import random
from collections import deque, namedtuple
from model import QNetwork
from replay_buffer import ReplayBuffer
from argparse import Namespace    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, **kwargs):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            kwargs: training keyword arguments
        """
        #kwargs
        self.BUFFER_SIZE = kwargs.pop('BUFFER_SIZE' , int(1e6))  
        self.BATCH_SIZE = kwargs.pop('BATCH_SIZE' , 64)              
        self.GAMMA = kwargs.pop('GAMMA' , 0.99 )                 
        self.TAU = kwargs.pop('TAU' , 1e-3 )                   
        self.LR = kwargs.pop('LR' , 0.001  )                  
        self.UPDATE_EVERY = kwargs.pop('UPDATE_EVERY' , 10 )           
        self.ACT_EVERY = kwargs.pop('ACT_EVERY' , 1 )                
        self.FC1 = kwargs.pop('SEED' , 32)
        self.FC2 = kwargs.pop('SEED' , 64)
        self.BN = kwargs.pop('BN' , True)
        self.SEED = kwargs.pop('SEED' , 42)

        #cast string to bool
        if self.BN == 'True':
            self.BN = True
        else:
            self.BN = False

        self.state_size = state_size
        self.action_size = action_size
        random.seed(self.SEED)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, self.SEED, fc1_units=self.FC1, fc2_units=self.FC2, batch_norm=self.BN).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.SEED, fc1_units=self.FC1, fc2_units=self.FC2, batch_norm=self.BN).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.SEED, device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Act every S steps
        self.t_act = 2       #fixed only at the beginning
        self.action_last = 0 #fixed only at the beginning
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE: #only for the first time, then it will always be true
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Act every time steps.
        self.t_act = (self.t_act + 1) % self.ACT_EVERY
        if self.t_act == 0:
            # Epsilon-greedy action selection
            if random.random() > eps:
                action = np.argmax(action_values.cpu().data.numpy()).astype(int)
            else:
                action = random.choice(np.arange(self.action_size))
        else:
            action = self.action_last
        
        self.action_last = action
        return action
        
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        targets = self.qnetwork_target(next_states)
        max_targets, _ = torch.max(targets, 1)       
        max_targets = max_targets.unsqueeze(1)
        q_target = rewards + self.GAMMA*max_targets*(1-dones)
        q_expected = self.qnetwork_local(states).gather(1, actions)           
        
        loss = F.mse_loss(q_expected, q_target)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
