import os
import gym
import numpy as np
from collections import deque
import torch

class Process:
    """ Process class: object that instantiate a interaction step between
        the agent and the environment.

        Implements the step method given an agent, env and replay_buffer"""

    def __init__(self, agent, env, memory, 
                       batch_size, 
                       warmup_timesteps = int(1e3),
                       seed=42, 
                       process_name='foo_process', 
                       device="cpu"):
        """Initialize a Process object.

        Params
        ======
            agent (object): instance of the agent to be trained
            env (object): environment
            memory(object): instance of the RB used during training
            batch_size(int): batch size
            seed (int): random seed
            device (str): "cpu" or "cuda"
        """
        self.name = process_name
        self.agent = agent
        self.env = env
        self.memory = memory
        self.batch_size = batch_size
        self.warmup_timesteps = warmup_timesteps
        self.device = device
        self.max_episode_steps = env._max_episode_steps

        assert self.batch_size < self.warmup_timesteps, "Error: warmup_timesteps < batch_size "
        
        # Set numpy random seed
        np.random.seed(seed)

        # Get env features dims
        self.state_dim, self.action_dim, self.max_action = self.agent.state_dim, self.agent.action_dim, self.agent.max_action

        # Set initial Process variables
        self.total_timesteps = 0
        self.timesteps_since_eval = 0
        self.done = True

        # Reset Process
        self.reset()

    def reset(self):
        self.agent.noise.reset()
        self.obs_old = self.env.reset()

    def agent_act(self, obs, noise_epsilon):
        # One step action
        if self.total_timesteps < self.warmup_timesteps:
            action = self.env.action_space.sample()
        else:
            action = self.agent.select_action(obs, noise_epsilon)
        return action

    def step(self, episode_timesteps, noise_epsilon):     
        # One step in the mdp, pushed to memory
        action = self.agent_act(self.obs_old, noise_epsilon)
        obs_new, reward, done, _ = self.env.step(action)
        if episode_timesteps + 1 > self.max_episode_steps:
            done = False
            finished = True
        elif done: finished = True
        else: finished = False
        experience = (self.obs_old, action, reward, obs_new, done)
        self.memory.add(experience)
        self.obs_old = obs_new
        self.total_timesteps += episode_timesteps
        return reward, finished
    
    def agent_train(self, training_steps = 100):
        experiences = self.memory.sample(self.batch_size)
        actor_loss, critic_loss = self.agent.train_step(experiences, training_steps, self.batch_size)
        return actor_loss, critic_loss

    def agent_eval(self, number_of_eval_episodes):
        
        #evaluates the process over n episodes

        pass

