import os
import gym
import numpy as np
from collections import deque
import torch

class Trainer:
    """ Trainer class: object that instantiate a training
    routine for the process

        Implements the step method given an agent, env and replay_buffer"""

    def __init__(self, process, config_trainer):
        """Initialize a Trainer object """
        self.process = process
        self.max_episodes = config_trainer['max_episodes']
        self.warmup_timesteps = config_trainer['warmup_timesteps']
        self.noise_decay = config_trainer['noise_decay']
        self.train_every = config_trainer['train_every']
        self.print_every = config_trainer['print_every']
        self.noise_epsilon = 1.
        self.scores = []

    def warmup(self):
        for _ in range(self.warmup_timesteps):
            episode_timesteps = 0
            while True:
                # Perform one process step and get episode reward and finished boolean
                _, finished = self.process.step(episode_timesteps, self.noise_epsilon)
                episode_timesteps += 1
                if finished:
                    break

    def train(self):
        """ Main training routine """    
        
        # Run warmup before training
        self.warmup()
        
        rewards_deque = deque(maxlen=100)

        for episode_num in range(self.max_episodes):

            # Reset episode
            episode_reward = 0
            episode_timesteps = 0
            self.process.reset()

            while True: 
                # Perform one process step and get episode reward and done mask
                reward, finished = self.process.step(episode_timesteps, self.noise_epsilon)

                # Add reward to episode_reward
                episode_reward += reward
                episode_timesteps += 1
                self.noise_epsilon *= self.noise_decay
                
                if finished:
                    break

            # Append episode reward
            rewards_deque.append(episode_reward)

            # Train agent
            if episode_num % self.train_every == 0:
                actor_loss, critic_loss = self.process.agent_train(episode_timesteps)

            # Print every
            if episode_num % self.print_every == 0: 
                # Mean score
                avg_reward = np.mean(rewards_deque)
                self.scores.append(avg_reward)
                print('Episode Num: {:.2f} AvgReward: {:.2f} ActorLoss: {:.2f} CriticLoss: {:.2f} Epsilon: {:.2f}'.format(episode_num, episode_reward,actor_loss, critic_loss, self.noise_epsilon))
                    
        return self.scores

    def plot_agent_learning_curve(self, scores):    
        pass
