
# Factory methods ###############################
# 
# Return instances of components objects 
#
#################################################

import os
import gym
from utils.misc import *
from memory import replay_buffer as rb
from agents import ddpg
import numpy as np
from collections import deque
import torch
import process as p
import trainer as t

def MakeConfig(config_paths):

    # Retrieve component configs
    config_env = get_config(config_paths["env"])
    config_agent = get_config(config_paths["agent"])
    config_memory = get_config(config_paths["memory"])
    config_process = get_config(config_paths["process"])
    config_trainer = get_config(config_paths["trainer"])
    configs = (config_env, config_agent, config_memory, config_process, config_trainer)

    return configs


def MakeEnvironment(config_env):
    
    # Retrieve config parameters
    env_name = config_env['env_name']
    seed = config_env['env_seed']
    
    #Make and reset environment
    env = gym.make(env_name)
    env.seed(seed)
    env.reset()

    # Get env params
    env_dims = {}
    env_dims['env_action_dim'] = env.action_space.shape[0]
    env_dims['env_max_action'] = env.action_space.high.item()
    env_dims['env_min_action'] = env.action_space.low.item()
    env_dims['env_state_dim'] = len(env.reset())

    return env, env_dims

def MakeAgent(config_agent, env_dims):
    
    # Retrieve config parameters
    agent_name = config_agent['agent_name']
    
    # Set torch seed
    torch.manual_seed(config_agent['torch_seed'])

    # Set device
    if config_agent['torch_use_cuda']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Setting ", agent_name, " agent on:", device)
    else:
        device = "cpu"
    
    used_device = {'used_device': device}
    
    # Make agent
    if agent_name == 'ddpg':
        lr_actor = config_agent['agent_lr_actor']
        lr_critic = config_agent['agent_lr_critic']
        noise = config_agent['agent_noise']
        noise_seed = config_agent['agent_noise_seed']
        agent = ddpg.DDPG(env_dims, lr_actor, lr_critic, device, noise, noise_seed)
    else:
        pass
    
    return agent, device, used_device

def MakeMemory(config_memory):

    # Retrieve config parameters
    memory_type = config_memory['memory_type']
    size = config_memory['memory_size']

    # Make replay buffer
    if memory_type == 'vanilla':
        memory = rb.ReplayBuffer(size)
    else:
        pass

    return memory


# import Process class

def MakeProcess(agent, env, memory, device,  config_process):

    # Retrieve config parameters
    process_name = config_process['process_name']
    seed = config_process['process_np_seed']
    warmup_timesteps = config_process['warmup_timesteps']
    batch_size = config_process['batch_size']
    process = p.Process(agent, env, memory, batch_size, warmup_timesteps, seed, process_name, device)
    return process

# import trainer class

def MakeTrainer(process, config_trainer):
    trainer = t.Trainer(process, config_trainer)
    return trainer

def MakeNet(net_class):
    pass