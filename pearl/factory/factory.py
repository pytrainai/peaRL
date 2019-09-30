
# Factory methods ###############################
# 
# Return instances of components objects 
#
#################################################

import os
import gym
from pearl.utils.misc import *
from pearl.memory import replay_buffer as rb
from pearl.algos import *
import numpy as np
from collections import deque
import torch
import pearl.process.process as p
import pearl.trainer.trainer as t

def MakeConfig(config_file):

    configs = get_config(config_file)

    # make component configs
    config_env = configs['env']
    config_agent = configs['agent']
    config_memory = configs['memory']
    config_process = configs['process']
    config_trainer = configs['trainer']
    configs = (config_env, config_agent, config_memory, config_process, config_trainer)

    return configs


def MakeEnvironment(config_env):
    
    # Retrieve config parameters
    env_name = config_env['env_name']
    seed = config_env['env_seed']
    env_type = config_env['env_type']
    
    #Make and reset environment
    env = gym.make(env_name)
    env.seed(seed)
    env.reset()

    # Get env params
    env_dims = {}
    if env_type == 'continuous':
        env_dims['env_action_dim'] = env.action_space.shape[0]
        env_dims['env_max_action'] = env.action_space.high.item()
        env_dims['env_min_action'] = env.action_space.low.item()
        env_dims['env_state_dim'] = len(env.reset())
    if env_type == 'discrete':
        env_dims['env_action_dim'] = env.action_space.n
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
    
    # Make agent #################
    # DDPG
    if agent_name == 'ddpg':
        lr_actor = config_agent['agent_lr_actor']
        lr_critic = config_agent['agent_lr_critic']
        noise = config_agent['agent_noise']
        noise_seed = config_agent['agent_noise_seed']
        agent = ddpg_linear_algo.DDPG(env_dims, lr_actor, lr_critic, device, noise, noise_seed)
    
    # DQN
    elif agent_name == 'dqn':
        lr = config_agent['agent_lr']
        eps_initial = config_agent['agent_eps_initial']
        eps_final = config_agent['agent_eps_final']
        eps_decay = config_agent['agent_eps_decay']
        seed = config_agent['torch_seed']
        agent = dqn_linear_algo.DQN(env_dims, lr, eps_initial, eps_final, eps_decay, device, seed)
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


def MakeProcess(agent, env, memory, device,  config_process):

    # Retrieve config parameters
    process_name = config_process['process_name']
    seed = config_process['process_np_seed']
    batch_size = config_process['batch_size']
    process = p.Process(agent, env, memory, batch_size, seed, process_name, device)
    return process

def MakeTrainer(process, config_trainer):
    trainer = t.Trainer(process, config_trainer)
    return trainer

