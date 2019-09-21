# bunch of utils 

from IPython import display
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import operator
import yaml
from gym import envs

def get_config(config_path):
    with open(config_path) as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data

def save_config(configs_dict):
        pass

def get_list_envs():
    envids = [spec.id for spec in envs.registry.all()]
    for envid in sorted(envids):
        print(envid)

# visual helper for transition batch, for jupyter notebooks only
def plot_transition(env, obs, plot_gray = True):
    plt.figure(figsize=(10,10))
    plt.axis('off')
    for i in range(env.obs_batch_size):
        plt.subplot(env.obs_batch_size,1,i+1)
        plt.axis('off')
        if plot_gray:
            plt.imshow(obs[i], cmap='gray')
        else:
            plt.imshow(obs[i])

# random play, for jupyter notebooks only
def random_play(env, steps):
    plt.figure(figsize=(4,4))
    obs = env.reset()
    img = plt.imshow(obs[0], cmap='gray')
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        img.set_data(obs[0]) 
        display.display(plt.gcf())
        display.clear_output(wait=True)

def batch_get(arr, idxs):
        '''Get multi-idxs from an array depending if it's a python list or np.array'''
        if isinstance(arr, (list, deque)):
                return np.array(operator.itemgetter(*idxs)(arr))
        else:
                return arr[idxs]
        
def set_attr(obj, attr_dict, keys=None):
        '''Set attribute of an object from a dict'''
        if keys is not None:
                # pick keys and return a subdict with {keys:attr_values} 
                attr_dict = dict((k,attr_dict[k]) for k in keys if k in attr_dict)
        for attr, val in attr_dict.items():
                setattr(obj, attr, val)
        return obj

def hidden_init(layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

def plot(rewards, file_name, smoothing_window = 10):
	scores = pd.DataFrame({'Scores':rewards})
	fig = plt.figure(figsize=(10,5))
	plt.grid(True)
	plt.style.use('seaborn-bright')
	rewards_smoothed = scores.rolling(smoothing_window, min_periods=smoothing_window).mean()
	plt.plot(rewards_smoothed)
	plt.xlabel("Episode")
	plt.ylabel("Episode Reward (Smoothed)")
	plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
	#plt.show()
	plt.savefig(file_name)

def save_kwargs(kwargs, file_name = 'test_kwargs.txt'):
    with open(file_name, 'w') as f:
        for key in kwargs.keys():
            f.write("%s,%s\n"%(key,kwargs[key]))
        f.close()
    print('Test kwargs saved in', file_name)

def save_scores(scores, file_name):
    with open(file_name, 'w') as f:
        for score in scores:
            f.write("%s\n"%(score))
        f.close()
    print('Test scores saved in', file_name)