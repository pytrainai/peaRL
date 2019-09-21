
import gym
from mywrapper import MyWrapper
import matplotlib.pyplot as plt
from utils import random_play

env = gym.make('PongDeterministic-v0')

#using gym prepro class for atari
env = MyWrapper(env)
env.reset()
total_reward = 0
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
    print('\rTotal Reward: {:.2f}'.format(total_reward), end="")
    # pause
    #for _ in range(int(1e6)):
    #    pass
    if done: break
print('\n',len(obs))

#print postpro last obs
plt.figure(figsize=(9,9))
plt.axis('off')
img = plt.imshow(obs[0])
plt.show()
env.close()
print('\nSimulation finished!')