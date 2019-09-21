import gym
from utils.atari_preprocessing import AtariPreprocessing
import numpy as np
import cv2


class MyWrapper(AtariPreprocessing):
    def __init__(self, env, 
                       frame_skip = 4,
                       action_repeat = False,
                       obs_batch_size = 3, 
                       normalize_obs = True):
        super().__init__(env, 
                         noop_max = 30, 
                         frame_skip = frame_skip,
                         screen_size = 84, 
                         terminal_on_life_loss = False, 
                         grayscale_obs = True)
        
        assert obs_batch_size < frame_skip
        self.action_repeat = action_repeat
        self.obs_batch_size = obs_batch_size
        self.normalize_obs = normalize_obs
               
        # Extend obs_buffer to obs_batch_size
        if self.grayscale_obs:
            self.obs_buffer = [np.empty(env.observation_space.shape[:2], dtype=np.uint8) 
                               for _ in range(self.frame_skip)]
        else:
            self.obs_buffer = [np.empty(env.observation_space.shape, dtype=np.uint8) 
                               for _ in range(self.frame_skip)]
    
    # Overrite step() method from superclass
    def step(self, action):
        R = 0.0
            
        # repeats the same action obs_batch_size times
        for i in range(self.frame_skip):
            # first transition
            if i==0:
                _, reward, done, info = self.env.step(action)
            else:
                if self.action_repeat:
                    _, reward, done, info = self.env.step(action)
                else:
                    _, reward, done, info = self.env.step(0)
            
            R += reward
            self.game_over = done                    

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.lives = new_lives

            if done:
                break 

            # transform obs_batch[i] into Grayscale or RGB
            if self.grayscale_obs:
                self.ale.getScreenGrayscale(self.obs_buffer[i])
            else:
                self.ale.getScreenRGB2(self.obs_buffer[i])

        return self._get_obs(), R, done, info

    # overrite abstract get_obs method
    def _get_obs(self, pooling = True):
        obs = []
        for i in range(self.obs_batch_size):
            if pooling:
                obs.append(cv2.resize(np.maximum(self.obs_buffer[i], self.obs_buffer[i+1]),
                          (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA))
            else:
                obs.append(cv2.resize(self.obs_buffer[i], 
                          (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA))
            obs[i] = np.asarray(obs[i], dtype=np.uint8)
        # last frame removed if no pooling
        if not pooling:
            obs = obs[:-1]
        return obs
    
        