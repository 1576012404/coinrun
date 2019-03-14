import numpy as np
from coinrun import setup_utils,make
from train.wrappers import CourierWrapper

import time

from baselines.ppo2 import ppo2
from baselines.common.vec_env.vec_monitor import VecMonitor
import gym
from collections import deque
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2

class MyReward(gym.Wrapper):
    def __init__(self,env):
        super(MyReward,self).__init__(env)
        self.num_envs=env.num_envs
        self.m_Reward=np.zeros(self.num_envs)
        self.m_RewardHis=deque(maxlen=10)
        self.m_RewardHis100=deque(maxlen=100)

    def step(self,action):
        self.m_Count+=1
        obs,reward,done,info=self.env.step(action)
        self.m_Reward+=reward
        for i,done in enumerate(info):
            if done:
                self.m_RewardHis100.append(self.m_Reward)


        return obs,reward,done,info





def Train():
    setup_utils.setup_and_load(use_cmd_line_args=False, set_seed=3, num_levels=1, use_black_white=True, frame_stack=4)
    # env=make("platform",num_envs=8)
    env = make("platform", num_envs=8)
    env=CourierWrapper(env)
    env=VecMonitor(env)
    learning_rate=3e-4
    clip_range=0.2
    n_timesteps=int(1e8)
    hyperparmas = {'nsteps': 256, 'noptepochs': 4, 'nminibatches': 8, 'lr': learning_rate, 'cliprange': clip_range,
                   'vf_coef': 0.5, 'ent_coef': 0.01}

    act = ppo2.learn(
        network="cnn",
        env=env,
        total_timesteps=n_timesteps,
        **hyperparmas,
        save_interval=100,
        log_interval=100,

        # value_network="copy"
    )

if __name__=="__main__":
    Train()





