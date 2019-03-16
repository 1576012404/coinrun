import numpy as np
from coinrun import setup_utils,make
import tensorflow as tf
from baselines.common.vec_env import VecEnvWrapper
import time

from baselines.ppo2 import ppo2
import gym
from collections import deque

from baselines.common.vec_env.vec_monitor import VecMonitor
from course_learn.wrappers import CourierWrapper

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt100
import cv2





def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        # print("ortho_init", a, q)
        # import traceback
        # traceback.print_stack()
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format) + b

def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


def MyPolicy(X):

    agent_pos=X[:,0,0:2,1]
    goal_pos=X[:,0,2:4,1]
    all_pos=tf.concat((agent_pos,goal_pos),axis=-1)

    scaled_images = tf.cast(X[:,:,:,0:1], tf.float32) / 255.
    print("scaled_image", scaled_images)
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), ))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2),))
    h3 = conv_to_fc(h3)
    h3=tf.concat((h3,all_pos),axis=-1)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))



class MyReward(VecEnvWrapper):
    def __init__(self,env):
        super(MyReward,self).__init__(env)
        self.num_envs=env.num_envs
        self.m_Reward=np.zeros(self.num_envs)
        self.m_RewardHis=deque(maxlen=10)
        self.m_RewardHis100=deque(maxlen=100)
        self.m_Count=0

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        self.m_Count+=1
        obs,reward,done,info=self.venv.step_wait()
        self.m_Reward+=reward


        for i ,d in enumerate(done):
            if d:
                self.m_RewardHis.append(self.m_Reward[i])
                self.m_RewardHis100.append(self.m_Reward[i])
                self.m_Reward[i]=0
        if self.m_Count%1000==0:
            print("winrate",np.sum(self.m_RewardHis100)/100,self.m_RewardHis)

        # for i,r in enumerate(reward):
        #     if r>=1:
        #         reward[i]=10

        return obs,reward,done,info

class StickAct(VecEnvWrapper):
    def __init__(self, env,stick_prob=0.5):
        super(StickAct, self).__init__(env)
        self.num_envs = env.num_envs
        self.stick_prob=stick_prob
        self.HisAction=np.zeros(self.num_envs,dtype=np.int32)

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_async(self, actions):
        bStick=np.random.uniform(size=self.num_envs)
        bStick=bStick<self.stick_prob
        actions=np.where(bStick,self.HisAction,actions)
        self.HisAction=actions
        self.venv.step_async(actions)

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs, reward, done, info



def Train():
    setup_utils.setup_and_load(use_cmd_line_args=False, set_seed=3, num_levels=1, use_black_white=True, frame_stack=4)
    # env=make("platform",num_envs=8)
    env = make("platform", num_envs=128)
    env = CourierWrapper(env,False)
    env=MyReward(env)
    env=StickAct(env,0.5)
    env = VecMonitor(env)
    learning_rate=5e-4
    clip_range=0.2
    n_timesteps=int(1e8)
    hyperparmas = {'nsteps': 256, 'noptepochs': 4, 'nminibatches': 8, 'lr': learning_rate, 'cliprange': clip_range,
                   'vf_coef': 0.5, 'ent_coef': 0.01}

    act = ppo2.learn(
        network=MyPolicy,
        env=env,
        total_timesteps=n_timesteps,
        **hyperparmas,
        save_interval=100,
        log_interval=20,

        # value_network="copy"
    )

if __name__=="__main__":
    Train()





