"""
Python interface to the CoinRun shared library using ctypes.

On import, this will attempt to build the shared library.
"""

import os
import atexit
import random
import sys
from ctypes import c_int, c_char_p, c_float, c_bool

import gym
import gym.spaces
import numpy as np
import numpy.ctypeslib as npct
from baselines.common.vec_env import VecEnv
from baselines import logger

from coinrun.config import Config

from mpi4py import MPI
from baselines.common import mpi_util
import cv2


# if the environment is crashing, try using the debug build to get
# a readable stack trace
DEBUG = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

game_versions = {
    'standard':   1000,
    'platform': 1001,
    'maze': 1002,
}

def build():
    lrank, _lsize = mpi_util.get_local_rank_size(MPI.COMM_WORLD)
    if lrank == 0:
        dirname = os.path.dirname(__file__)
        if len(dirname):
            make_cmd = "QT_SELECT=5 make -C %s" % dirname
        else:
            make_cmd = "QT_SELECT=5 make"

        r = os.system(make_cmd)
        if r != 0:
            logger.error('coinrun: make failed')
            sys.exit(1)
    MPI.COMM_WORLD.barrier()

build()

if DEBUG:
    lib_path = '.build-debug/coinrun_cpp_d'
else:
    lib_path = '.build-release/coinrun_cpp'

lib = npct.load_library(lib_path, os.path.dirname(__file__))
lib.init.argtypes = [c_int]
lib.get_NUM_ACTIONS.restype = c_int
lib.get_RES_W.restype = c_int
lib.get_RES_H.restype = c_int
lib.get_VIDEORES.restype = c_int

lib.vec_create.argtypes = [
    c_int,    # game_type
    c_int,    # nenvs
    c_int,    # lump_n
    c_bool,   # want_hires_render
    c_float,  # default_zoom
    ]
lib.vec_create.restype = c_int

lib.vec_close.argtypes = [c_int]

lib.vec_step_async_discrete.argtypes = [c_int, npct.ndpointer(dtype=np.int32, ndim=1)]

lib.initialize_args.argtypes = [npct.ndpointer(dtype=np.int32, ndim=1)]
lib.initialize_set_monitor_dir.argtypes = [c_char_p, c_int]

lib.vec_wait.argtypes = [
    c_int,
    npct.ndpointer(dtype=np.uint8, ndim=4),    # normal rgb
    npct.ndpointer(dtype=np.uint8, ndim=4),    # larger rgb for render()
    npct.ndpointer(dtype=np.float32, ndim=1),  # rew
    npct.ndpointer(dtype=np.bool, ndim=1),     # done
    ]

lib.vec_map_info.argtypes = [
    c_int,
    npct.ndpointer(dtype=np.float32, ndim=1),  # mapwalls
    npct.ndpointer(dtype=np.float32, ndim=1),  # ax
    npct.ndpointer(dtype=np.float32, ndim=1),  # ay
    ]

already_inited = False

def init_args_and_threads(cpu_count=4,
                          monitor_csv_policy='all',
                          rand_seed=None):
    """
    Perform one-time global init for the CoinRun library.  This must be called
    before creating an instance of CoinRunVecEnv.  You should not
    call this multiple times from the same process.
    """
    os.environ['COINRUN_RESOURCES_PATH'] = os.path.join(SCRIPT_DIR, 'assets')
    is_high_difficulty = Config.HIGH_DIFFICULTY

    if rand_seed is None:
        rand_seed = random.SystemRandom().randint(0, 1000000000)

        # ensure different MPI processes get different seeds (just in case SystemRandom implementation is poor)
        mpi_rank, mpi_size = mpi_util.get_local_rank_size(MPI.COMM_WORLD)
        rand_seed = rand_seed - rand_seed % mpi_size + mpi_rank

    int_args = np.array([int(is_high_difficulty), Config.NUM_LEVELS, int(Config.PAINT_VEL_INFO), Config.USE_DATA_AUGMENTATION, game_versions[Config.GAME_TYPE], Config.SET_SEED, rand_seed]).astype(np.int32)

    lib.initialize_args(int_args)
    lib.initialize_set_monitor_dir(logger.get_dir().encode('utf-8'), {'off': 0, 'first_env': 1, 'all': 2}[monitor_csv_policy])

    global already_inited
    if already_inited:
        return

    lib.init(cpu_count)
    already_inited = True

@atexit.register
def shutdown():
    global already_inited
    if not already_inited:
        return
    lib.shutdown()

class Goal:
    def __init__(self, rank=0):
        self.rank = rank
        self.walls = None
        self.current_goal = None
        self.past_goals = []
        self.debug = False
    
    # 设置目标点
    def set_goal(self, p, high, low):
        while True:
            dp = np.random.sample(len(high))*(high-low)+ low
            g = (p + dp).astype(np.int)
            if self.is_valid_goal(g):
                break
        self.current_goal = g
    
    # 是否到达目标
    def reach_goal(self, p):
        if np.sum(np.absolute(p.astype(int)-self.current_goal.astype(int))>1)==0:
            return True
        if np.sum(p.astype(int)!=self.current_goal.astype(int))==0:
            return True
        return False

    # 获取可通过区域
    def get_valid_walls(self):
        # empty=46 coin=49 ladder=61 box=35,36,37,38 plant=83,97,98, block=65
        accessable = np.isin(self.walls,[61,35,36,37,38])
        standable = 1-np.isin(self.walls,[46,49])
        kernel = np.array([[0,0,0],[1,1,0],[0,0,0]],np.uint8)
        reachable = cv2.dilate(standable.astype(np.uint8), kernel,iterations=7) - standable + accessable
        if self.debug:
            for i in np.unique(self.walls):
                if not i in [46,49,61,35,36,37,38,83,97,98,65]:
                    cv2.imshow(str(i), np.rot90(cv2.resize((self.walls==i)*1., (300,300), interpolation=cv2.INTER_NEAREST)))
            cv2.imshow('acc', np.rot90(cv2.resize(accessable*1., (300,300), interpolation=cv2.INTER_NEAREST)))
            cv2.imshow('reachable', np.rot90(cv2.resize(reachable*1., (300,300), interpolation=cv2.INTER_NEAREST)))
        return reachable
    
    # 判断目标点是否合法
    def is_valid_goal(self, g):
        # 没有越界
        if np.sum(g<0)+ np.sum(g>=self.walls.shape) <= 0:
            # 可通过
            if self.valid_walls[tuple(g)]:
                return True
        return False
    
    def reset(self):
        self.walls = None
        self.current_goal = None
        self.past_goals = []

    def step(self, p, walls):
        if self.walls is None or np.sum(self.walls != walls)>0:
            self.walls = walls
            self.valid_walls = self.get_valid_walls()

        self.p = p.astype(int)
        reward = 0
        high = np.array([20, 20])
        low = np.array([-20, -20])
        if self.current_goal is None:
            self.set_goal(p, high, low)
        if self.reach_goal(p):
            reward = 1
            self.past_goals.append(self.current_goal)
            self.set_goal(p, high, low)
        return reward

    def render(self):
        goals_img = self.valid_walls*0.3
        # goals_img = np.zeros_like(self.walls)
        # print(np.unique(self.walls, return_counts=True))
        for g in self.past_goals:
            goals_img[tuple(g)] = 0.8
        goals_img[tuple(self.current_goal)] = 1
        goals_img[tuple(self.p)] = 1
        # print(self.past_goals)
        cv2.imshow('goals'+str(self.rank), np.rot90(cv2.resize(goals_img, (300,300), interpolation=cv2.INTER_NEAREST)))
        # cv2.imshow('walls'+str(self.rank), np.rot90(cv2.resize(self.valid_walls*1., (300,300), interpolation=cv2.INTER_NEAREST)))
        cv2.waitKey(1)


class CourierVecEnv(VecEnv):
    def __init__(self, game_type, num_envs, lump_n=0, default_zoom=5.0):
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))

        self.NUM_ACTIONS = lib.get_NUM_ACTIONS()
        self.RES_W       = lib.get_RES_W()
        self.RES_H       = lib.get_RES_H()
        self.VIDEORES    = lib.get_VIDEORES()

        self.buf_rew = np.zeros([num_envs], dtype=np.float32)
        self.buf_done = np.zeros([num_envs], dtype=np.bool)
        self.buf_rgb   = np.zeros([num_envs, self.RES_H, self.RES_W, 3], dtype=np.uint8)
        self.hires_render = Config.IS_HIGH_RES
        if self.hires_render:
            self.buf_render_rgb = np.zeros([num_envs, self.VIDEORES, self.VIDEORES, 3], dtype=np.uint8)
        else:
            self.buf_render_rgb = np.zeros([1, 1, 1, 1], dtype=np.uint8)

        num_channels = 1 if Config.USE_BLACK_WHITE else 3
        obs_space = gym.spaces.Dict({'obs_frames': gym.spaces.Box(0, 255, shape=[self.RES_H, self.RES_W, num_channels], dtype=np.uint8),
                'p': gym.spaces.Box(low=0, high=64, shape=(2,)),
                'g': gym.spaces.Box(low=0, high=64, shape=(2,)),
            })

        super().__init__(
            num_envs=num_envs,
            observation_space=obs_space,
            action_space=gym.spaces.Discrete(self.NUM_ACTIONS),
            )
        self.handle = lib.vec_create(
            game_versions[game_type],
            self.num_envs,
            lump_n,
            self.hires_render,
            default_zoom)
        self.dummy_info = [{} for _ in range(num_envs)]

        # init Goal manager
        self.gms = [Goal(i) for i in range(num_envs)]
        
                


    def __del__(self):
        if hasattr(self, 'handle'):
            lib.vec_close(self.handle)
        self.handle = 0

    def close(self):
        lib.vec_close(self.handle)
        self.handle = 0

    def reset(self):
        print("Courier ignores resets")
        obs, _, _, _ = self.step_wait()
        return obs

    def get_images(self):
        if self.hires_render:
            return self.buf_render_rgb
        else:
            return self.buf_rgb

    def step_async(self, actions):
        assert actions.dtype in [np.int32, np.int64]
        actions = actions.astype(np.int32)
        lib.vec_step_async_discrete(self.handle, actions)

    def step_wait(self):
        self.buf_rew = np.zeros_like(self.buf_rew)
        self.buf_done = np.zeros_like(self.buf_done)

        lib.vec_wait(
            self.handle,
            self.buf_rgb,
            self.buf_render_rgb,
            self.buf_rew,
            self.buf_done)
        
        obs_frames = self.buf_rgb

        self.buf_walls = np.zeros([self.num_envs*self.RES_W*self.RES_H], dtype=np.float32)
        self.buf_ax = np.zeros([self.num_envs], dtype=np.float32)
        self.buf_ay = np.zeros([self.num_envs], dtype=np.float32)
        lib.vec_map_info(
            self.handle,
            self.buf_walls,
            self.buf_ax,
            self.buf_ay)
        walls = np.transpose(self.buf_walls.reshape(self.num_envs,self.RES_H,self.RES_W), (0, 2, 1))
        ax = self.buf_ax
        ay = self.buf_ay
        ps = np.zeros([self.num_envs, 2]) 
        gs = np.zeros([self.num_envs, 2]) 
        for i in range(self.num_envs):
            # self.dummy_info[i]['map_info']={'walls': walls[i], 
            #                                     'ax':ax[i], 
            #                                     'ay':ay[i]}
            p = np.array([ax[i], ay[i]])
            self.buf_rew[i] = self.gms[i].step(p, walls[i])
            self.gms[i].render()
            cv2.imshow('', cv2.resize(obs_frames[i],(300,300)))
            # print(p.astype(int), self.gms[i].current_goal)
            ps[i] = p
            gs[i] = self.gms[i].current_goal
            if self.buf_done[i]:
                self.gms[i].reset()
            
        if Config.USE_BLACK_WHITE:
            obs_frames = np.mean(obs_frames, axis=-1).astype(np.uint8)[...,None]

        obs = {'obs_frames': obs_frames,'p': ps,'g': gs} 
        return obs, self.buf_rew, self.buf_done, self.dummy_info

def make(env_id, num_envs, **kwargs):
    assert env_id in game_versions, 'cannot find environment "%s", maybe you mean one of %s' % (env_id, list(game_versions.keys()))
    return CourierVecEnv(env_id, num_envs, **kwargs)
