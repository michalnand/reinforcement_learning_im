import gym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *

import models.ppo_specter_past.src.model_ppo                   as ModelPPO
import models.ppo_specter_past.src.model_autoencoder           as ModelAutoencoder
import models.ppo_specter_past.src.config                      as Config

path = "models/ppo_specter_past/"

class Wrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward = reward / 10.0

        if reward < -1.0: 
            reward = -1.0

        if reward > 1.0:
            reward = 1.0

        return obs, reward, done, info

config  = Config.Config()
envs    = []
for e in range(config.actors):
    env = gym.make("LunarLander-v2")
    env = Wrapper(env)
    envs.append(env)

agent = libs_agents.AgentPPOSpecterPast(envs, ModelPPO, ModelAutoencoder, Config)

max_iterations = 100000
trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run() 
