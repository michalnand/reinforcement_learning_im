import gym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *

import models.ppo_baseline.model.src.model    as Model
import models.ppo_baseline.model.src.config   as Config

import bit_flip

path = "models/ppo_baseline/model/"

config  = Config.Config()
envs    = []
for e in range(config.actors):
    env = bit_flip.BitFlip(size=10)
    envs.append(env)


agent = libs_agents.AgentPPO(envs, Model, Config)


max_iterations = 200000
trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    agent.main()
    envs[0].render()
    time.sleep(0.01)
'''