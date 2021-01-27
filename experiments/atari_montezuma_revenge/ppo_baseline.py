import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *
from libs_common.MultiEnv import *

import models.ppo_baseline.src.model            as Model
import models.ppo_baseline.src.config           as Config


path = "models/ppo_baseline/"

config  = Config.Config()
envs    = []
for e in range(config.actors):
    env = gym.make("MontezumaRevengeNoFrameskip-v4")
    env = AtariWrapper(env)
    envs.append(env)

envs = MultiEnvParallel(envs)

agent = libs_agents.AgentPPO(envs, Model, Config)

max_iterations = 1000000

trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    envs[0].render()
    time.sleep(0.01)
'''