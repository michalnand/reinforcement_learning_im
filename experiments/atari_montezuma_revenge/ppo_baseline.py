import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.MontezumaWrapper import *
from libs_common.MultiEnv import *

import models.ppo_baseline.src.model            as Model
import models.ppo_baseline.src.config           as Config


path = "models/ppo_baseline/"

config  = Config.Config()

envs = MultiEnvParallel("MontezumaRevengeNoFrameskip-v4", MontezumaWrapper, config.actors)

agent = libs_agents.AgentPPO(envs, Model, Config)

max_iterations = config.actors*30000

trainig = TrainingIterations(envs, agent, max_iterations, path, 100)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    envs[0].render()
    time.sleep(0.01)

    if reward != 0:
        print(reward)
'''