import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *

import models.dqn_curiosity_em.src.model_dqn                   as ModelDQN
import models.dqn_curiosity_em.src.model_reachability          as ModelReachability
import models.dqn_curiosity_em.src.config                      as Config


path = "models/dqn_curiosity_em/"

env = gym.make("MsPacmanNoFrameskip-v4")

env = AtariWrapper(env)
env.reset()


agent = libs_agents.AgentDQNCuriosityEM(env, ModelDQN, ModelReachability, Config)

max_iterations = 6*(10**6) 

trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(False)

    env.render()
    time.sleep(0.01)
'''