import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *

import models.dqn_curious_goals.src.model_dqn         as ModelDQN
import models.dqn_curious_goals.src.model_forward     as ModelForward
import models.dqn_curious_goals.src.config            as Config


path = "models/dqn_curious_goals/"

env = gym.make("MsPacmanNoFrameskip-v4")

env = AtariWrapper(env)
env.reset()


agent = libs_agents.AgentDQNCuriousGoals(env, ModelDQN, ModelForward, Config)

max_iterations = 6*(10**6) 

trainig = TrainingIterations(env, agent, max_iterations, path, 64) #10000)
trainig.run() 

'''
#agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(False)

    env.render()
    time.sleep(0.01)
'''