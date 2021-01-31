import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *
from libs_common.MultiEnv import *

import models.ppo_curiosity.src.model_ppo               as ModelPPO
import models.ppo_curiosity.src.model_forward           as ModelForward
import models.ppo_curiosity.src.model_forward_target    as ModelForwardTarget
import models.ppo_curiosity.src.config                  as Config


path = "models/ppo_curiosity/"

config  = Config.Config()

envs = MultiEnvSeq("MsPacmanNoFrameskip-v4", AtariWrapper, config.actors)

agent = libs_agents.AgentPPOCuriosity(envs, ModelPPO, ModelForward, ModelForwardTarget, Config)

max_iterations = 1*(10**6) 

trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    envs.render(0)
    time.sleep(0.01)
'''