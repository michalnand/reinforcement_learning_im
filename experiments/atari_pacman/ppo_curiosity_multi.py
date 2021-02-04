import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *
from libs_common.MultiEnv import *

import models.ppo_curiosity_multi.src.model_ppo               as ModelPPO
import models.ppo_curiosity_multi.src.model_forward_0         as ModelForward0
import models.ppo_curiosity_multi.src.model_forward_target_0  as ModelForwardTarget0
import models.ppo_curiosity_multi.src.model_forward_1         as ModelForward1
import models.ppo_curiosity_multi.src.model_forward_target_1  as ModelForwardTarget1
import models.ppo_curiosity_multi.src.model_forward_2         as ModelForward2
import models.ppo_curiosity_multi.src.model_forward_target_2  as ModelForwardTarget2
import models.ppo_curiosity_multi.src.config                  as Config


path = "models/ppo_curiosity_multi/"

config  = Config.Config()
 
envs = MultiEnvSeq("MsPacmanNoFrameskip-v4", AtariWrapper, config.actors)

ModelsForward       = [ModelForward0, ModelForward1, ModelForward2]
ModelsForwardTarget = [ModelForwardTarget0, ModelForwardTarget1, ModelForwardTarget2]
 
agent = libs_agents.AgentPPOCuriosityMulti(envs, ModelPPO, ModelsForward, ModelsForwardTarget, Config)

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