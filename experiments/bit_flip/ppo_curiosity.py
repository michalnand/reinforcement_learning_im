import gym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *

import models.ppo_curiosity.model.src.model_ppo                   as ModelPPO
import models.ppo_curiosity.model.src.model_forward               as ModelForward
import models.ppo_curiosity.model.src.model_forward_target        as ModelForwardTarget
import models.ppo_curiosity.model.src.config                      as Config

import bit_flip

path = "models/ppo_curiosity/model/"

config  = Config.Config()
envs    = []
for e in range(config.actors):
    env = bit_flip.BitFlip(size=10)
    envs.append(env)


agent = libs_agents.AgentPPOCuriosity(envs, ModelPPO, ModelForward, ModelForwardTarget, Config)


max_iterations = 200000
trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run() 
