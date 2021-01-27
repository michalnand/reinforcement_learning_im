import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *
from libs_common.MultiEnv import *

import models.ppo_entropy.src.model_ppo             as ModelPPO
import models.ppo_entropy.src.model_forward         as ModelForward
import models.ppo_entropy.src.model_forward_target  as ModelForwardTarget
import models.ppo_entropy.src.model_autoencoder     as ModelAutoencoder
import models.ppo_entropy.src.config                as Config


path = "models/ppo_entropy/"

config  = Config.Config()
envs    = []
for e in range(config.actors):
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = AtariWrapper(env)
    envs.append(env)

envs = MultiEnvSeq(envs)

agent = libs_agents.AgentPPOEntropy(envs, ModelPPO, ModelForward, ModelForwardTarget, ModelAutoencoder, Config)

max_iterations = 1*(10**6) 

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