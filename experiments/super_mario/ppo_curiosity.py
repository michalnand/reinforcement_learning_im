import gym
import numpy
import time

import RLAgents

import models.ppo_curiosity.src.model_ppo               as ModelPPO
import models.ppo_curiosity.src.model_forward           as ModelForward
import models.ppo_curiosity.src.model_forward_target    as ModelForwardTarget
import models.ppo_curiosity.src.config                  as Config


path = "models/ppo_curiosity/"

config  = Config.Config()

envs = RLAgents.MultiEnvParallel("SuperMarioBros-v0", RLAgents.WrapperSuperMario, config.actors, envs_per_thread=4)
#envs = RLAgents.MultiEnvSeq("SuperMarioBros-v0", RLAgents.WrapperSuperMario, config.actors)

agent = RLAgents.AgentPPOCuriosity(envs, ModelPPO, ModelForward, ModelForwardTarget, Config)

max_iterations = 1*(10**6)  

trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    envs.render(0)
    time.sleep(0.01)
'''