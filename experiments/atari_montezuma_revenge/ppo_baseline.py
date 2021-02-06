import gym
import numpy
import time

import RLAgents

import models.ppo_baseline.src.model            as Model
import models.ppo_baseline.src.config           as Config


path = "models/ppo_baseline/"

config  = Config.Config()

envs = RLAgents.MultiEnvParallel("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, config.actors)

agent = RLAgents.AgentPPO(envs, Model, Config)

max_iterations = config.actors*30000

trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 100)
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