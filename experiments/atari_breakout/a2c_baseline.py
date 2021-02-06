import gym
import numpy
import time

import RLAgents

import models.a2c_baseline.src.model            as Model
import models.a2c_baseline.src.config           as Config


path = "models/a2c_baseline/"

config  = Config.Config()

envs = RLAgents.MultiEnvParallel("BreakoutNoFrameskip-v4", RLAgents.WrapperAtari, config.actors)

agent = RLAgents.AgentA2C(envs, Model, Config)

max_iterations = 125000

trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 100)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    envs.render(0)
    time.sleep(0.01)
'''