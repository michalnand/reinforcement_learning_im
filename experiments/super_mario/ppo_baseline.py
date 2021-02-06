import gym
import numpy
import time

import RLAgents

import models.ppo_baseline.src.model            as Model
import models.ppo_baseline.src.config           as Config


path = "models/ppo_baseline/"

config  = Config.Config()

#envs = RLAgents.MultiEnvParallel("SuperMarioBros-v0", RLAgents.WrapperSuperMario, config.actors, envs_per_thread=16)
envs = RLAgents.MultiEnvSeq("SuperMarioBros-v0", RLAgents.WrapperSuperMario, config.actors)

agent = RLAgents.AgentPPO(envs, Model, Config)

max_iterations = 1*(10**6)  

#trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
#trainig.run() 


agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    envs.render(0)
    time.sleep(0.01)

