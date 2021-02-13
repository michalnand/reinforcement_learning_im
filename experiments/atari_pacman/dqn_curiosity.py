import gym
import numpy
import time

import RLAgents

import models.dqn_curiosity.src.model_dqn                   as ModelDQN
import models.dqn_curiosity.src.model_forward               as ModelForward
import models.dqn_curiosity.src.model_forward_target        as ModelForwardTarget
import models.dqn_curiosity.src.config                      as Config


path = "models/dqn_curiosity/"

env = gym.make("MsPacmanNoFrameskip-v4")

env = RLAgents.WrapperAtari(env)
env.reset()


agent = RLAgents.AgentDQNCuriosity(env, ModelDQN, ModelForward, ModelForwardTarget, Config)

max_iterations = 8*(10**6) 

trainig = RLAgents.TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(False)

    env.render()
    time.sleep(0.01)
'''