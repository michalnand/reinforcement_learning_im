import gym
import numpy
import time

import RLAgents

import models.ppo_baseline.src.model     as Model
import models.ppo_baseline.src.config    as Config

path = "models/ppo_baseline/"

config  = Config.Config()

class ScoreWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

       
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = reward/10.0

        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

def Wrapper(env):
    env = ScoreWrapper(env)
    
    return env


envs = RLAgents.MultiEnvSeq("LunarLanderContinuous-v2", Wrapper, config.actors)

agent = RLAgents.AgentPPOContinuous(envs, Model, Config)

max_iterations = 100000
trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 100)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()
    env.render()
    time.sleep(0.01)
'''