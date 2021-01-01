import gym
import pybullet_envs
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *


import models.ddpg_curiosity_goals.model.src.model_critic     as ModelCritic
import models.ddpg_curiosity_goals.model.src.model_actor      as ModelActor
import models.ddpg_curiosity_goals.model.src.model_forward    as ModelForward
import models.ddpg_curiosity_goals.model.src.model_forward_target    as ModelForwardTarget
import models.ddpg_curiosity_goals.model.src.model_goal_creator    as ModelForwardGoalCreator
import models.ddpg_curiosity_goals.model.src.config           as Config

path = "models/ddpg_curiosity_goals/model/"

env = pybullet_envs.make("HalfCheetahBulletEnv-v0")
#env.render()

agent = libs_agents.AgentDDPGCuriosityGoals(env, ModelCritic, ModelActor, ModelForward, ModelForwardTarget, ModelForwardGoalCreator, Config)

max_iterations = 6*(10**6)
trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()
    env.render()
    time.sleep(0.01)
'''