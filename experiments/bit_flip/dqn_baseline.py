import gym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *

import models.dqn_baseline.src.model    as Model
import models.dqn_baseline.src.config   as Config

import chart_move

path = "models/dqn_baseline/"

env = chart_move.ChartMove(size=16)
env.reset()

agent = libs_agents.AgentDQN(env, Model, Config)

max_iterations = 100000
trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
trainig.run() 
