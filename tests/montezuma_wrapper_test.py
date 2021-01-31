import sys
sys.path.insert(0, '..')
import gym
import numpy
import time

import libs_agents

import libs_common.MontezumaWrapper



env = gym.make("MontezumaRevengeNoFrameskip-v4")
env = libs_common.MontezumaWrapper.MontezumaWrapper(env)
env.reset()

agent = libs_agents.AgentRandom(env)


obs, _, _, _ = env.step(0)


k = 0.02
fps = 0

steps               = 0
non_zero_rewards    = 0
while True:
    time_start = time.time()
    reward, done = agent.main()
    time_stop  = time.time()

    fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)
    steps+=1

    if reward != 0:
        non_zero_rewards+= 1

        print("steps    = ", steps)
        print("non_zero_rewards    = ", non_zero_rewards)
        print("fps    = ", fps)
        print("reward = ", reward)
        print("episode= ", env.raw_episodes)
        print("raw_score_per_episode= ", env.raw_score_per_episode)
        print("\n\n") 

    
 