import sys
sys.path.insert(0, '..')
import gym
import numpy
import time

import libs_agents

import libs_common.atari_wrapper
import libs_common.MontezumaWrapper

from PIL import Image

import cv2

def show(x):
    image = cv2.resize(x, (400, 400), interpolation = cv2.INTER_AREA)
    cv2.imshow('state activity', image)
    cv2.waitKey(1)


#env = gym.make("BreakoutNoFrameskip-v4")
env = gym.make("MsPacmanNoFrameskip-v4")
env = libs_common.atari_wrapper.AtariWrapper(env)

#env = gym.make("MontezumaRevengeNoFrameskip-v4")
#env = libs_common.MontezumaWrapper.MontezumaWrapper(env)
obs = env.reset()

agent = libs_agents.AgentRandom(env)

states_running_mean_t = obs.copy()
states_running_std_t  = 0.1*numpy.ones(obs.shape)

#im = Image.fromarray(obs[0]*255.0)
#im.show()

k = 0.02
fps = 0

eps = 0.0001

steps = 0
while True:
    time_start = time.time()
    reward, done = agent.main()
    time_stop  = time.time()


    fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)

    mean_t = agent.state.mean(axis = 0)
    std_t  = agent.state.std(axis = 0)
    states_running_mean_t = (1.0 - eps)*states_running_mean_t + eps*mean_t
    states_running_std_t  = (1.0 - eps)*states_running_std_t  + eps*std_t

    if reward != 0:
        print("reward = ", reward)
    
    
    if done:
        print("FPS = ", round(fps, 1))
        print("DONE \n\n")

    steps+= 1

    if steps%10 == 0:
        tmp = (agent.state - states_running_mean_t) #/(states_running_std_t + 0.001)

        print(">>> ", tmp.min(), tmp.max(), tmp.mean(), tmp.std())

        tmp = (tmp - tmp.min())/(tmp.max() - tmp.min())
        show(tmp[0])

    
    #env.render()
    #time.sleep(0.01)
    