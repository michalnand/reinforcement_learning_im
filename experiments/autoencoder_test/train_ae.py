import RLAgents
import gym
import numpy
import cv2
import torch

from models import model_ae


class EpisodicMemory:
    def __init__(self, size, initial_count = 16):
        self.size               = size
        self.initial_count      = initial_count
        self.episodic_memory    = None

    def reset(self, state_t): 
        self.episodic_memory = torch.zeros((self.size , ) + state_t.shape).to(state_t.device)
        for i in range(self.size):
            self.episodic_memory[i] = state_t
        self.count = 0

    def add(self, state_t):
        if self.episodic_memory is None:
            self.reset(state_t)
        else:
            if self.count < self.initial_count: 
                n = self.size//self.initial_count 
                for i in range(n):
                    idx = numpy.random.randint(self.size)
                    self.episodic_memory[idx] = state_t

                self.count+= 1
            else:
                idx = numpy.random.randint(self.size)
                self.episodic_memory[idx] = state_t

        

    def entropy(self):
        mean = self.episodic_memory.mean(axis=0)
        diff = (self.episodic_memory - mean)**2
        max_ = diff.max(axis=0)[0] 
 
        result = max_.mean().detach().to("cpu").numpy()

        if self.count < self.initial_count:
            return 0.0
        else:
            return result

def train_batch(model, state_t):
    state_predicted_t, _ = model(state_t)

    loss = (state_t.detach() - state_predicted_t)**2
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def eval_features(model, state_np):
    states_t = torch.from_numpy(state_diff).unsqueeze(0).to(model.device)
    state_predicted_t, z_t = model(states_t, False)

    state_predicted_np = state_predicted_t.squeeze(0).detach().to("cpu").numpy()

    return state_predicted_np, z_t


def show(model, state_np):
    state_diff = state_np - state_mean
    state_predicted_np, z_t = eval_features(model, state_diff)

    episodic_memory.add(z_t)

    print("entropy = ", episodic_memory.entropy())

    height  = state.shape[1]
    width   = state.shape[2]
    height_ = height
    width_  = 3*width
    im      = numpy.zeros((height_, width_))


    im[0:height, 0*width:1*width]  = state_np[0]
    im[0:height, 1*width:2*width]  = state_diff[0]
    im[0:height, 2*width:3*width]  = state_predicted_np[0]

    im = cv2.resize(im, (2*width_, 2*height_), interpolation = cv2.INTER_AREA)

    cv2.imshow("state", im)
    cv2.waitKey(1)



env = gym.make("MsPacmanNoFrameskip-v4")
#env = gym.make("MontezumaRevengeNoFrameskip-v4")
env = RLAgents.WrapperAtari(env)

state = env.reset()

for i in range(10):
    state, _, _, _ = env.step(0)

state_mean = state.copy()

state_shape = env.observation_space.shape
actions_count = env.action_space.n
batch_size  = 16


model = model_ae.Model(state_shape) 
model.load("./models/")

episodic_memory = EpisodicMemory(256, 16)

idx = 0
k = 0.0001
while True:
    action = numpy.random.randint(actions_count)

    state, reward, done, info = env.step(action)

    state_mean = (1.0 - k)*state_mean + k*state
    state_diff = state - state_mean
    
    if done:
        state = env.reset()
        _, z_t = eval_features(model, state_diff)
        episodic_memory.reset(z_t)

    
    show(model, state)



