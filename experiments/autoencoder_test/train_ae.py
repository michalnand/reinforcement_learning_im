import RLAgents
import gym
import numpy
import cv2

from ae import *

def train_batch(model, state_t):
    state_predicted_t, _ = model(state_t)

    loss = (state_t.detach() - state_predicted_t)**2
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def show(model, states_t):
    states_np = states_t.detach().to("cpu").numpy()
    state_predicted_t, z_t = model(states_t, False)

    state_predicted_np = state_predicted_t.detach().to("cpu").numpy()

    height  = state.shape[1]
    width   = state.shape[2]
    im      = numpy.zeros((2*height, 2*width))


    im[0:height, 0:width]           = states_np[0][0]
    im[0:height, width:2*width]     = state_predicted_np[0][0]
    im[1*height:2*height, 0:width]  = states_np.mean(axis=0)[0]

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
batch_size  = 64


model = Model(state_shape) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


buffer_t = torch.zeros( (batch_size, ) + state_shape ).to(model.device)

idx = 0
k = 0.001
while True:
    action = 0 #numpy.random.randint(actions_count)

    state, reward, done, info = env.step(action)

    if done:
        state = env.reset()

    state_mean = (1.0 - k)*state_mean + k*state
    state_diff = state - state_mean

    buffer_t[idx] = torch.from_numpy(state).to(model.device)
    idx = (idx + 1)%batch_size

    if idx == 0:
        train_batch(model, buffer_t)
        show(model, buffer_t)



