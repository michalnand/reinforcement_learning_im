import RLAgents
import gym
import numpy
import cv2

from inverse_model import *

def train_batch(model, state_t, state_next_t, action_t):
    action_predicted_t = model(state_t, state_next_t)

    loss = (action_t.detach() - action_predicted_t)**2
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    action_target_t     = torch.argmax(action_t, dim=1)
    action_predicted_t  = torch.argmax(action_predicted_t, dim=1)

    correct = (action_target_t == action_predicted_t).sum()
    wrong   = (action_target_t != action_predicted_t).sum()

    accuracy = 100.0*correct/(correct + wrong)

    print("loss = ", loss, accuracy)

def motivation(features):
    count       = features.shape[0]
    fa_dots     = (features*features).sum(axis=1).reshape((count,1))*numpy.ones(shape=(1,count))
    fb_dots     = (features*features).sum(axis=1)*numpy.ones(shape=(count,1))
    distances   = fa_dots + fb_dots - 2*features.dot(features.T)

    return numpy.mean(distances), numpy.std(distances)



def show(model, states_t):
    states_np = states_t.detach().to("cpu").numpy()
    z_t = model.eval_features(states_t)

    z_np = z_t.detach().to("cpu").numpy()

    z_mean, z_std = motivation(z_np)
    print("motivation = ", z_mean, z_std)


def action_one_hot(action, actions_count):
    action_one_hot_np = torch.zeros(actions_count)
    action_one_hot_np[action] = 1.0

    return action_one_hot_np


env = gym.make("MsPacmanNoFrameskip-v4")
#env = gym.make("SolarisNoFrameskip-v4")
env = RLAgents.WrapperAtari(env)

state = env.reset()

state_shape = env.observation_space.shape
actions_count = env.action_space.n
batch_size  = 64


model = Model(state_shape, actions_count) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


buffer_t      = torch.zeros( (batch_size, ) + state_shape ).to(model.device)
buffer_next_t = torch.zeros( (batch_size, ) + state_shape ).to(model.device)
action_t      = torch.zeros( (batch_size, actions_count)).to(model.device)

idx = 0
action = 0
while True:
    if numpy.random.rand() < 0.2:
        action = numpy.random.randint(actions_count)

    buffer_t[idx] = torch.from_numpy(state).to(model.device)
    
    state, reward, done, info = env.step(action)

    buffer_next_t[idx] = torch.from_numpy(state).to(model.device)
    action_t[idx] = action_one_hot(action, actions_count).to(model.device)

    if done:
        state = env.reset()
  
    idx = (idx + 1)%batch_size

    if idx == 0:
        train_batch(model, buffer_t, buffer_next_t, action_t)
        show(model, buffer_t)
        env.render()



