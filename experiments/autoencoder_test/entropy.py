import RLAgents
import gym
import numpy
import cv2


def entropy(states_np, actions_np):    

    state_std   = numpy.std(states_np, axis=0)
    state_mean  = numpy.mean(states_np, axis=0)

    actions_std = numpy.std(actions_np, axis=0)
    
    print(state_std.mean(), actions_std.mean(), (state_std.mean()*actions_std.mean())**0.5 )

    height  = state.shape[1]
    width   = state.shape[2]
    im      = numpy.zeros((height, 3*width))

    x_variance = numpy.std(states_np, axis=0).mean()

    im[0:height, 0:width]           = states_np[0][0]
    im[0:height, width:2*width]     = state_mean[0]
    im[0:height, 2*width:3*width]   = state_std[0]

    im = cv2.resize(im, (2*im.shape[1], 2*im.shape[0]), interpolation = cv2.INTER_AREA)

    cv2.imshow("state", im)
    cv2.waitKey(1)

def init_buffer(size, actions_count, state):
    state_buffer = numpy.zeros( (size, ) + state.shape )
    actions_buffer = numpy.zeros( (size, actions_count))
    for i in range(size):
        state_buffer[i] = state
    

    return state_buffer, actions_buffer

def action_one_hot(action, actions_count):
    action_one_hot_np = numpy.zeros(actions_count)
    action_one_hot_np[action] = 1.0

    return action_one_hot_np

#env = gym.make("MsPacmanNoFrameskip-v4")
env = gym.make("MontezumaRevengeNoFrameskip-v4")
env = RLAgents.WrapperAtari(env)

state           = env.reset()
state_mean      = state.copy()

state_shape     = env.observation_space.shape
actions_count   = env.action_space.n


batch_size  = 256

buffer_state_np, buffer_actions_np = init_buffer(batch_size, actions_count, state)

action = 0
idx = 0
k = 0.001
while True:
    if numpy.random.rand() < 0.02:
        action = numpy.random.randint(actions_count)

    state, reward, done, info = env.step(action)

    if done:
        state = env.reset()
        buffer_state_np, buffer_actions_np = init_buffer(batch_size, actions_count, state)


    buffer_state_np[idx]    = state
    buffer_actions_np[idx]  = action_one_hot(action, actions_count)
    idx = (idx + 1)%batch_size

    if idx%batch_size == 0:
        env.render()
        entropy(buffer_state_np, buffer_actions_np)



