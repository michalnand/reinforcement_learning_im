import sys
sys.path.insert(0, '..')

import gym
import numpy
import torch
import torch.nn as nn

import libs_common.atari_wrapper
import libs_common.super_mario_wrapper

import cv2

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, weight_init_gain = 1.0):
        super(ResidualBlock, self).__init__()

        
        self.conv0  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act0   = nn.ReLU()
        self.conv1  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act1   = nn.ReLU()
            
        torch.nn.init.xavier_uniform_(self.conv0.weight, gain=weight_init_gain)
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=weight_init_gain)


    def forward(self, x):
        y  = self.conv0(x)
        y  = self.act0(y)
        y  = self.conv1(y)
        y  = self.act1(y + x)
        
        return y

class Model(torch.nn.Module):
    def __init__(self, input_shape, latent_size = 16):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_encoder = [ 
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            ResidualBlock(128),
            ResidualBlock(128),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            ResidualBlock(128), 
            ResidualBlock(128),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, latent_size, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        ]
 
        self.layers_decoder = [ 
            nn.ConvTranspose2d(latent_size, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
          
            nn.Conv2d(64, input_shape[0], kernel_size=3, stride=1, padding=1)
        ] 
  
        for i in range(len(self.layers_encoder)):
            if hasattr(self.layers_encoder[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_encoder[i].weight)

        for i in range(len(self.layers_decoder)):
            if hasattr(self.layers_decoder[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_decoder[i].weight)

       
        self.model_encoder = nn.Sequential(*self.layers_encoder)
        self.model_encoder.to(self.device)

        self.model_decoder = nn.Sequential(*self.layers_decoder)
        self.model_decoder.to(self.device)

        print("model_autoencoder")
        print(self.model_encoder)
        print(self.model_decoder)
        print("\n\n")

    def forward(self, state):
        features    = self.model_encoder(state)

        noise       = torch.randn(features.shape).to(features.device)
        f_noised    = features + 0.01*noise

        return self.model_decoder(f_noised), features


def cv_show(images_a, images_b, count):

    height = images_a.shape[2]
    width  = images_a.shape[3]

    result = numpy.zeros((2*height, count*width))

    for x in range(count):
        result[0*height:1*height, x*width:(x+1)*width]    = images_a[x][0]
        result[1*height:2*height, x*width:(x+1)*width]    = images_b[x][0]

    cv2.imshow("states", result)
    cv2.waitKey(1)

def kl_divergence(mu0, mu1, std0, std1):

    result = numpy.log(std1/(std0 + 0.00001))
    result+= (std0**2 + (mu0 - mu1)**2)/(2*(std1**2))
    result+= -0.5

    return result

def compute_distance(state):
    
    state_t = torch.from_numpy(state).unsqueeze(0).to(model.device)

    _, features_t = model(state_t)
    features = features_t.view(features_t.size(0), -1).squeeze(0).detach().to("cpu").numpy()

    distances_features = ((features - episodic_buffer)**2).mean(axis=1)
    mean = distances_features.mean()
    std  = distances_features.std()

    

    distances_features_buffer = numpy.zeros((episodic_buffer_size, episodic_buffer_size))
    for i in range(episodic_buffer_size):
        d = ((episodic_buffer - episodic_buffer[i])**2).mean(axis=1)
        distances_features_buffer[i] = d

    features_mean = distances_features_buffer.mean()
    features_std = distances_features_buffer.std()

    print(features_mean, features_std)
    print(mean, std)
    print(kl_divergence(features_mean, mean, features_std, std))
    print("\n\n\n")


    idx = numpy.random.randint(episodic_buffer_size)
    episodic_buffer[idx] = features

    return 0


#env = gym.make("MsPacmanNoFrameskip-v4")
#env = libs_common.atari_wrapper.AtariWrapper(env)

env = gym.make("SolarisNoFrameskip-v4")
env = libs_common.atari_wrapper.AtariWrapper(env)



state           = env.reset()
actions_count   = env.action_space.n

state_shape     = state.shape
batch_size      = 32
dataset_size    = 4096
episodic_buffer_size = 256

buffer_idx       = 0
states_buffer    = numpy.zeros((dataset_size, ) + state_shape, dtype=float)
episodic_buffer  = numpy.zeros((episodic_buffer_size, 8*6*6))

model           = Model(state_shape)
optimizer       = torch.optim.Adam(model.parameters(), lr=0.001)

steps = 0
for i in range(100000):
    action = numpy.random.randint(actions_count)
    state, reward, done, _ = env.step(action)

    if done:
        state = env.reset()

    steps+= 1

    states_buffer[buffer_idx] = state.copy()
    buffer_idx  = (buffer_idx + 1)%dataset_size
    
    
    if steps > dataset_size and steps%batch_size == 0:
        indices     = numpy.random.randint(0, dataset_size, size=batch_size)

        states_t    = torch.from_numpy(numpy.take(states_buffer, indices, axis=0)).to(model.device).float()

        states_predicted_t, features = model(states_t)

        loss = (states_t.detach() - states_predicted_t)**2
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_idx = 0

        states_np             = states_t.detach().to("cpu").numpy()
        states_predicted_np   = states_predicted_t.detach().to("cpu").numpy()

        cv_show(states_np, states_predicted_np, 8)

        state_t = torch.from_numpy(state).unsqueeze(0)

        

        '''
        features_np = features.view(features.size(0), -1).detach().to("cpu").numpy()
        
        distances_features = numpy.zeros((batch_size, batch_size))
        for i in range(batch_size):
            d = ((features_np - features_np[i])**2).mean(axis=1)
            distances_features[i]   = d


        states_np             = states_t.view(features.size(0), -1).detach().to("cpu").numpy()
        distances_states = numpy.zeros((batch_size, batch_size))
        for i in range(batch_size):
            d = ((states_np - states_np[i])**2).mean(axis=1)
            distances_states[i]   = d

        #print(numpy.round(distances_features, 3))
        #print(numpy.round(distances_states, 3))

        print("loss = ", loss)
        print("raw =      ", numpy.round(distances_states.mean(), 4), numpy.round(distances_states.std(), 4))
        print("features = ", numpy.round(distances_features.mean(), 4), numpy.round(distances_features.std(), 4))
        print("\n\n\n")
        '''
    '''
    if steps > dataset_size:
        compute_distance(state)
    '''