import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import numpy

class Model(torch.nn.Module):
    def __init__(self, state_size, actions_count, hidden_count = 256, hidden_layers_count = 1):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers = []
        self.layers.append(nn.Linear(state_size + actions_count, hidden_count))
        self.layers.append(nn.ReLU())

        for h in range(hidden_layers_count):
            self.layers.append(nn.Linear(hidden_count, hidden_count))
            self.layers.append(nn.ReLU()) 

        self.layers.append(nn.Linear(hidden_count, state_size))

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.normal_(self.layers[i].weight, std = 0.1)
        

    
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model")
        print(self.model)
        print("\n\n")
       
    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)
        return self.model(x)

def process_imagination(model, state, actions):
    count  = actions.shape[0]
    states = state.unsqueeze(0).repeat(count, 1)

    states_predicted = model(states, actions)

    y = torch.std(states_predicted, dim=0).mean()
    return y.detach().numpy()


state_size      = 32
actions_count   = 8

model = Model(state_size, actions_count)


batch_size = 64

action_stds      = []
states_stds      = []
states_comp_stds = []

for i in range(100):
    action_std = (i+1)/100

    state   = torch.randn(state_size)
    actions = action_std*torch.randn((batch_size, actions_count))

    a_std   = torch.std(actions, dim=0).mean().detach().numpy()

    y = process_imagination(model, state, actions)
    y_comp = y/a_std

    action_stds.append(action_std)
    states_stds.append(y)
    states_comp_stds.append(y_comp)


plt.plot(action_stds, states_stds, label = "raw entropy")
plt.plot(action_stds, states_comp_stds, label = "compensated entropy")
plt.legend()
plt.ylabel('imagined state entropy')
plt.xlabel('action entropy')
plt.show()