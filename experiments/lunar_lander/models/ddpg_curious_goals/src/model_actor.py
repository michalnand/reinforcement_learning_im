import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../..')
import libs_layers


class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers = [ 
            nn.Linear(input_shape[0]*2, hidden_count),
            nn.ReLU(),           
            libs_layers.NoisyLinearFull(hidden_count, hidden_count),
            nn.ReLU(),    
            libs_layers.NoisyLinearFull(hidden_count, outputs_count),
            nn.Tanh()
        ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.uniform_(self.layers[4].weight, -0.3, 0.3)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("\n\nmodel_actor")
        print(self.model)
       
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        return self.model(x)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_actor.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_actor.pt", map_location = self.device))
        self.model.eval()  
    
