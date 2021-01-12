import torch
import torch.nn as nn


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"
  
        self.layers_features = [ 
            nn.Linear(input_shape[0], hidden_count),
            nn.ReLU()
        ] 

        self.layers_policy = [
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),                      
            nn.Linear(hidden_count, outputs_count)
        ]

        self.layers_value = [
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),                       
            nn.Linear(hidden_count, 1)    
        ]  

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)
        
        for i in range(len(self.layers_policy)):
            if hasattr(self.layers_policy[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_policy[i].weight)

        for i in range(len(self.layers_value)):
            if hasattr(self.layers_value[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_value[i].weight)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_policy = nn.Sequential(*self.layers_policy)
        self.model_policy.to(self.device)

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        
        print("model_ppo")
        print(self.model_features)
        print(self.model_policy)
        print(self.model_value)
        print("\n\n")


    def forward(self, state):
        features    = self.model_features(state)

        policy   = self.model_policy(features)
        value    = self.model_value(features)
        
        return policy, value

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_features.pt")
        torch.save(self.model_policy.state_dict(), path + "model_policy.pt")
        torch.save(self.model_value.state_dict(), path + "model_value.pt")
        

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_policy.load_state_dict(torch.load(path + "model_policy.pt", map_location = self.device))
        self.model_value.load_state_dict(torch.load(path + "model_value.pt", map_location = self.device))
        
        self.model_features.eval() 
        self.model_policy.eval() 
        self.model_value.eval() 