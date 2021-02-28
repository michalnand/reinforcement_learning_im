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

        self.layers_mu = [
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU(),
            nn.Linear(hidden_count//2, outputs_count),
            nn.Tanh()
        ]

        self.layers_var = [
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU(), 
            nn.Linear(hidden_count//2, outputs_count),
            nn.Softplus()
        ]

        self.layers_value = [
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU(),
            nn.Linear(hidden_count//2, 1)
        ]


        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_mu = nn.Sequential(*self.layers_mu)
        self.model_mu.to(self.device)

        self.model_var = nn.Sequential(*self.layers_var)
        self.model_var.to(self.device)

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        print("model_ppo")
        print(self.model_features)
        print(self.model_mu)
        print(self.model_var)
        print(self.model_value)
        print("\n\n")
       

    def forward(self, state):
        features = self.model_features(state)

        mu      = self.model_mu(features)
        var     = self.model_var(features)
        value   = self.model_value(features)
        
        return mu, var, value

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model_features.state_dict(), path + "model_features.pt")
        torch.save(self.model_mu.state_dict(), path + "model_mu.pt")
        torch.save(self.model_var.state_dict(), path + "model_var.pt")
        torch.save(self.model_value.state_dict(), path + "model_value.pt")

    def load(self, path):       
        print("loading from ", path)
        #self.model.load_state_dict(torch.load(path + "model_actor.pt", map_location = self.device))
        #self.model.eval()  
    
