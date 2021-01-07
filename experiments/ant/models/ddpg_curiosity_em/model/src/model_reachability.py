import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, hidden_count = 256):
        super(Model, self).__init__()

        self.device = "cpu"


        self.layers = [ 
            nn.Linear(input_shape[0]*2, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU(),            
            nn.Linear(hidden_count//2, 1)           
        ] 

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.xavier_uniform_(self.layers[4].weight)
 
        self.model = nn.Sequential(*self.layers) 
        self.model.to(self.device)

        print("model_reachability")
        print(self.model)
        print("\n\n")
       

    def forward(self, state_a, state_b):
        x = torch.cat([state_a, state_b], dim = 1)
        return self.model(x)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "model_reachability.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "model_reachability.pt", map_location = self.device))
        self.model.eval()  
    
