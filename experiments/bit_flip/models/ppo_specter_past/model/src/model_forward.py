import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"
        
        channels    = input_shape[0]
        width       = input_shape[1]

        self.layers = [ 
            nn.Flatten(),
            nn.Linear((channels + outputs_count)*width, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),    
            nn.Linear(hidden_count, hidden_count//2)
        ]

        torch.nn.init.xavier_uniform_(self.layers[1].weight)
        torch.nn.init.xavier_uniform_(self.layers[3].weight)
        torch.nn.init.xavier_uniform_(self.layers[5].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_forward")
        print(self.model)
        print("\n\n")
       
    def forward(self, state, action):
        action_ = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x = torch.cat([state, action_], dim=1)
        return self.model(x)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "model_forward.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "model_forward.pt", map_location = self.device))
        self.model.eval()  
    
