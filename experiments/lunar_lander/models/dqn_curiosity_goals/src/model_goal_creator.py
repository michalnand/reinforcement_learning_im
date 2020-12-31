import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape, sequence_length, kernels_count = 64, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers = [ 
            nn.Conv1d(input_shape[0], kernels_count, kernel_size=1, stride=1,padding=0),
            nn.ReLU(),   

            Flatten(),        

            nn.Linear(kernels_count*sequence_length, hidden_count),
            nn.ReLU(), 

            nn.Linear(hidden_count, input_shape[0])
        ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[3].weight)
        torch.nn.init.xavier_uniform_(self.layers[5].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_goal_creator")
        print(self.model)
        print("\n\n")
       
    def forward(self, states):
        x = states.transpose(1, 2)
        return self.model(x)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "model_goal_creator.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "model_goal_creator.pt", map_location = self.device))
        self.model.eval()  
    

if __name__ == "__main__":
    state_shape     = (10, )
    sequence_length = 8
    batch_size      = 32

    states          = torch.randn(( batch_size, sequence_length) + state_shape)

    model           = Model(state_shape, sequence_length)

    y               = model(states)

    print(">>> ", y.shape)