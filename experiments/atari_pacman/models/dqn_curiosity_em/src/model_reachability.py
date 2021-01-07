import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//16) * (input_shape[2]//16)
        
        self.layers = [
            nn.Conv2d(input_shape[0]*2, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  
        
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
             
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Flatten(), 

            nn.Linear(64*fc_size, 512),
            nn.ReLU(),                      
            nn.Linear(512, 1)
        ]
 
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_reachability")
        print(self.model)
        print("\n\n")

    def forward(self, state_a, state_b): 
        x = torch.cat([state_a, state_b], dim =1)
        return self.model(x)
      
    def save(self, path):
        torch.save(self.model.state_dict(), path + "model_reachability.pt")
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path + "model_reachability.pt", map_location = self.device))
        self.model.eval() 


if __name__ == "__main__":
    batch_size = 32

    channels = 3
    height   = 96
    width    = 96

    state           = torch.rand((batch_size, channels, height, width))
    state_ref       = torch.rand((batch_size, channels, height, width))

    model = Model((channels, height, width))

    state_predicted = model.forward(state, state_ref)

    print(state_predicted)


