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
        
        self.layers_features = [
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), 
        
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
             
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Flatten() 
        ]
 
        self.layers_output = [
            nn.Linear(2*64*fc_size, 256),
            nn.ReLU(),                      
            nn.Linear(256, 1)
        ]

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        for i in range(len(self.layers_output)):
            if hasattr(self.layers_output[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_output[i].weight)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_output = nn.Sequential(*self.layers_output)
        self.model_output.to(self.device)

        print("model_reachability")
        print(self.model_features)
        print(self.model_output)
        print("\n\n")

    def forward(self, state_a, state_b): 
        features_a      = self.model_features(state_a)
        features_b      = self.model_features(state_b)

        features        = torch.cat([features_a, features_b], dim=1)

        y = self.model_output(features)

        return y

    def save(self, path):
        torch.save(self.model_features.state_dict(), path + "model_reachability_features.pt")
        torch.save(self.model_output.state_dict(), path + "model_reachability_output.pt")
        
    def load(self, path):
        self.model_features.load_state_dict(torch.load(path + "model_reachability_features.pt", map_location = self.device))
        self.model_features.eval() 

        self.model_output.load_state_dict(torch.load(path + "model_reachability_output.pt", map_location = self.device))
        self.model_output.eval() 

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


