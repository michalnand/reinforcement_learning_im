import torch
import torch.nn as nn

class OutputHead(torch.nn.Module):

    def __init__(self, inputs_count, outputs_count):
        super(OutputHead, self).__init__()
        
        self.fc0    = nn.Linear(inputs_count, inputs_count)
        self.act0   = nn.ReLU()                   
        self.fc1    = nn.Linear(inputs_count, outputs_count)  

        nn.init.orthogonal_(self.fc0.weight, 0.1**0.5)
        self.fc0.bias.data.zero_()  

        nn.init.orthogonal_(self.fc1.weight, 0.01**0.5)
        self.fc1.bias.data.zero_()  

    def forward(self, x):
        y = self.fc0(x)
        y = self.act0(y)
        y = self.fc1(x + y)

        return y






class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    

        fc_inputs_count = 128*(input_width//16)*(input_height//16)
  
        self.layers_features = [ 
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten(),

            nn.Linear(fc_inputs_count, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        ] 
  
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                self.layers_features[i].bias.data.zero_()

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_value = OutputHead(512, 1)
        self.model_value.to(self.device)

        self.model_policy = OutputHead(512, self.outputs_count)
        self.model_policy.to(self.device)

        print("model_ppo")
        print(self.model_features)
        print(self.model_value)
        print(self.model_policy)
        print("\n\n")


    def forward(self, state):
        features    = self.model_features(state)

        value    = self.model_value(features)
        policy   = self.model_policy(features)

        return policy, value

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_features.pt")
        torch.save(self.model_value.state_dict(), path + "model_value.pt")
        torch.save(self.model_policy.state_dict(), path + "model_policy.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_value.load_state_dict(torch.load(path + "model_value.pt", map_location = self.device))
        self.model_policy.load_state_dict(torch.load(path + "model_policy.pt", map_location = self.device))
        
        self.model_features.eval() 
        self.model_value.eval() 
        self.model_policy.eval() 
