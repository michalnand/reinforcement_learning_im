import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_channels  = input_shape[0]
        input_height    = input_shape[1]
        input_width     = input_shape[2]

        fc_inputs_count = 64*(input_width//16)*(input_height//16)
  
        self.layers_features = [ 
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten()
        ] 

        self.layers_output = [
            nn.Linear(2*fc_inputs_count, 512),
            nn.ELU(),                       
            nn.Linear(512, 256),
            nn.ELU(),                       
            nn.Linear(256, outputs_count)    
        ] 

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)

        for i in range(len(self.layers_output)):
            if hasattr(self.layers_output[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_output[i].weight, 2**0.5)
       
        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_output = nn.Sequential(*self.layers_output)
        self.model_output.to(self.device)

        print("inverse_model")
        print(self.model_features)
        print(self.model_output)
        print("\n\n")

    def forward(self, state_t, state_next_t):
        features_state_t        = self.model_features(state_t)
        features_state_next_t   = self.model_features(state_next_t)
        
        x = torch.cat([features_state_t, features_state_next_t], dim=1)
        
        return self.model_output(x)

    def eval_features(self, state_t):
        return self.model_features(state_t)
       
    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_inverse_features.pt")
        torch.save(self.model_output.state_dict(), path + "model_inverse_output.pt")


    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_inverse_features.pt", map_location = self.device))
        self.model_output.load_state_dict(torch.load(path + "model_inverse_output.pt", map_location = self.device))

        self.model_features.eval() 
        self.model_output.eval() 
        