import torch
import torch.nn as nn

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
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten()
        ] 

        self.layers_value = [
            nn.Linear(fc_inputs_count, 512),
            nn.ELU(),                       
            nn.Linear(512, 1)    
        ]  

        self.layers_policy = [
            nn.Linear(fc_inputs_count, 512),
            nn.ELU(),                      
            nn.Linear(512, outputs_count)
        ]
 
  
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2.0**0.5)

        for i in range(len(self.layers_value)):
            if hasattr(self.layers_value[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_value[i].weight, 2.0**0.5)

        for i in range(len(self.layers_policy)):
            if hasattr(self.layers_policy[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_policy[i].weight, 2.0**0.5)


        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        self.model_policy = nn.Sequential(*self.layers_policy)
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


    def get_activity_map(self, state):
 
        state_t     = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
        features    = self.model_features(state_t)
        features    = features.reshape((1, 128, 6, 6))

        upsample = nn.Upsample(size=(self.input_shape[1], self.input_shape[2]), mode='bicubic')

        features = upsample(features).sum(dim = 1)

        result = features[0].to("cpu").detach().numpy()

        k = 1.0/(result.max() - result.min())
        q = 1.0 - k*result.max()
        result = k*result + q
        
        return result
