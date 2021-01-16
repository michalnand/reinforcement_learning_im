import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers_encoder = [ 
            nn.Linear(input_shape[0], hidden_count),
            nn.ReLU(),           
            nn.Linear(hidden_count, input_shape[0]//2),
            nn.ReLU()
        ]

        self.layers_decoder = [ 
            nn.Linear(input_shape[0]//2, hidden_count),
            nn.ReLU(),           
            nn.Linear(hidden_count, input_shape[0]),
        ]

        self.model_encoder = nn.Sequential(*self.layers_encoder)
        self.model_encoder.to(self.device)

        self.model_decoder = nn.Sequential(*self.layers_decoder)
        self.model_decoder.to(self.device)

        print("model_autoencoder")
        print(self.model_encoder)
        print(self.model_decoder)
        print("\n\n")
       
    def forward(self, state):
        features = self.model_encoder(state)
        features_noised = features + 0.1*torch.randn(features.shape).to(features.device)
        return self.model_decoder(features_noised), features

    def eval_features(self, state):
        return self.model_encoder(state)
     
    def save(self, path):
        print("saving to ", path)

        torch.save(self.model_encoder.state_dict(), path + "model_autoecnoder_encoder.pt")
        torch.save(self.model_decoder.state_dict(), path + "model_autoecnoder_decoder.pt")

    def load(self, path):       
        print("loading from ", path)

        self.model_encoder.load_state_dict(torch.load(path + "model_autoecnoder_encoder.pt", map_location = self.device))
        self.model_decoder.load_state_dict(torch.load(path + "model_autoecnoder_decoder.pt", map_location = self.device))

        self.model_encoder.eval() 
        self.model_decoder.eval()  
    
