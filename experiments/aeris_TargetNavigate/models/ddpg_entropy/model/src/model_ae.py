import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, kernels_count = 32):
        super(Model, self).__init__()

        self.device = "cpu"

        width = input_shape[1]

        self.layers_encoder = [ 
            nn.Conv1d(input_shape[0], kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),

            nn.Conv1d(kernels_count, kernels_count, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv1d(kernels_count, kernels_count//2, kernel_size=3, stride=1, padding=1)
        ]

        self.layers_decoder = [ 
            nn.Conv1d(kernels_count//2, kernels_count, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv1d(kernels_count, kernels_count, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(kernels_count, kernels_count, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.ReLU(),

            nn.Conv1d(kernels_count, input_shape[0], kernel_size=3, stride=1, padding=1)
        ]

        torch.nn.init.orthogonal_(self.layers_encoder[0].weight, 2**0.5)
        torch.nn.init.orthogonal_(self.layers_encoder[2].weight, 2**0.5)
        torch.nn.init.orthogonal_(self.layers_encoder[4].weight, 2**0.5)

        
        torch.nn.init.orthogonal_(self.layers_decoder[0].weight, 2**0.5)
        torch.nn.init.orthogonal_(self.layers_decoder[2].weight, 2**0.5)
        torch.nn.init.orthogonal_(self.layers_decoder[4].weight, 2**0.5)
        torch.nn.init.orthogonal_(self.layers_decoder[6].weight, 2**0.5)
        
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
        return self.model_decoder(features), features.view(len(state), -1)

    def eval_features(self, state):
        features = self.model_encoder(state)
        return features.view(len(state), -1)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model_encoder.state_dict(), path + "model_ae_encoder.pt")
        torch.save(self.model_decoder.state_dict(), path + "model_ae_decoder.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_encoder.load_state_dict(torch.load(path + "model_ae_encoder.pt", map_location = self.device))
        self.model_encoder.eval()  

        self.model_decoder.load_state_dict(torch.load(path + "model_ae_decoder.pt", map_location = self.device))
        self.model_decoder.eval()  
    



if __name__ == "__main__":
    batch_size      = 64
    input_shape     = (6, 128)

    model = Model(input_shape)

    state   = torch.randn((batch_size, ) + input_shape)

    y, features = model.forward(state)

    print(state.shape, features.shape, y.shape)
