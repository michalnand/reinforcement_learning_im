import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, latent_size = 16):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_encoder = [ 
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, latent_size, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        ]
 
        self.layers_decoder = [ 
            nn.ConvTranspose2d(latent_size, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
          
            nn.Conv2d(32, input_shape[0], kernel_size=1, stride=1, padding=0)
        ] 
  
        for i in range(len(self.layers_encoder)):
            if hasattr(self.layers_encoder[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_encoder[i].weight)

        for i in range(len(self.layers_decoder)):
            if hasattr(self.layers_decoder[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_decoder[i].weight)

       
        self.model_encoder = nn.Sequential(*self.layers_encoder)
        self.model_encoder.to(self.device)

        self.model_decoder = nn.Sequential(*self.layers_decoder)
        self.model_decoder.to(self.device)

        print("model_autoencoder")
        print(self.model_encoder)
        print(self.model_decoder)
        print("\n\n")

    def forward(self, state):
        features    = self.model_encoder(state)

        noise       = torch.randn(features.shape).to(features.device)
        f_noised    = features + 0.01*noise

        return self.model_decoder(f_noised), features


    def eval_features(self, state):
        return self.model_encoder(state)
       
    def save(self, path):
        print("saving ", path)

        torch.save(self.model_encoder.state_dict(), path + "model_ae_encoder.pt")
        torch.save(self.model_decoder.state_dict(), path + "model_ae_decoder.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_encoder.load_state_dict(torch.load(path + "model_ae_encoder.pt", map_location = self.device))
        self.model_decoder.load_state_dict(torch.load(path + "model_ae_decoder.pt", map_location = self.device))
        
        self.model_encoder.eval() 
        self.model_decoder.eval() 
