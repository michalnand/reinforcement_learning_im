import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_encoder = [ 
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        ] 

        
 
        self.layers_decoder = [ 
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.ELU(),
          
            nn.Conv2d(32, input_shape[0], kernel_size=1, stride=1, padding=0)
        ]  
  
        for i in range(len(self.layers_encoder)):
            if hasattr(self.layers_encoder[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_encoder[i].weight, 2**0.5)

        for i in range(len(self.layers_decoder)):
            if hasattr(self.layers_decoder[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_decoder[i].weight, 2**0.5)

        self.model_mu       = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.model_mu.to(self.device)

        self.model_logvar   = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.model_logvar.to(self.device)

        torch.nn.init.orthogonal_(self.model_mu[0].weight, 2**0.5)
        torch.nn.init.orthogonal_(self.model_logvar[0].weight, 2**0.5)
       
        self.model_encoder = nn.Sequential(*self.layers_encoder)
        self.model_encoder.to(self.device)

        self.model_decoder = nn.Sequential(*self.layers_decoder)
        self.model_decoder.to(self.device)

        self.model_encoder = nn.Sequential(*self.layers_encoder)
        self.model_encoder.to(self.device)

        print("model_autoencoder")
        print(self.model_encoder)
        print(self.model_mu)
        print(self.model_logvar)
        print(self.model_decoder)
        print("\n\n")

    def forward(self, x, noise_enabled = True):
        features    = self.model_encoder(x)
        mu          = self.model_mu(features)
        logvar      = self.model_logvar(features)
        
        if noise_enabled:
            z       = self._reparameterize(mu, logvar)
        else:
            z       = mu

        return self.model_decoder(z), mu, logvar

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def eval_features(self, x):
        features        = self.model_encoder(x)
        mu              = self.model_mu(features)
        return mu.view((mu.shape[0], -1))
       
    def save(self, path):
        print("saving ", path)

        torch.save(self.model_encoder.state_dict(), path + "model_ae_encoder.pt")
        torch.save(self.model_decoder.state_dict(), path + "model_ae_decoder.pt")

        torch.save(self.model_mu.state_dict(), path + "model_ae_mu.pt")
        torch.save(self.model_logvar.state_dict(), path + "model_ae_logvar.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_encoder.load_state_dict(torch.load(path + "model_ae_encoder.pt", map_location = self.device))
        self.model_decoder.load_state_dict(torch.load(path + "model_ae_decoder.pt", map_location = self.device))

        self.model_mu.load_state_dict(torch.load(path + "model_ae_mu.pt", map_location = self.device))
        self.model_logvar.load_state_dict(torch.load(path + "model_ae_logvar.pt", map_location = self.device))
        
        self.model_encoder.eval() 
        self.model_decoder.eval() 
        self.model_mu.eval()
        self.model_logvar.eval()
