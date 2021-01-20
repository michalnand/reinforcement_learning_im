import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, hidden_count = 128, lattent_size = 8):
        super(Model, self).__init__()

        self.device = "cpu"

        self.layers_encoder = [ 
            nn.Linear(input_shape[0], hidden_count),
            nn.Tanh(),
            nn.Linear(hidden_count, hidden_count//2),
            nn.Tanh(),
            nn.Linear(hidden_count//2, lattent_size)
        ] 

        self.layers_decoder = [ 
            nn.Linear(lattent_size, hidden_count//2),
            nn.Tanh(),         
            nn.Linear(hidden_count//2, hidden_count),
            nn.Tanh(),         
            nn.Linear(hidden_count, input_shape[0])           
        ] 

        torch.nn.init.xavier_uniform_(self.layers_encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_encoder[2].weight)
        torch.nn.init.xavier_uniform_(self.layers_encoder[4].weight)

        torch.nn.init.xavier_uniform_(self.layers_decoder[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_decoder[2].weight)
        torch.nn.init.xavier_uniform_(self.layers_decoder[4].weight)
 
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
        return self.model_decoder(features), features

    def eval_features(self, state):
        return self.model_encoder(state)

     
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
    
