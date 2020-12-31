import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape, sequence_length, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"

        self.lstm   = nn.LSTM(input_size=input_shape[0], hidden_size=hidden_count, num_layers=1, batch_first=True, bidirectional=True)
        self.fc     = nn.Linear(2*hidden_count, input_shape[0])
        
        torch.nn.init.xavier_uniform_(self.fc.weight)

        self.lstm.to(self.device)
        self.fc.to(self.device)
        
        print("model_goal_creator")
        print(self.lstm)
        print(self.fc)
        print("\n\n")
       
    def forward(self, states):
        #state.shape = (batch, sequence, features)
        packed_output, (hidden, cell) = self.lstm(states)

        #concat the forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        #fc output layer
        return self.fc(hidden)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.lstm.state_dict(), path + "model_goal_creator_lstm.pt")
        torch.save(self.fc.state_dict(), path + "model_goal_creator_fc.pt")

    def load(self, path):       
        print("loading from ", path)
        self.lstm.load_state_dict(torch.load(path + "model_goal_creator_lstm.pt", map_location = self.device))
        self.lstm.eval()  

        self.fc.load_state_dict(torch.load(path + "model_goal_creator_fc.pt", map_location = self.device))
        self.fc.eval()  
    

if __name__ == "__main__":
    state_shape     = (10, )
    sequence_length = 8
    batch_size      = 64

    states          = torch.randn(( batch_size, sequence_length) + state_shape)

    model           = Model(state_shape, sequence_length)

    y               = model(states)

    print(">>> ", y.shape)
