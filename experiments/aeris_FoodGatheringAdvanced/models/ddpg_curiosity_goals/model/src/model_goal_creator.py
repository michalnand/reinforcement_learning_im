import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, weight_init_gain = 1.0):
        super(ResidualBlock, self).__init__()

        
        self.conv0  = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act0   = nn.ReLU()
        self.conv1  = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act1   = nn.ReLU()
            
        torch.nn.init.xavier_uniform_(self.conv0.weight, gain=weight_init_gain)
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=weight_init_gain)


    def forward(self, x):
        y  = self.conv0(x)
        y  = self.act0(y)
        y  = self.conv1(y)
        y  = self.act1(y + x)
        
        return y


class Model(torch.nn.Module):
    def __init__(self, input_shape, sequence_length, kernels_count = 32, hidden_count = 256):
        super(Model, self).__init__()

        self.device = "cpu"
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape= input_shape
        self.channels   = input_shape[0]
        self.width      = input_shape[1]

        fc_count        = sequence_length*kernels_count*self.width//4

        self.layers_head = [
            nn.Conv1d(self.channels, kernels_count, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ]

        self.layers_output = [   
            nn.Conv1d(sequence_length*kernels_count, kernels_count, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            ResidualBlock(kernels_count),
            ResidualBlock(kernels_count),
            ResidualBlock(kernels_count),
            ResidualBlock(kernels_count),

            nn.Conv1d(kernels_count, self.channels, kernel_size=1, stride=1, padding=0)
        ]

        torch.nn.init.xavier_uniform_(self.layers_head[0].weight)

        torch.nn.init.xavier_uniform_(self.layers_output[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_output[6].weight)

        self.model_head = nn.Sequential(*self.layers_head)
        self.model_head.to(self.device)

        self.model_output = nn.Sequential(*self.layers_output)
        self.model_output.to(self.device)

        print("model_goal_creator")
        print(self.model_head)
        print(self.model_output)
        print("\n\n")

    def forward(self, states): 

        heads_in   = states.reshape( (states.shape[0]*states.shape[1], ) + self.input_shape )
        
        heads_out  = self.model_head(heads_in)

        heads_out  = heads_out.reshape( (states.shape[0], states.shape[1]*heads_out.shape[1], heads_out.shape[2]) )

        return self.model_output(heads_out)


    def save(self, path):
        torch.save(self.model_output.state_dict(), path + "model_goal_creator_output.pt")
        torch.save(self.model_head.state_dict(), path + "model_goal_creator_head.pt")
         
    def load(self, path):
        self.model_output.load_state_dict(torch.load(path + "model_goal_creator_output.pt", map_location = self.device))
        self.model_output.eval() 
 
        self.model_head.load_state_dict(torch.load(path + "model_goal_creator_head.pt", map_location = self.device))
        self.model_head.eval() 



if __name__ == "__main__":
    batch_size      = 64
    sequence_length = 32
    input_shape     = (6, 32)

    model = Model(input_shape, sequence_length)

    state           = torch.rand((batch_size, sequence_length, ) + input_shape)

    y = model.forward(state)

    print(y.shape)
