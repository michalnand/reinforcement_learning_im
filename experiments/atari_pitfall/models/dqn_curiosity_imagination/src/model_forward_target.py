import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//16) * (input_shape[2]//16)
        self.layers = [
            nn.Conv2d(input_shape[0] + outputs_count, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            Flatten(),

            nn.Linear(64*fc_size, 512)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)
                
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_forward_target")
        print(self.model)
        print("\n\n")

    def forward(self, state, action):
        height  = state.shape[2]
        width   = state.shape[3]
        action_ = action.unsqueeze(2).unsqueeze(2).repeat(1, 1, height, width)

        x = torch.cat([state, action_], dim=1).detach()

        return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), path + "model_forward_target.pt")
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path + "model_forward_target.pt", map_location = self.device))
        self.model.eval() 

if __name__ == "__main__":
    batch_size = 8

    channels = 3
    height   = 96
    width    = 96

    actions_count = 9


    state           = torch.rand((batch_size, channels, height, width))
    action          = torch.rand((batch_size, actions_count))

    model = Model((channels, height, width), actions_count)

    state_predicted = model.forward(state, action)

    print(state_predicted.shape)


