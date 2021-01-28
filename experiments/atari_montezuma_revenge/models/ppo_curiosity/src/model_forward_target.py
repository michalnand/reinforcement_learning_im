import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//12) * (input_shape[2]//12)
        self.layers = [
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ELU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),

            nn.Flatten(),

            nn.Linear(64*fc_size, 512)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2.0**0.5)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_forward_target")
        print(self.model)
        print("\n\n")

    def forward(self, state, action):
        return self.model(state)

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


