import torch
import torch.nn as nn

class ModelSpatialBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(ModelSpatialBlock, self).__init__()

        self.conv0  = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act0   = nn.ELU()

        self.conv1  = nn.Conv2d(output_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.act1   = nn.ELU()
        self.conv2  = nn.Conv2d(output_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.act2   = nn.ELU()
        self.conv3  = nn.Conv2d(output_channels, output_channels, kernel_size=1, stride=1, padding=0)

        torch.nn.init.orthogonal_(self.conv0.weight, 2.0**0.5)
        torch.nn.init.orthogonal_(self.conv1.weight, 2.0**0.5)
        torch.nn.init.orthogonal_(self.conv2.weight, 2.0**0.5)
        torch.nn.init.orthogonal_(self.conv3.weight, 2.0**0.5)

    def forward(self, x):
        y = self.conv0(x)
        y = self.act0(y)

        s = self.conv1(y)
        s = self.act1(s)
        s = self.conv2(s)
        s = self.act2(s)
        s = self.conv3(s)

        return y, s


class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//12) * (input_shape[2]//12)

        self.model_s0 = ModelSpatialBlock(input_shape[0], 32, 8, 4, 0)
        self.model_s1 = ModelSpatialBlock(32, 64, 4, 2, 0)
        self.model_s2 = ModelSpatialBlock(64, 64, 3, 1, 0)
 
        self.layers_fc = [
            nn.Flatten(),

            nn.Linear(fc_size*64, 512),
            nn.ELU(),

            nn.Linear(512, 512),
            nn.ELU(),

            nn.Linear(512, 512)
        ]
        
        for i in range(len(self.layers_fc)):
            if hasattr(self.layers_fc[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_fc[i].weight, 2**0.5)
                
        self.model_s0.to(self.device)
        self.model_s1.to(self.device)
        self.model_s2.to(self.device)

        self.model_fc = nn.Sequential(*self.layers_fc)
        self.model_fc.to(self.device)

        print("model_forward_spatial")
        print(self.model_s0)
        print(self.model_s1)
        print(self.model_s2)
        print(self.model_fc)
        print("\n\n")

    def forward(self, state, action): 
        y0, s0 = self.model_s0(state)
        y1, s1 = self.model_s1(y0)
        y2, s2 = self.model_s2(y1)

        g = self.model_fc(y2)

        spatial = [s0, s1, s2]
        
        return spatial, g

    def save(self, path):
        torch.save(self.model_s0.state_dict(), path + "model_forward_s0.pt")
        torch.save(self.model_s1.state_dict(), path + "model_forward_s1.pt")
        torch.save(self.model_s2.state_dict(), path + "model_forward_s2.pt")

        torch.save(self.model_fc.state_dict(), path + "model_forward_fc.pt")

        
    def load(self, path):
        self.model_s0.load_state_dict(torch.load(path + "model_forward_s0.pt", map_location = self.device))
        self.model_s1.load_state_dict(torch.load(path + "model_forward_s1.pt", map_location = self.device))
        self.model_s2.load_state_dict(torch.load(path + "model_forward_s2.pt", map_location = self.device))

        self.model_fc.load_state_dict(torch.load(path + "model_forward_fc.pt", map_location = self.device))

        self.model_s0.eval()
        self.model_s1.eval()
        self.model_s2.eval()

        self.model_fc.eval() 


if __name__ == "__main__":
    batch_size = 8

    channels = 3
    height   = 96
    width    = 96

    actions_count = 9


    state           = torch.rand((batch_size, channels, height, width))
    action          = torch.rand((batch_size, actions_count))

    model = Model((channels, height, width), actions_count)

    spatial, g = model.forward(state, action)

    print(spatial[0].shape, spatial[0].max())
    print(spatial[1].shape, spatial[1].max())
    print(spatial[2].shape, spatial[2].max())
    print(g.shape, g.max())
