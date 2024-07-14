import torch
import numpy as np

default_quad_bias = [
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
]
default_plane_bias = [
    [0,0,1,0],
    [1,0,0,0],
    [0,1,0,0]
]

class PrsModel(torch.nn.Module):
    def __init__(self,input_nc=1, first_output_nc=4, grid_size=32, conv_num = 5, plane_num=3, quad_num=3, linear_num=3, plane_bias=default_plane_bias, quad_bias=default_quad_bias):
        super(PrsModel, self).__init__()
        self.input_nc = input_nc
        self.first_output_nc = first_output_nc
        self.plane_num = plane_num
        self.quad_num = quad_num
        self.grid_size = grid_size
        self.conv_num = conv_num
        model = []
        output_nc = first_output_nc
        grid_size = grid_size
        for _ in range(conv_num):
            model.append(torch.nn.Conv3d(input_nc, output_nc, kernel_size=3, stride=1, padding=1))
            # model.append(torch.nn.BatchNorm3d(first_output_nc))
            model.append(torch.nn.MaxPool3d(2))
            model.append(torch.nn.LeakyReLU())
            input_nc = output_nc
            output_nc *= 2
            grid_size = int(grid_size / 2)
        output_nc = int(output_nc/2) 
        
        model.append(torch.nn.Flatten())
        self.encoder = torch.nn.Sequential(*model)
        self.plane_models = torch.nn.ModuleList()
        for i in  range(plane_num):
            model = []
            input_nc = output_nc * grid_size
            for _ in range(linear_num - 1):
                model.append(torch.nn.Linear(input_nc, int(input_nc/2)))
                model.append(torch.nn.LeakyReLU())
                input_nc = int(input_nc/2)
            
            last = torch.nn.Linear(input_nc, 4)
            # last.weight.data = torch.zeros((4, input_nc)).float()
            last.bias.data = torch.tensor(plane_bias[i]).float()
            model.append(last)
            self.plane_models.append(torch.nn.Sequential(*model))
        
        self.quad_models = torch.nn.ModuleList()
        for i in range(quad_num):
            model = []
            input_nc = output_nc * grid_size
            for _ in range(linear_num - 1):
                model.append(torch.nn.Linear(input_nc, int(input_nc/2)))
                model.append(torch.nn.LeakyReLU())
                input_nc = int(input_nc/2)
            
            last = torch.nn.Linear(input_nc, 4)
            # last.weight.data = torch.zeros((4, input_nc))? why
            last.bias.data = torch.tensor(quad_bias[i]).float()
            model.append(last)
            self.quad_models.append(torch.nn.Sequential(*model))
            
    def forward(self, x):
        x = self.encoder(x)
        planes = []
        quads = []
        x = x.view(x.shape[0], 1, -1)
        for plane_model in self.plane_models:
            planes.append(plane_model(x))
        for quad_model in self.quad_models:
            quads.append(quad_model(x))
        return torch.stack(planes, dim=1), torch.stack(quads, dim=1)

if __name__ == '__main__':
    model = PrsModel()
    x = torch.randint(0,2,(8,1,32,32,32)).float()
    planes, quads = model(x)
    print(planes)
    print(quads)
    print('done')