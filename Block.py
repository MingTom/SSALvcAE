import torch.nn as nn
import torch
import torch.nn.functional as F

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            # m.weight.data.normal_(0, 0.2)
            if hasattr(m.bias, 'data'):
                m.bias.data.zero_()

class Clulayer(nn.Module):
    def __init__(self, common_dim, out_dim):
        super(Clulayer, self).__init__()
        Clu = [nn.Linear(common_dim, 1024), nn.BatchNorm1d(1024, affine=True), nn.LeakyReLU(0.2, inplace=True)]
        Clu += [nn.Linear(1024, 1024),  nn.BatchNorm1d(1024, affine=True), nn.LeakyReLU(0.2, inplace=True)]
        Clu += [nn.Linear(1024, out_dim), nn.BatchNorm1d(out_dim, affine=True)]
        self.Clu = torch.nn.Sequential(*Clu)
        initialize_weights(self)
    def forward(self,x, t=1):    # NUS，Cal：0.2  Yale,MSRC:1
        x = self.Clu(x)
        x = x / t
        x = nn.Softmax(dim=1)(x)
        return x

class MappingFromLatent(nn.Module):
    def __init__(self, z_dim, w_dim, common_dim):
        super(MappingFromLatent, self).__init__()
        F_layers = [nn.Linear(z_dim, 64), nn.LeakyReLU(0.2, inplace=True)]
        F_layers += [nn.Linear(64, 64), nn.LeakyReLU(0.2, inplace=True)]
        F_layers += [nn.Linear(64, 64), nn.LeakyReLU(0.2, inplace=True)]
        F_layers += [nn.Linear(64, 64), nn.LeakyReLU(0.2, inplace=True)]
        F_layers += [nn.Linear(64, 64), nn.LeakyReLU(0.2, inplace=True)]
        F_layers += [nn.Linear(64, w_dim), nn.LeakyReLU(0.2, inplace=True)]
        self.F = torch.nn.Sequential(*F_layers)
        initialize_weights(self)
        self.common_dim = common_dim
    def forward(self, x):
        x = self.F(x)
        return x, x[:, 0:self.common_dim]

class Generator(nn.Module):
    def __init__(self, w_dim, output_dim):
        super(Generator, self).__init__()

        G_layers = [nn.Linear(w_dim, 1024), nn.LeakyReLU(0.2, inplace=True)]
        G_layers += [nn.Linear(1024, 1024), nn.LeakyReLU(0.2, inplace=True)]
        G_layers += [nn.Linear(1024, 1024), nn.LeakyReLU(0.2, inplace=True)]
        G_layers += [nn.Linear(1024, output_dim), nn.Sigmoid()]
        self.G = torch.nn.Sequential(*G_layers)
        initialize_weights(self)
    def forward(self, x):
        x = self.G(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, w_dim, common_dim):
        super(Encoder, self).__init__()

        E_layers = [nn.Linear(input_dim, 1024), nn.LeakyReLU(0.2, inplace=True)]
        E_layers += [nn.Linear(1024, 1024), nn.LeakyReLU(0.2, inplace=True)]
        E_layers += [nn.Linear(1024, 1024), nn.LeakyReLU(0.2, inplace=True)]
        E_layers += [nn.Linear(1024, w_dim), nn.LeakyReLU(0.2, inplace=True)]
        self.E = torch.nn.Sequential(*E_layers)
        initialize_weights(self)
        self.common_dim = common_dim
    def forward(self, x):
        x = self.E(x)
        return x, x[:, 0:self.common_dim]

class Project(nn.Module):
    def __init__(self, layers_dim):
        super(Project, self).__init__()
        layers = []
        layers.append(nn.Linear(layers_dim[0], layers_dim[1], bias=False))
        layers.append(nn.BatchNorm1d(layers_dim[1]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Linear(layers_dim[1], layers_dim[2], bias=False))
        layers.append(nn.BatchNorm1d(layers_dim[2]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Linear(layers_dim[2], layers_dim[2], bias=False))
        layers.append(nn.BatchNorm1d(layers_dim[2], affine=False))
        self.projector = nn.Sequential(*layers)
        initialize_weights(self)
    def forward(self, w):
        z = self.projector(w)
        return z

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Discriminator(nn.Module):
    def __init__(self, w_dim, class_dim):
        super(Discriminator, self).__init__()
        D_layers = []
        D_layers += [nn.Linear(w_dim, 64), nn.LeakyReLU(0.2, inplace=True)]
        D_layers += [nn.Linear(64, 64), nn.LeakyReLU(0.2, inplace=True)]
        D_layers += [nn.Linear(64, 1), nn.LeakyReLU(0.2, inplace=True)]
        self.D = torch.nn.Sequential(*D_layers)
        initialize_weights(self)
    def forward(self, x):
        x = self.D(x)
        x = x.view(-1)
        return x

