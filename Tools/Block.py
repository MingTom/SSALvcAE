import torch.nn as nn
import torch

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
            m.bias.data.zero_()

class Clulayer(nn.Module):
    def __init__(self, common_dim, out_dim):
        super(Clulayer, self).__init__()
        Clu = [nn.Linear(common_dim, 512), nn.LeakyReLU(0.2, inplace=True)]
        Clu += [nn.Linear(512, 512), nn.LeakyReLU(0.2, inplace=True)]
        Clu += [nn.Linear(512, out_dim), nn.LeakyReLU(0.2, inplace=True)]
        self.Clu = torch.nn.Sequential(*Clu)
        initialize_weights(self)
    def forward(self,x):
        x = self.Clu(x)
        return x

class MappingFromLatent(nn.Module):
    def __init__(self, num_layers, input_dim, out_dim, common_dim, class_dim):
        super(MappingFromLatent, self).__init__()
        F_layers = [nn.Linear(input_dim+class_dim, 128), nn.LeakyReLU(0.2, inplace=True)]
        F_layers += [nn.Linear(128, 128), nn.LeakyReLU(0.2, inplace=True)]
        F_layers += [nn.Linear(128, 128), nn.LeakyReLU(0.2, inplace=True)]
        F_layers += [nn.Linear(128, 128), nn.LeakyReLU(0.2, inplace=True)]
        F_layers += [nn.Linear(128, out_dim), nn.LeakyReLU(0.2, inplace=True)]
        self.F = torch.nn.Sequential(*F_layers)
        self.embedding = torch.nn.Embedding(class_dim, class_dim)
        initialize_weights(self)
        self.common_dim = common_dim
        self.class_dim = class_dim
    def forward(self, x, label):
        if label == None:
            x = torch.cat((x, torch.zeros((x.shape[0], self.class_dim), device='cuda')), dim=1)
        else:
            x = torch.cat((x, self.embedding(label.long())), dim=1)
        x = self.F(x)
        return x, x[:, 0:self.common_dim]

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()

        G_layers = [nn.Linear(latent_dim, 1024), nn.LeakyReLU(0.2, inplace=True)]
        G_layers += [nn.Linear(1024, 1024), nn.LeakyReLU(0.2, inplace=True)]
        G_layers += [nn.Linear(1024, 1024), nn.LeakyReLU(0.2, inplace=True)]
        G_layers += [nn.Linear(1024, output_dim), nn.Sigmoid()]
        self.G = torch.nn.Sequential(*G_layers)
        initialize_weights(self)
    def forward(self, x):
        x = self.G(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, common_dim):
        super(Encoder, self).__init__()

        E_layers = [nn.Linear(input_dim, 1024), nn.LeakyReLU(0.2, inplace=True)]
        E_layers += [nn.Linear(1024, 1024), nn.LeakyReLU(0.2, inplace=True)]
        E_layers += [nn.Linear(1024, 1024), nn.LeakyReLU(0.2, inplace=True)]
        E_layers += [nn.Linear(1024, latent_dim), nn.LeakyReLU(0.2, inplace=True)]
        self.E = torch.nn.Sequential(*E_layers)
        initialize_weights(self)
        self.common_dim = common_dim
    def forward(self, x):
        x = self.E(x)
        return x, x[:, 0:self.common_dim]

class Discriminator(nn.Module):
    def __init__(self, num_layers, input_dim, class_dim):
        super(Discriminator, self).__init__()
        D_layers = []
        input_dim = input_dim + class_dim
        D_layers += [nn.Linear(input_dim, 128), nn.LeakyReLU(0.2, inplace=True)]
        D_layers += [nn.Linear(128, 128), nn.LeakyReLU(0.2, inplace=True)]
        D_layers += [nn.Linear(128, 128), nn.LeakyReLU(0.2, inplace=True)]
        D_layers += [nn.Linear(128, 1), nn.LeakyReLU(0.2, inplace=True)]
        self.D = torch.nn.Sequential(*D_layers)
        self.embedding = torch.nn.Embedding(class_dim, class_dim)
        initialize_weights(self)
        self.class_dim = class_dim
    def forward(self, x, label):
        if label == None:
            x = torch.cat((x, torch.zeros((x.shape[0], self.class_dim), device='cuda')), dim=1)
        else:
            x = torch.cat((x, self.embedding(label.long())), dim=1)
        x = self.D(x)
        x = x.view(-1)
        return x

