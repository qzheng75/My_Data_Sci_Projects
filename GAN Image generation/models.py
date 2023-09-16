import torch.nn as nn
import numpy as np


class GeneratorLayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalize=True, device='cpu'):
        super(GeneratorLayer, self).__init__()
        self.layer_list = [nn.Linear(in_feat, out_feat, device=device)]
        if normalize:
            self.layer_list.append(nn.BatchNorm1d(out_feat, device=device))
        self.layer_list.append(nn.LeakyReLU(0.2, inplace=True))

    def forward(self, X):
        for layer in self.layer_list:
            X = layer(X)
        return X


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, device='cpu'):
        """
        img_shape:(channels, height, width)
        """
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            GeneratorLayer(latent_dim, 128, normalize=False, device=device),
            GeneratorLayer(128, 256, device=device),
            GeneratorLayer(256, 512, device=device),
            GeneratorLayer(512, 1024, device=device),
            nn.Linear(1024, int(np.prod(img_shape)), device=device),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, X):
        img = self.layers(X)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, img_shape, device='cpu'):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512, device=device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256, device=device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1, device=device),
            nn.Sigmoid()
        )

    def forward(self, X):
        flattened_img = X.view(X.size(0), -1)
        return self.layers(flattened_img)
