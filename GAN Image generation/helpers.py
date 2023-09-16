import torch
import yaml
import os
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from time import time
import numpy as np
import logging
from torchvision.utils import save_image


logging.basicConfig(level=logging.INFO)


def config_to_dict(config_path):
    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Dimension of latent space
    latent_dim = config['model'].get('latent_dim', 100)
    # Dimensionality of image
    img_size = config['model'].get('img_size', 28)
    # Number of image channels
    num_channels = config['model'].get('num_channels', 1)
    # interval between image samples
    sample_interval = config['model'].get('sample_interval', 400)

    n_epochs = config['trainer'].get('n_epochs', 100)
    batch_size = config['trainer'].get('batch_size', 128)
    device = config['trainer'].get('device', 'cpu')
    batch_verbosity = config['trainer'].get('batch_verbosity', 100)
    epoch_verbosity = config['trainer'].get('epoch_verbosity', 1)

    learning_rate = config['optimizer'].get('learning_rate', 1e-4)
    # b1, b2: betas for decay of first order momentum of gradient
    b1 = config['optimizer'].get('b1', 0.5)
    b2 = config['optimizer'].get('b2', 0.999)

    img_shape = (num_channels, img_size, img_size)

    return {
        'latent_dim': latent_dim,
        'img_shape': img_shape,
        'sample_interval': sample_interval,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'device': device,
        'batch_verbosity': batch_verbosity,
        'epoch_verbosity': epoch_verbosity,
        'learning_rate': learning_rate,
        'b1': b1,
        'b2': b2
    }


def load_mnist_dataset(config_dict, download_path='./data/mnist'):
    os.makedirs(download_path, exist_ok=True)
    return DataLoader(
        dataset=datasets.MNIST(
            download_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(config_dict['img_shape'][1]),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=config_dict['batch_size'],
        shuffle=True
    )


def train(epochs, dataloader, g_optimizer, d_optimizer, latent_dim,
          g_model, d_model, loss_fn, batch_verbosity, epoch_verbosity, sample_interval, device='cpu'):
    logging.info(f"Started training for {epochs} epochs on {device}.")

    g_model = g_model.to(device)
    d_model = d_model.to(device)
    tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    for epoch in range(epochs):
        g_losses = []
        d_losses = []
        epoch_start = time()
        for i, (imgs, _) in enumerate(dataloader):
            real = tensor(imgs.size(0), 1).fill_(1.0).to(device)
            fake = tensor(imgs.size(0), 1).fill_(0.).to(device)
            real_img = imgs.type(tensor).to(device)

            # Train the generator first
            # Input: sample noise
            z = tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))).to(device)
            
            gen_imgs = g_model(z)

            g_loss = loss_fn(d_model(gen_imgs), real)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Then train the discriminator
            real_loss = loss_fn(d_model(real_img), real)
            fake_loss = loss_fn(d_model(gen_imgs.detach()), fake)

            d_optimizer.zero_grad()
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            d_losses.append(d_loss.cpu().item())
            g_losses.append(g_loss.cpu().item())

            if batch_verbosity != 0 and i % batch_verbosity == 0:
                logging.info(f"{'-' * 25}")
                logging.info(f"Batch {i} for epoch {epoch + 1}:")
                logging.info(f"Discriminator loss: {d_loss:.4f}, Generator loss: {g_loss:.4f}\n")

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

        if epoch_verbosity != 0 and epoch % epoch_verbosity == 0:
            logging.info(f"{'-' * 25}")
            logging.info(f"Epoch {epoch + 1}/{epochs}: {(time() - epoch_start):.3f} seconds")
            logging.info(f"Average discriminator loss: {np.mean(d_losses):.4f}, Average generator loss: {np.mean(g_losses):.4f}\n")
