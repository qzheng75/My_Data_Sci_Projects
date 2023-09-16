import argparse
import os
import logging
import helpers
import torch
import models


logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config.yml')
    opt = parser.parse_args()
    

    logging.info(f"Using config from {opt.config_path}")
    config_dict = helpers.config_to_dict(config_path=opt.config_path)
    
    device = config_dict['device']

    loss_fn = torch.nn.BCELoss()
    generator = models.Generator(latent_dim=config_dict['latent_dim'],
                                 img_shape=config_dict['img_shape'], device=device)
    discriminator = models.Discriminator(img_shape=config_dict['img_shape'], device=device)
    dataloader = helpers.load_mnist_dataset(config_dict)

    b1 = config_dict['b1']
    b2 = config_dict['b2']
    lr = config_dict['learning_rate']
    g_optim = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    helpers.train(epochs=config_dict['n_epochs'],
                  dataloader=dataloader,
                  g_optimizer=g_optim,
                  d_optimizer=d_optim,
                  latent_dim=config_dict['latent_dim'],
                  g_model=generator,
                  d_model=discriminator,
                  loss_fn=loss_fn,
                  batch_verbosity=config_dict['batch_verbosity'],
                  epoch_verbosity=config_dict['epoch_verbosity'],
                  sample_interval=config_dict['sample_interval'],
                  device=device
                  )
