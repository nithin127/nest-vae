import os
import argparse
import re
import glob
import numpy as np

# QKFIX: Add the parent path to PYTHONPATH
import sys
sys.path.insert(0, 'tristan')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from datasets.dsprites import DSprites
from torchvision import datasets, transforms

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Disentanglement')
parser.add_argument('--save-dir', type=str,
                    help='Path to the save folder (subfolder of `.saves` by default)')
parser.add_argument('--num-images', type=int, default=10, metavar='N',
                    help='Number of images (default: 10)')
parser.add_argument('--num-samples', type=int, default=11, metavar='N',
                    help='Number of samples (default: 11)')
parser.add_argument('--dataset', type=str, default='fashion-mnist',
                    help='Dataset to train the VAE on (default: fashion-mnist)')
args = parser.parse_args()

def to_var(x, **kwargs):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, **kwargs)

# Dataset
if args.dataset == 'fashion-mnist':
    dataset = datasets.FashionMNIST(root='./data/fashion-mnist',
        train=True, transform=transforms.ToTensor(), download=True)
elif args.dataset == 'mnist':
    dataset = datasets.MNIST(root='./data/mnist',
        train=True, transform=transforms.ToTensor(), download=True)
elif args.dataset == 'dsprites':
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    dataset = DSprites(root='./data/dsprites', transform=transform, download=True)
else:
    raise ValueError('The `dataset` argument must be fashion-mnist, mnist or dsprites')

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
    batch_size=args.num_images, shuffle=True)
fixed_x, _ = next(iter(data_loader))
fixed_x = to_var(fixed_x)

class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))

        self.encoder_mean = nn.Linear(128, z_dim)
        self.encoder_logvar = nn.Sequential(
            nn.Linear(128, z_dim),
            nn.Softplus())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, padding=1, stride=2))

    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(0.5 * log_var) # 0.5 to convert var to std
        return z

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu, log_var = self.encoder_mean(h), self.encoder_logvar(h)

        z = self.reparametrize(mu, log_var)
        z = z.view(h.size(0), self.z_dim, 1, 1)
        logits = self.decoder(z)

        return logits, mu, log_var

    def sample(self, z):
        return self.decoder(z)


if not os.path.exists(args.save_dir):
    raise ValueError('The saves directory `{0}` does '
        'not exist.'.format(args.save_dir))

checkpoints = glob.glob(os.path.join(args.save_dir, '*.ckpt'))
if not checkpoints:
    raise ValueError('There is no valid checkpoint '
        'in folder `{0}`.'.format(args.save_dir))

ckpt_number_re = re.compile(os.path.join(args.save_dir, r'.+_(\d+)\.ckpt'))
ckpt_numbers = map(lambda x: int(ckpt_number_re.match(x).group(1)), checkpoints)
ckpt_numbers = sorted(ckpt_numbers)

last_checkpoint = glob.glob(os.path.join(args.save_dir, '*_{0}.ckpt'.format(ckpt_numbers[-1])))[0]

with open(last_checkpoint, 'r') as f:
    if torch.cuda.is_available():
        ckpt = torch.load(f)
    else:
        ckpt = torch.load(f, map_location=lambda storage, loc: storage)

vae = VAE()
if torch.cuda.is_available():
    vae.cuda()
vae.load_state_dict(ckpt['model'])

writer = SummaryWriter(os.path.join('.logs', 'disentanglement', args.save_dir))

# Get the latent representation for each example
fixed_z = vae.encoder(fixed_x)
fixed_z = fixed_z.view(fixed_z.size(0), -1)
fixed_z = vae.encoder_mean(fixed_z)

# Get the interpolation deltas
interpolation = np.expand_dims(
    np.linspace(-1., 1., args.num_samples), axis=1)
eye = np.expand_dims(np.eye(vae.z_dim), axis=1)
interpolations = (eye * interpolation).astype(np.float32)

# Loop over the examples to get an image per example
for i in range(args.num_images):
    latent_interpolation = fixed_z[i].cpu().data.numpy() + interpolations
    latent_interpolation = to_var(torch.from_numpy(latent_interpolation))
    latent_interpolation = latent_interpolation.view(-1, vae.z_dim, 1, 1)

    logits = vae.decoder(latent_interpolation)
    grid = torchvision.utils.make_grid(F.sigmoid(logits).data,
        nrow=args.num_samples, normalize=True, scale_each=True)

    writer.add_image('disentanglement/reconstruction', grid, i)

# Get images from prior noise
for i in range(args.num_images):
    latent_interpolation = np.random.randn(20).astype(np.float32) + interpolations
    latent_interpolation = to_var(torch.from_numpy(latent_interpolation))
    latent_interpolation = latent_interpolation.view(-1, vae.z_dim, 1, 1)

    logits = vae.decoder(latent_interpolation)
    grid = torchvision.utils.make_grid(F.sigmoid(logits).data,
        nrow=args.num_samples, normalize=True, scale_each=True)

    writer.add_image('disentanglement/noise', grid, i)
