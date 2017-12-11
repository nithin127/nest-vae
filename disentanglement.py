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
from datasets.celeba import CelebA
from torchvision import datasets, transforms
from models.vae_dsprites import VAE as VAE64

from utils.io_utils import get_latest_checkpoint
from utils.torch_utils import to_var

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Disentanglement')
parser.add_argument('--save-dir', type=str,
                    help='Path to the save folder (subfolder of `.saves` by default)')
parser.add_argument('--save-file', type=str, default=None, help='Checkpoint file')
parser.add_argument('--num-images', type=int, default=10, metavar='N',
                    help='Number of images (default: 10)')
parser.add_argument('--num-samples', type=int, default=11, metavar='N',
                    help='Number of samples (default: 11)')
parser.add_argument('--dataset', type=str, default='fashion-mnist',
                    help='Dataset to train the VAE on (default: fashion-mnist)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Data loading
if args.dataset == 'dsprites':
    dataset = DSprites(root='./data/dsprites',
        transform=transforms.ToTensor(), download=True)
    vae = VAE64(num_channels=1, zdim=10)
elif args.dataset == 'celeba':
    dataset = CelebA(root='./data/celeba',
        transform=transforms.ToTensor())
    vae = VAE64(num_channels=3, zdim=32)
    args.obs = 'normal'
else:
    raise ValueError('The `dataset` argument must be fashion-mnist, mnist, dsprites or celeba')

data_loader = torch.utils.data.DataLoader(dataset=dataset,
    batch_size=args.num_images, shuffle=True)

fixed_x, _ = next(iter(data_loader))
fixed_x = to_var(fixed_x, args.cuda, volatile=True)

if args.save_file is not None:
    filename = args.save_file
else:
    filename = get_latest_checkpoint(args.save_dir)

with open(filename, 'r') as f:
    if args.cuda:
        ckpt = torch.load(f)
    else:
        ckpt = torch.load(f, map_location=lambda storage, loc: storage)

if args.cuda:
    vae.cuda()
vae.load_state_dict(ckpt['model'])

# fixed_code = to_var(torch.from_numpy(np.random.rand(8, 1024).astype(np.float32)), args.cuda, volatile=True)
# print vae.encoder_ffwd(fixed_code)
# vae.train()
# print vae.encoder_ffwd(fixed_code)

writer = SummaryWriter(os.path.join('.logs', 'disentanglement', args.save_dir))

# Get the latent representation for each example
fixed_z = vae.encode(fixed_x)
reconst = vae.decoder_ffwd(fixed_z)
reconst = reconst.view(reconst.size(0), 64, 4, 4)
reconst = vae.decoder(reconst)
# reconst, _, _, _ = vae(fixed_x)
vae.eval()

writer.add_image('reconstruction/original', torchvision.utils.make_grid(fixed_x.data,
    nrow=4, normalize=True, scale_each=True))
writer.add_image('reconstruction/reconstruction',
    torchvision.utils.make_grid(F.sigmoid(reconst).data,
    nrow=4, normalize=True, scale_each=True))

# Get the interpolation deltas
interpolation = np.expand_dims(
    np.linspace(-3., 3., args.num_samples), axis=1)
eye = np.expand_dims(np.eye(vae.zdim), axis=1)
interpolations = (eye * interpolation).astype(np.float32)
interpolations = interpolations.reshape((-1, vae.zdim))
# interpolations = np.zeros_like(interpolations, dtype=np.float32)

# Loop over the examples to get an image per example
for i in range(args.num_images):
    latent_interpolation = fixed_z[i].cpu().data.numpy() + interpolations
    latent_interpolation = to_var(torch.from_numpy(latent_interpolation), args.cuda, volatile=True)
    latent_interpolation = vae.decoder_ffwd(latent_interpolation)
    latent_interpolation = latent_interpolation.view(latent_interpolation.size(0), 64, 4, 4)

    logits = vae.decoder(latent_interpolation)
    grid = torchvision.utils.make_grid(F.sigmoid(logits).data,
        nrow=args.num_samples, normalize=True, scale_each=True)

    writer.add_image('disentanglement/reconstruction', grid, i)

# Get images from prior noise
for i in range(args.num_images):
    latent_interpolation = np.random.randn(vae.zdim).astype(np.float32) + interpolations
    latent_interpolation = to_var(torch.from_numpy(latent_interpolation), args.cuda, volatile=True)
    latent_interpolation = vae.decoder_ffwd(latent_interpolation)
    latent_interpolation = latent_interpolation.view(latent_interpolation.size(0), 64, 4, 4)

    logits = vae.decoder(latent_interpolation)
    grid = torchvision.utils.make_grid(F.sigmoid(logits).data,
        nrow=args.num_samples, normalize=True, scale_each=True)

    writer.add_image('disentanglement/noise', grid, i)
