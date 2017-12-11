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

from kernel import get_HSIC

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


# Get the latent representation for each example
fixed_z = vae.encode(fixed_x)

HSIC_array = get_HSIC(fixed_z.data.cpu().numpy())
print('The mean value of the HSIC_array is {}'.format(np.mean(HSIC_array)))

result_path = os.path.join('.logs', 'hsic_dependency', args.save_dir, '{}.txt'.format(args.dataset))

with open(result_path, 'w') as f:
    f.write('The HSIC array for {} is:\n'.format(HSIC_array))
    f.write('The mean value of the HSIC_array is {}'.format(np.mean(HSIC_array)))

