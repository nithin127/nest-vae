import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter

from datasets import DSprites, Reconstruction
from datasets.celeba import CelebA
from datasets.sampler import FactorSampler

from models.vae_dsprites import VAE as VAE64
from utils.torch_utils import to_var
from utils.io_utils import get_latest_checkpoint

parser = argparse.ArgumentParser(description='metric')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='Input batch size for training (default: 100)')
parser.add_argument('--batch-size2', type=int, default=50, metavar='N',
                    help='Input batch size for training (default: 50)')
parser.add_argument('--load', type=str, default=None,
                    help='Save folder for the model')
parser.add_argument('--metric', type=str, default=None,
                    help='Save folder for the metric')
parser.add_argument('--dataset', type=str, default='dsprites',
                    help='Dataset to train the VAE on (default: dsprites)')
parser.add_argument('--num-steps', type=int, default=50, metavar='N',
                    help='Number training steps (default: 50)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training')
parser.add_argument('--seed', type=int, default=7691, metavar='S',
                    help='Random seed (default: 7691)')
parser.add_argument('--output-folder', type=str, default='metric',
                    help='Name of the output folder (default: metric)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if 'SLURM_JOB_ID' in os.environ:
    args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
if not os.path.exists('./.saves/{0}'.format(args.output_folder)):
    os.makedirs('./.saves/{0}'.format(args.output_folder))

# Data loading
if args.dataset == 'dsprites':
    dataset = DSprites(root='./data/dsprites',
        transform=transforms.ToTensor(), download=True)
    batch_sampler = FactorSampler('data/dsprites/processed/factors.hdf5',
        batch_size=2 * args.batch_size2)
    vae = VAE64(num_channels=1, zdim=10)
elif args.dataset == 'celeba':
    dataset = CelebA(root='./data/celeba',
        transform=transforms.ToTensor())
    batch_sampler = FactorSampler('data/celeba/processed/factors.hdf5',
        batch_size=2 * args.batch_size2)
    vae = VAE64(num_channels=3, zdim=32)
else:
    raise ValueError('The `dataset` argument must be dsprites or celeba')

data_loader = torch.utils.data.DataLoader(dataset=dataset,
    batch_sampler=batch_sampler)

# Model
if args.cuda:
    vae.cuda()
with open(get_latest_checkpoint(args.load), 'r') as f:
    state_dict = torch.load(f)
    state_dict = state_dict['model']
vae.load_state_dict(state_dict)
vae.eval()

metric = nn.Linear(vae.zdim, batch_sampler.num_factors)
with open(get_latest_checkpoint(args.metric), 'r') as f:
    state_dict = torch.load(f)
    state_dict = state_dict['model']
metric.load_state_dict(state_dict)
metric.eval()

accs = []
for _ in range(args.num_steps):
    diffs, factors = [], []
    for images, targets in data_loader:
        images = to_var(images, args.cuda, volatile=True)
        latents = vae.encode(images)

        z1, z2 = torch.chunk(latents, 2, dim=0)
        diff = torch.mean(torch.abs(z1 - z2), dim=0)
        diffs.append(diff.data)

        targets_np = targets.numpy()
        common_factors = np.all(targets_np == targets_np[0], axis=0)
        p = common_factors.astype(np.float32) / np.sum(common_factors)
        factor = np.random.choice(len(common_factors), p=p)
        factors.append(factor)

        if len(diffs) == args.batch_size:
            diffs = to_var(torch.stack(diffs, dim=0), args.cuda)
            factors = to_var(torch.from_numpy(np.asarray(factors)).long(), args.cuda)
            logits = model(diffs)

            _, predictions = torch.max(logits, dim=1)
            correct_prediction = (predictions == factors)
            accuracy = correct_prediction.data.cpu().numpy().mean()
            accs.append(accuracy)

            break

print 'Mean'
print np.mean(accs)
