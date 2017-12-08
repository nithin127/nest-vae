import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter

from datasets import DSprites, Reconstruction
from datasets.transforms import RandomMask

from models.dae_dsprites import DAE
from utils.torch_utils import to_var


parser = argparse.ArgumentParser(description='DAE')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='Input batch size for training (default: 100)')
parser.add_argument('--num-steps', type=int, default=200000, metavar='N',
                    help='Number training steps (default: 200000)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='Log interval, number of steps before save/image '
                    'in Tensorboard (default: 200)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training')
parser.add_argument('--seed', type=int, default=7691, metavar='S',
                    help='Random seed (default: 7691)')
parser.add_argument('--output-folder', type=str, default='dae',
                    help='Name of the output folder (default: dae)')
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
random_mask = transforms.Compose([
    RandomMask(),
    transforms.ToTensor()
])

dataset = DSprites(root='./data/dsprites', download=True)
dataset = Reconstruction(dataset=dataset, transform=random_mask,
    target_transform=transforms.ToTensor())

data_loader = torch.utils.data.DataLoader(dataset=dataset,
    batch_size=args.batch_size, shuffle=True)

# Model
model = DAE()
if args.cuda:
    model.cuda()
writer = SummaryWriter('./.logs/{0}'.format(args.output_folder))

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Fixed input for Tensorboard
fixed_x, _ = next(iter(data_loader))
fixed_grid = torchvision.utils.make_grid(fixed_x, normalize=True, scale_each=True)
writer.add_image('original', fixed_grid, 0)
fixed_x = to_var(fixed_x, args.cuda)

steps = 0
while steps < args.num_steps:
    for noisy_imgs, true_imgs in data_loader:
        noisy_imgs = to_var(noisy_imgs, args.cuda)
        true_imgs = to_var(true_imgs, args.cuda)
        logits = model(noisy_imgs)

        loss = F.mse_loss(F.sigmoid(logits), true_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss.data[0], steps)

        if (steps > 0) and (steps % args.log_interval == 0):
            # Save the reconstructed images
            logits = model(fixed_x)
            grid = torchvision.utils.make_grid(F.sigmoid(logits).data,
                normalize=True, scale_each=True)
            writer.add_image('reconstruction', grid, steps)

            # Save the checkpoint
            state = {
                'steps': steps,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss.data[0],
                'args': args
            }
            torch.save(state, './.saves/{0}/{0}_{1:d}.ckpt'.format(
                args.output_folder, steps))

        steps += 1
        if steps >= args.num_steps:
            break
