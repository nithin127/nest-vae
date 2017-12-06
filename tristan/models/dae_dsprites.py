import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def encoder_block(input_filters, output_filters,
          kernel_size=4, stride=2):
    return (nn.Sequential(
        nn.Conv2d(input_filters, output_filters,
                  kernel_size=kernel_size, stride=stride,
                  padding=1),
        nn.BatchNorm2d(output_filters),
        nn.ELU()))

def decoder_block(input_filters, output_filters,
          kernel_size=4, stride=2):
    return (nn.Sequential(
        nn.ConvTranspose2d(input_filters, output_filters,
                  kernel_size=kernel_size, stride=stride,
                  padding=1),
        nn.BatchNorm2d(output_filters),
        nn.ELU()))

class DAE(nn.Module):

    def __init__(self):
        super(DAE, self).__init__()
        
        self.encoder = nn.Sequential(
            encoder_block(1, 32),
            encoder_block(32, 32),
            encoder_block(32, 64),
            encoder_block(64, 64))

        self.bottleneck = nn.Sequential(
            nn.Linear(1024, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Linear(100, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU())

        self.decoder = nn.Sequential(
            decoder_block(64, 64),
            decoder_block(64, 32),
            decoder_block(32, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), 1024)
        z = self.bottleneck(z)
        z = z.view(z.size(0), 64, 4, 4)
        z = self.decoder(z)

        return z

if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'tristan')

    from transforms import RandomMask
    from datasets.dsprites import DSprites
    from datasets.reconstruction import Reconstruction

    import torchvision
    from torchvision import transforms
    from torch.autograd import Variable

    transform = transforms.Compose([
        RandomMask(),
        transforms.ToTensor()
    ])
    dataset = DSprites(root='./data/dsprites', download=True)
    dataset = Reconstruction(dataset=dataset, transform=transform,
        target_transform=transforms.ToTensor())

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_size=10, shuffle=True)

    model = DAE()

    noisy_images, images = next(iter(data_loader))

    # images = Variable(images)
    # code = model(images)

    # print code.size()