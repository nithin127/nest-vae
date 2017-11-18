import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

from tensorboardX import SummaryWriter

# MNIST dataset
dataset = datasets.FashionMNIST(root='./data/fashion-mnist',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
# dataset = datasets.MNIST(root='./data/mnist',
#                          train=True,
#                          transform=transforms.ToTensor(),
#                          download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=100, 
                                          shuffle=True)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.LeakyReLU(0.2))

        self.encoder_mean = nn.Linear(128, z_dim)
        self.encoder_logvar = nn.Sequential(
            nn.Linear(128, z_dim),
            nn.Softplus())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=5),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2),
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
        z = z.view(h.size(0), 20, 1, 1)
        logits = self.decoder(z)

        return logits, mu, log_var
    
    def sample(self, z):
        return self.decoder(z)
    
vae = VAE()

if torch.cuda.is_available():
    vae.cuda()
    
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
iter_per_epoch = len(data_loader)

writer = SummaryWriter('./.logs/vae')

# fixed inputs for debugging
fixed_x, _ = next(iter(data_loader))
fixed_grid = torchvision.utils.make_grid(fixed_x, normalize=True, scale_each=True)
writer.add_image('vae/original', fixed_grid, 0)
fixed_x = to_var(fixed_x)

for epoch in range(50):
    for i, (images, _) in enumerate(data_loader):
        
        images = to_var(images)
        logits, mu, log_var = vae(images)
        
        # Compute reconstruction loss and kl divergence
        # For kl_divergence, see Appendix B in the paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy_with_logits(logits, images, size_average=False)
        kl_divergence = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))
        
        # Backprop + Optimize
        total_loss = reconst_loss + kl_divergence
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        writer.add_scalar('loss', total_loss[0], epoch * iter_per_epoch + i)
        
        if i % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f" 
                   %(epoch + 1, 50, i + 1, iter_per_epoch, total_loss.data[0],
                     reconst_loss.data[0], kl_divergence.data[0]))
    
    # Save the reconstructed images
    reconst_logits, _, _ = vae(fixed_x)
    reconst_grid = torchvision.utils.make_grid(F.sigmoid(reconst_logits).data,
        normalize=True, scale_each=True)
    writer.add_image('vae/reconstruction', reconst_grid, epoch)

    # Save the checkpoint
    state = {
        'epoch': epoch + 1,
        'model': vae.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': {
            'total': total_loss.data[0],
            'reconstruction': reconst_loss.data[0],
            'kl_divergence': kl_divergence.data[0]
        }
    }
    if not os.path.exists('./.saves/vae/'):
        os.makedirs('./.saves/vae/')
    torch.save(state, './.saves/vae/vae_%d.ckpt' % (epoch + 1,))
