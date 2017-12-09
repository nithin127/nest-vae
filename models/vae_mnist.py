import torch
import torch.nn as nn
from torch.autograd import Variable

def weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform(module.weight.data, gain=3.)
        if module.bias is not None:
            module.bias.data.fill_(0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform(module.weight.data, gain=3.)
        if module.bias is not None:
            module.bias.data.fill_(0)

def encoder_block(input_filters, output_filters, **kwargs):
    return (nn.Sequential(
        nn.Conv2d(input_filters, output_filters, **kwargs),
        nn.BatchNorm2d(output_filters),
        nn.LeakyReLU(0.2)))

def decoder_block(input_filters, output_filters, **kwargs):
    return (nn.Sequential(
        nn.ConvTranspose2d(input_filters, output_filters, **kwargs),
        nn.BatchNorm2d(output_filters),
        nn.LeakyReLU(0.2)))

def load_module(state_dict, module_name):
    return (dict((k.replace(module_name, ''), v) for k, v in state_dict.items()
            if k.startswith(module_name)))

class VAE(nn.Module):

    def __init__(self, num_channels=1, zdim=10):
        super(VAE, self).__init__()
        self.num_channels = num_channels
        self.zdim = zdim

        self.encoder = nn.Sequential(
            encoder_block(num_channels, 32, kernel_size=4, stride=2),
            encoder_block(32, 64, kernel_size=4, stride=2),
            encoder_block(64, 128, kernel_size=5))

        self.encoder_ffwd = nn.Linear(128, 2 * zdim)

        self.decoder = nn.Sequential(
            decoder_block(zdim, 128, kernel_size=3),
            decoder_block(128, 64, kernel_size=5),
            decoder_block(64, 32, kernel_size=4, padding=1, stride=2),
            nn.ConvTranspose2d(32, num_channels, kernel_size=4, stride=2, padding=1))

        self.apply(weights_init)

    def load(self, state_dict):
        encoder_state_dict = load_module(state_dict, 'encoder.')
        self.encoder.load_state_dict(encoder_state_dict)

        decoder_state_dict = load_module(state_dict, 'decoder.')
        self.decoder.load_state_dict(decoder_state_dict)

    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = Variable(mu.data.new(*mu.size()).normal_(), requires_grad=False)
        z = mu + eps * torch.exp(0.5 * log_var) # 0.5 to convert var to std

        return z

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), 1024)
        params = self.encoder_ffwd(h)
        mu, _ = torch.chunk(params, 2, dim=1)

        return mu

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        params = self.encoder_ffwd(h)
        mu, log_var = torch.chunk(params, 2, dim=1)

        z = self.reparametrize(mu, log_var)
        z = z.view(z.size(0), self.zdim, 1, 1)
        logits = self.decoder(z)

        return logits, mu, log_var, z
