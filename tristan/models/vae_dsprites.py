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

def load_module(state_dict, module_name):
    return (dict((k.replace(module_name, ''), v) for k, v in state_dict.items()
            if k.startswith(module_name)))

class VAE(nn.Module):

    def __init__(self, num_channels=1, zdim=10):
        super(VAE, self).__init__()
        self.num_channels = num_channels
        self.zdim = zdim
        
        self.encoder = nn.Sequential(
            encoder_block(num_channels, 32),
            encoder_block(32, 32),
            encoder_block(32, 64),
            encoder_block(64, 64))

        self.encoder_ffwd = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 2 * zdim))

        self.decoder_ffwd = nn.Sequential(
            nn.Linear(zdim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU())

        self.decoder = nn.Sequential(
            decoder_block(64, 64),
            decoder_block(64, 32),
            decoder_block(32, 32),
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
        h = h.view(h.size(0), 1024)
        params = self.encoder_ffwd(h)
        mu, log_var = torch.chunk(params, 2, dim=1)

        z = self.reparametrize(mu, log_var)
        z = self.decoder_ffwd(z)
        z = z.view(z.size(0), 64, 4, 4)
        logits = self.decoder(z)

        return logits, mu, log_var

class VAEFlatten(nn.Module):

    def __init__(self, num_channels=1, zdim=10):
        super(VAEFlatten, self).__init__()
        self.num_channels = num_channels
        self.zdim = zdim

        self.encoder = nn.Sequential(
            nn.Linear(4096 * num_channels, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Linear(1200, 2 * zdim))
        
        self.decoder = nn.Sequential(
            nn.Linear(zdim, 1200),
            nn.BatchNorm1d(1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.BatchNorm1d(1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.BatchNorm1d(1200),
            nn.Tanh(),
            nn.Linear(1200, 4096 * num_channels))

        self.apply(weights_init)

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
        h = x.view(x.size(0), 4096 * self.num_channels)
        params = self.encoder(h)
        mu, log_var = torch.chunk(params, 2, dim=1)

        z = self.reparametrize(mu, log_var)
        z = self.decoder(z)
        logits = z.view(z.size(0), self.num_channels, 64, 64)

        return logits, mu, log_var
