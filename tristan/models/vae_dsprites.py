import torch
import torch.nn as nn
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

def load_module(state_dict, module_name):
    return (dict((k.replace(module_name, ''), v) for k, v in state_dict.items()
            if k.startswith(module_name)))

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            encoder_block(1, 32),
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
            nn.Linear(256, 2 * 10))

        self.decoder_ffwd = nn.Sequential(
            nn.Linear(10, 256),
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
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1))

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
