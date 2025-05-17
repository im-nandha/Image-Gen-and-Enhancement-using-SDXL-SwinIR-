import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  # 512 -> 256
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
        )
        self.fc_mu = nn.Conv2d(512, latent_dim, 3, 1, 1)
        self.fc_logvar = nn.Conv2d(512, latent_dim, 3, 1, 1)

    def forward(self, x):
        h = self.conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=4):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 2, 1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128 -> 256
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 256 -> 512
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.Tanh(),  # Normalize to [-1, 1]
        )

    def forward(self, z):
        return self.deconv(z)


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss
