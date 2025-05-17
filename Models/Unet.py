import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(context_dim, hidden_dim)
        self.v = nn.Linear(context_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, x, context):
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = self.out(torch.matmul(attn, v))
        return out + x  # residual connection


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        h += self.time_emb_proj(t).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, latent_dim=4, base_dim=320, time_emb_dim=1280, text_emb_dim=1024):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Input projection
        self.init_conv = nn.Conv2d(latent_dim, base_dim, 3, padding=1)

        # Downsampling blocks
        self.down1 = ResBlock(base_dim, base_dim, time_emb_dim)
        self.down2 = ResBlock(base_dim, base_dim * 2, time_emb_dim)
        self.pool = nn.AvgPool2d(2)

        # Mid block with cross-attention
        self.mid_block1 = ResBlock(base_dim * 2, base_dim * 2, time_emb_dim)
        self.cross_attn = CrossAttentionBlock(base_dim * 2, text_emb_dim)
        self.mid_block2 = ResBlock(base_dim * 2, base_dim * 2, time_emb_dim)

        # Upsampling blocks
        self.up1 = ResBlock(base_dim * 2, base_dim, time_emb_dim)
        self.up2 = ResBlock(base_dim, base_dim, time_emb_dim)

        # Final conv
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, latent_dim, 3, padding=1)
        )

    def forward(self, x, t, text_emb):
        t_emb = self.time_mlp(t.unsqueeze(1))

        # Downsample
        x = self.init_conv(x)
        x = self.down1(x, t_emb)
        x = self.pool(x)
        x = self.down2(x, t_emb)

        # Mid block
        x = self.mid_block1(x, t_emb)
        b, c, h, w = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_reshaped = self.cross_attn(x_reshaped, text_emb)
        x = x_reshaped.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.mid_block2(x, t_emb)

        # Upsample
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up1(x, t_emb)
        x = self.up2(x, t_emb)

        return self.final_conv(x)
