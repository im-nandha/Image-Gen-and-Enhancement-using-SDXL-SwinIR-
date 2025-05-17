import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinIR(nn.Module):
    def __init__(self, img_size=64, in_chans=3, embed_dim=96, depths=[6,6,6,6], num_heads=[6,6,6,6]):
        super(SwinIR, self).__init__()
        self.swin = SwinTransformer(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
        )
        self.conv = nn.Conv2d(embed_dim, in_chans, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.swin(x)
        x = self.conv(x)
        return x
