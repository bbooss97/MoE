import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from nn import MoeFcTokens
from nn import MoeFcTokensConvolution
from nn import MoeFcTokensParallel


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.drop=dropout
        self.dim=dim
        self.hidden_dim=hidden_dim
        self.initialize=True

        # self.fc1=nn.Linear(dim,hidden_dim)
        # self.fc2=nn.Linear(hidden_dim,dim)

        # self.fc1=MoeFcTokensParallel(hidden_dim,dim,hidden_dim,1,useAttention=False)
        # self.fc2=MoeFcTokensParallel(dim,hidden_dim,hidden_dim,1,useAttention=False)

        # self.fc3=MoeFcTokensParallel(dim,dim,100,1,useAttention=False)
        # self.fc4=MoeFcTokensParallel(hidden_dim,dim,100,1,useAttention=False)

        

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        # self.net=MoeFcTokensParallel(dim,hidden_dim, dim, 64, 1, useAttention=False,dropout=dropout)

    def forward(self, x):
        # if self.initialize:
        #     self.net=MoeFcTokensConvolution(x.shape[-1],x.shape[-1],x.shape[1],1,useAttention=False).to(x.device)
        #     self.initialize=False
        x=self.net(x)
        return x
        
class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

        self.first=True

    def forward(self, x):
        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim,dropout=0.):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim,dropout))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
        self.first=True

    def forward(self, x):


        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)
        

        return self.mlp_head(x)
