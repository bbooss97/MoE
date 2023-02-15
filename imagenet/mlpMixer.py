import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from nn import MoeFcTokens
from nn import MoeFcTokensConvolution


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.drop=dropout
        self.dim=dim
        self.hidden_dim=hidden_dim
        self.firstSet=False

        # self.fc1=nn.Linear(dim,hidden_dim)
        # self.fc2=nn.Linear(hidden_dim,dim)

        

        # self.fc1=MoeFcTokens(dim,hidden_dim,20,1,useAttention=False)
        # self.fc2=MoeFcTokens(hidden_dim,dim,20,1,useAttention=False)

        self.net = nn.Sequential(
            # MoeFcTokensself must be a matrixConvolution(dim,hidden_dim,32,1,useAttention=True),
            MoeFcTokens(dim,hidden_dim,64,1,useAttention=True),
            # nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # MoeFcTokensConvolution(hidden_dim,dim,32,1,useAttention=True),
            MoeFcTokens(hidden_dim,dim,64,1,useAttention=True),
            # nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # if not self.firstSet:
        #     self.fc1=MoeFcTokensConvolution(self.dim,self.hidden_dim,x.shape[-2],1,useAttention=False)
        #     self.fc2=MoeFcTokensConvolution(self.hidden_dim,self.dim,x.shape[-2],1,useAttention=False)
            # self.fc1=MoeFcTokensConvolution(self.dim,self.hidden_dim,x.shape[-2],1,useAttention=False)
            # self.fc2=MoeFcTokensConvolution(self.hidden_dim,self.dim,x.shape[-2],1,useAttention=False)

        # x=self.fc1(x)
        # x=nn.GELU()(x)
        # x=nn.Dropout(self.drop)(x)
        # x=self.fc2(x)
        # x=nn.Dropout(self.drop)(x)
        # return x
        return self.net(x)
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

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):


        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)
