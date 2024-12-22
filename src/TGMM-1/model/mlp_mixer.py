import torch.nn as nn
from einops.layers.torch import Rearrange

from model.elements import FeedForward


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('B p d -> B d p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('B d p -> B p d'),
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
    def __init__(self, nhid, nlayer, n_patches, dropout=0, with_final_norm=True):
        super().__init__()

        self.n_patches = n_patches
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, nhid*4, nhid//2, dropout=dropout) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x
    

class MixerBlockTemporal(nn.Module):

    def __init__(self, n_features, n_spatial, n_timesteps, spatial_hiddim, features_hiddim, temporal_hiddim, dropout=0.):
        super().__init__()
        """
        Note that nn.Linear and hence FeedForward only work on the last dimension of the input tensor, 
        applying the transformation to all leading dimensions. So need to rearrange the input tensor to 
        apply the transformation to the last dimension.
        """
        self.token_mix = nn.Sequential(
            nn.LayerNorm(n_features),
            Rearrange('B t s f -> B t f s'),  # (batch, time, space, features)
            FeedForward(n_spatial, spatial_hiddim, dropout),
            Rearrange('B t f s -> B t s f'),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(n_features),
            FeedForward(n_features, features_hiddim, dropout),
        )
        self.temporal_mix = nn.Sequential(
            nn.LayerNorm(n_features),
            Rearrange('B t s f -> B s f t'),
            FeedForward(n_timesteps, temporal_hiddim, dropout),
            Rearrange('B s f t -> B t s f'),
        )

    def forward(self, x):
        """
        x: (batch_size, t_steps, n_patches, n_features) = (b, t, p, d)
        """
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        x = x + self.temporal_mix(x)
        return x

class MLPMixerTemporal(nn.Module):
    def __init__(self, n_features, n_spatial, n_timesteps, n_layer, dropout=0, with_final_norm=True):
        super().__init__()

        self.n_spatial = n_spatial
        self.n_timesteps = n_timesteps
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList([
            MixerBlockTemporal(n_features, n_spatial, n_timesteps, n_features*4, n_spatial*2, n_timesteps*2, dropout=dropout)  # FIXME: Check what to use for hidden dims
            for _ in range(n_layer)
        ])  
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(n_features)

    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x