import torch.nn as nn
from einops.layers.torch import Rearrange


BN = True


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b p d -> b d p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d p -> b p d'),
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
    def __init__(self,
                 nhid,
                 nlayer,
                 n_patches,
                 dropout=0,
                 with_final_norm=True
                 ):
        super().__init__()
        self.n_patches = n_patches
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, nhid*4, nhid//2, dropout=dropout) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x, coarsen_adj, mask):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x
    

class MixerBlockTemporal(nn.Module):

    def __init__(self, dim, num_patch, num_timesteps, token_dim, channel_dim, temporal_dim, dropout=0.):
        super().__init__()
        """
        Note that nn.Linear and hence FeedForward only work on the last dimension of the input tensor, 
        applying the transformation to all leading dimensions. So need to rearrange the input tensor to 
        apply the transformation to the last dimension.
        """
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('B t p d -> B t d p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('B t d p -> B t p d'),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )
        self.temporal_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('B t p d -> B p d t'),
            FeedForward(num_timesteps, temporal_dim, dropout),
            Rearrange('B p d t -> B t p d'),
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
    def __init__(self,
                 nhid,
                 nlayer,
                 n_patches,
                 num_timesteps,
                 dropout=0,
                 with_final_norm=True
                 ):
        super().__init__()
        self.n_patches = n_patches
        self.num_timesteps = num_timesteps
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlockTemporal(nhid, self.n_patches, self.num_timesteps, nhid*4, nhid//2, self.num_timesteps, dropout=dropout) for _ in range(nlayer)])  # FIXME: Check what to use for temporal_dim.
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x, coarsen_adj, mask):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x