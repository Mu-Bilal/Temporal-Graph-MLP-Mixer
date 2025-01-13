import torch.nn as nn
import torch.nn.functional as F


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


class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=True, bias=True, dropout=0.):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
            n_hid if i < nlayer-1 else nout,
            bias=bias)  # Simplified bias logic - just use the bias parameter
            for i in range(nlayer)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(n_hid if i < nlayer-1 else nout) 
            if (with_norm and i < nlayer-1 or (i == nlayer-1 and with_final_activation)) 
            else nn.Identity()
            for i in range(nlayer)
        ])
        
        if isinstance(dropout, (int, float)):
            dropout = [dropout] * nlayer
        assert len(dropout) == nlayer, f"Expected {nlayer} dropout values, got {len(dropout)}"
        
        self.dropouts = nn.ModuleList([nn.Dropout(drop) for drop in dropout])
        
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)  # TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            if isinstance(norm, nn.LayerNorm):
                norm.reset_parameters()

    def forward(self, x):
        # previous_x = x  # FIXME: Check why we are not using this??
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            x = dropout(x)
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)

        # if self.residual:
        #     x = x + previous_x
        return x
