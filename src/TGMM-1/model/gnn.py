import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from einops import rearrange

from model.elements import MLP

BN = True


class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True, **kwargs):
        super().__init__()
        # self.nn = MLP(nin, nout, 2, False, bias=bias)
        # self.layer = gnn.GCNConv(nin, nin, bias=True)
        self.layer = gnn.GCNConv(nin, nout, bias=bias, **kwargs)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)
        # return self.nn(F.relu(self.layer(x, edge_index)))


class ResGatedGraphConv(nn.Module):
    def __init__(self, nin, nout, bias=True, **kwargs):
        super().__init__()
        self.layer = gnn.ResGatedGraphConv(nin, nout, bias=bias, **kwargs)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


class GINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True, **kwargs):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = gnn.GINEConv(self.nn, train_eps=True, **kwargs)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class TransformerConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=8, **kwargs):
        super().__init__()
        self.layer = gnn.TransformerConv(
            in_channels=nin, out_channels=nout//nhead, heads=nhead, edge_dim=nin, bias=bias, **kwargs)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class GATConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=1, **kwargs):
        super().__init__()
        self.layer = gnn.GATConv(nin, nout//nhead, nhead, bias=bias, **kwargs)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class GatedGraphConv(nn.Module):
    def __init__(self, nin, nout, bias=True, **kwargs):
        super().__init__()
        self.layer = gnn.GatedGraphConv(nin, nout, bias=bias, **kwargs)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


gnns = {
    'GCNConv': GCNConv,
    'ResGatedGraphConv': ResGatedGraphConv,
    'GINEConv': GINEConv,
    'TransformerConv': TransformerConv,
    'GATConv': GATConv,
    'GatedGraphConv': GatedGraphConv
}


class GNN(nn.Module):
    def __init__(self,
                 nin,
                 nout,
                 nlayer_gnn,
                 gnn_type,
                 bn=BN,
                 dropout=0.0,
                 res=True):
        super().__init__()
        self.dropout = dropout
        self.res = res

        self.convs = nn.ModuleList([gnns[gnn_type](
            nin, nin, bias=not bn) for _ in range(nlayer_gnn)])
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(nin) if bn else nn.Identity() for _ in range(nlayer_gnn)])
        self.output_encoder = nn.Linear(nin, nout)

    def forward(self, x, edge_index, edge_attr):
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = rearrange(x, 'B n f -> (B n) f')  # batch, nodes, features FIXME: Fix this for additional time dim
            x = norm(x)
            x = rearrange(x, '(B n) f -> B n f', n=previous_x.shape[1])
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x

        x = self.output_encoder(x)
        return x