import torch 
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv


import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch.nn.functional as F


x = torch.tensor([1, 2, 3, 4], dtype=torch.int)
y = torch.tensor([0, 1, 0, 1], dtype=torch.int)
edge_index = torch.tensor([[0, 1, 2, 0, 3],
                           [1, 0, 1, 3, 2]], dtype=torch.int)

data = Data(x=x, y=y, edge_index=edge_index)
dataset = [data]


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize= False)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


model = GCN(2, 16, 2)
device = torch.device('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

model.train()
optimizer.zero_grad()
out = model(data.x, data.edge_index, data.edge_weight)
output = model(data)
print(output)
