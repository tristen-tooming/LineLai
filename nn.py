from torch.nn import Linear, ReLU
from torch_geometric.nn import SAGEConv
import torch

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()

        self.conv1 = SAGEConv(1, 32)
        self.lin = Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = ReLU(self.conv1(x, edge_index))
        x = self.lin(x) # Why?

        return x
