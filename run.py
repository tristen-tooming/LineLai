# Torch stuff
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# Torch Geometric
from torch_geometric.nn import SAGEConv, to_hetero, Sequential



from make_graph import Graph
from nn import GNN

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Graph(3)

model = to_hetero(GNN, metadata=data).to(device)
print(model)

####
#model = Sequential('x, edge_index', [
#    (SAGEConv(1, 32), 'x, edge_index -> x'),
#    nn.ReLU(inplace=True),
#    (SAGEConv(32, 32), 'x, edge_index -> x'),
#    nn.ReLU(inplace=True),
#    (nn.Linear(32, 1), 'x -> x'),
#])
#
#print(model)
#data, model = data.to(device), model.to(device)
#model = to_hetero(model, metadata=data.metadata(), aggr='sum')
#print(model)
#
#model.train()
#with torch.no_grad():  # Initialize lazy modules.
#    out = model(data.x_dict, data.edge_index_dict)
