from torch_geometric.data import HeteroData
import torch

class Graph(HeteroData):
    def __init__(self, n_nodes) -> None:
        self.n_nodes = n_nodes

    def _make_graph(self):
        data = HeteroData()
        for node in range(self.n_nodes):
            self.data[f'machine_{node}'].x = torch.randn([1])

        # Not in COO format
        for edge in range(self.n_nodes - 1):
            self.data[f'machine_{edge}', f'edge_{edge}', f'machine_{edge + 1}'].edge_index = torch.tensor([[edge, edge + 1],
                                                                                                    [edge + 1, edge]]).t().contiguous()
            #data[f'machine_{edge + 1}', f'edge_{edge}_reverse', f'machine_{edge}'].edge_index = torch.tensor([[edge + 1, edge]])




if __name__ == "__main__":
    data = Graph(3)
    print(f"Data: {data}\n")
    print(f"Stores: {data.stores}\n")
    print(f"Node types: {data.node_types}\n")
    print(f"Metadata: {data.metadata()}\n")
    print(f"Machines 0-1:\n{data['machine_0', 'edge_0', 'machine_1']}")
    #print(data['machine_1', 'edge_1', 'machine_2'])

    print(data.x_dict)