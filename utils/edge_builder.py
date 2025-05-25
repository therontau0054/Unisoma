import torch
import torch_geometric.nn as nng

def knn_edge_builder(pc, k, offset = False):
    B, N, C = pc.shape
    x = pc.view(-1, C)
    batch_tensor = torch.arange(B, device = x.device).repeat_interleave(N)
    edge_idx = nng.knn_graph(x, k = k, batch = batch_tensor)
    edge_attr = torch.norm(x[edge_idx[0]] - x[edge_idx[1]], dim = -1, p = 2).reshape(B, -1, 1)

    with torch.no_grad():
        src, dst = edge_idx
        assert (batch_tensor[src] == batch_tensor[dst]).all(), "KNN边跨越了batch之间"
    
    edge_idx = edge_idx.T.reshape(B, -1, 2)
    if not offset:
        edge_idx -= (torch.arange(B, device = pc.device) * N).reshape(B, 1, 1)

    return [edge_attr, edge_idx]