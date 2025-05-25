import torch
import torch.nn as nn
from .attention import FFN

class SliceBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        slice_num,
        use_edge = False
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.slice_num = slice_num
        self.use_edge = use_edge

        self.in_project_x = nn.Linear(emb_dim, emb_dim)
        self.in_project_fx = nn.Linear(emb_dim, emb_dim)
        self.in_project_slice = nn.Linear(emb_dim, slice_num)
        self.temperature_x = nn.Parameter(
            torch.ones([1, 1, slice_num]) * 0.5,
            requires_grad = True
        )

        self.place_holder_x = nn.Parameter(torch.rand(1, 1, emb_dim) / emb_dim)

        if use_edge:
            self.in_project_fedge = nn.Linear(1, emb_dim)
            self.temperature_edge = nn.Parameter(
                torch.ones([1, 1, slice_num]) * 0.5,
                requires_grad = True
            )
            self.project_edge_slice = nn.Linear(slice_num, slice_num)
            self.place_holder_edge = nn.Parameter(torch.rand(1, 1, emb_dim) / emb_dim)

    def forward(self, input):
        x = input[0] if self.use_edge else input
        fx_mid = self.in_project_fx(x)
        x_mid = self.in_project_x(x) + self.place_holder_x
        slice_weight = (self.in_project_slice(x_mid) / self.temperature_x.clamp(min = 0.01, max = 5)).softmax(-1)
        slice_norm = slice_weight.sum(dim = -2).unsqueeze(-1)
        slice_token = torch.einsum("bnm,bnc->bmc", slice_weight, fx_mid)

        if self.use_edge:
            edge_attr, edge_idx = input[1], input[2]
            
            fedge_mid = self.in_project_fedge(edge_attr) + self.place_holder_edge
            edge_slice_weight = self.project_edge_slice(
                (
                    slice_weight.gather(1, edge_idx[:, :, 0 : 1].expand(-1, -1, self.slice_num)) +
                    slice_weight.gather(1, edge_idx[:, :, 1 : 2].expand(-1, -1, self.slice_num)) 
                ) / self.temperature_edge.clamp(min = 0.01, max = 5)
            ).softmax(dim = -1)
            
            edge_slice_norm = edge_slice_weight.sum(dim = -2).unsqueeze(-1)
            edge_slice_token = torch.einsum("bnm,bnc->bmc", edge_slice_weight, fedge_mid)
            scale_factor = x.shape[1] / edge_attr.shape[1]
            
            slice_token = slice_token + edge_slice_token * scale_factor
            slice_norm = slice_norm + edge_slice_norm * scale_factor
        slice_token = slice_token / (slice_norm + 1e-5)

        return slice_token, slice_weight
    
class DesliceBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        mlp_ratio,
        act = "gelu"
    ):
        super().__init__()
        self.fc_out = FFN(
            input_dim = emb_dim,
            hidden_dim = emb_dim * mlp_ratio,
            output_dim = emb_dim,
            layer_num = 0,
            act = act
        )

        self.place_holder = nn.Parameter(torch.rand(1, 1, emb_dim) / emb_dim)

    def forward(self, slice_token, slice_weight, feature):
        out = slice_weight @ slice_token + self.place_holder + feature
        return self.fc_out(out) + out

