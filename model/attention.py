import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["FFN", "AttentionLayer"]

class MLP(nn.Module):
    ACTIVATION = {
        "gelu": nn.GELU, 
        "tanh": nn.Tanh, 
        "sigmoid": nn.Sigmoid, 
        "relu": nn.ReLU, 
        "leaky_relu": nn.LeakyReLU(0.1),
        "softplus": nn.Softplus, 
        "ELU": nn.ELU, 
        "silu": nn.SiLU
    }
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        output_dim, 
        layer_num,
        act = "gelu", 
        res = True,
        norm = True
    ):
        super().__init__()

        if act in self.ACTIVATION.keys():
            act = self.ACTIVATION[act]
        else:
            raise NotImplementedError
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.res = res
        self.fc_pre = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            act()
        )
        self.fc_post = nn.Linear(hidden_dim, output_dim)
        self.fcs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim) if norm else nn.Identity(),
                    nn.Linear(hidden_dim, hidden_dim), 
                    act()
                ) 
                for _ in range(layer_num)
            ]
        )

    def forward(self, input):
        x = self.fc_pre(input)
        for i in range(self.layer_num):
            if self.res:
                x = self.fcs[i](x) + x
            else:
                x = self.fcs[i](x)
        x = self.fc_post(x)
        return x
    
    
class FFN(nn.Module):
    def __init__(
        self,
        input_dim, 
        hidden_dim, 
        output_dim, 
        layer_num, 
        act = "gelu", 
        res = True,
        norm = True,
        experts_num = 1
    ):
        super().__init__()
        self.experts_num = experts_num
        if experts_num == 1:
            self.ffn = MLP(
                input_dim = input_dim,
                hidden_dim = hidden_dim,
                output_dim = output_dim,
                layer_num = layer_num,
                act = act,
                res = res,
                norm = norm
            )
        else:
            self.experts = nn.ModuleList([
                MLP(
                    input_dim = input_dim,
                    hidden_dim = hidden_dim,
                    output_dim = output_dim,
                    layer_num = layer_num,
                    act = act,
                    res = res,
                    norm = norm
                ) for _ in range(experts_num)
            ])
            self.fc_weight = nn.Linear(input_dim, experts_num)

    def forward(self, x):
        if self.experts_num == 1:
            out = self.ffn(x)
        else:
            weight = self.fc_weight(x).softmax(dim = -1).unsqueeze(3)
            outs = torch.stack([
                self.experts[i](x) for i in range(self.experts_num)
            ], dim = -1)
            out = (outs @ weight).sum(dim = -1)
        return out
    

class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        output_dim,
        heads_num,
        dropout = 0,
        bias = False
    ):
        super().__init__()
        assert output_dim % heads_num == 0
        self.heads_num = heads_num
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_dim, output_dim, bias = bias)
        self.W_k = nn.Linear(key_dim, output_dim, bias = bias)
        self.W_v = nn.Linear(value_dim, output_dim, bias = bias)
        self.W_o = nn.Linear(output_dim, output_dim)
        self.d = output_dim ** -0.5
        

    def forward(self, x, y, z):
        query = rearrange(self.W_q(x), "b n (h c) -> b h n c", h = self.heads_num)
        key = rearrange(self.W_k(y), "b n (h c) -> b h n c", h = self.heads_num)
        value = rearrange(self.W_v(z), "b n (h c) -> b h n c", h = self.heads_num)
        attention_weights = (query @ key.transpose(-2, -1) * self.d).softmax(dim = -1)
        scores = rearrange(
            self.dropout(attention_weights @ value),
            "b h n c -> b n (h c)"
        )
        return self.W_o(scores)
    

class AttentionLayer(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        output_dim,
        heads_num,
        mlp_ratio,
        dropout = 0,
        act = "gelu",
        bias = False,
        experts_num = 1
    ):
        super().__init__()

        self.attn_layer = Attention(
            query_dim = query_dim,
            key_dim = key_dim,
            value_dim = value_dim,
            output_dim = output_dim,
            heads_num = heads_num,
            dropout = dropout,
            bias = bias
        )

        self.ln_x = nn.LayerNorm(query_dim)
        self.ln_y = nn.LayerNorm(key_dim)
        self.ln_z = nn.LayerNorm(value_dim)

        self.experts_num = experts_num

        if self.experts_num > 0:
            self.ffn = FFN(
                input_dim = output_dim,
                hidden_dim = int(output_dim * mlp_ratio),
                output_dim = output_dim,
                layer_num = 0,
                act = act,
                experts_num = experts_num
            )

            self.ln_t = nn.LayerNorm(output_dim)

    def forward(self, x, y, z):
        t = self.attn_layer(self.ln_x(x), self.ln_y(y), self.ln_z(z)) + x
        if self.experts_num > 0:
            t = self.ffn(self.ln_t(t)) + t            
        return t