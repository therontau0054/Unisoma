from .attention import *
from .slice import *
from .processor import *

class Unisoma(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        slice_num: int,
        use_edge: bool,
        load_dim: int,
        rigid_dim: int,
        input_deformable_dim: int,
        output_deformable_dim: int,
        load_num: int,
        rigid_num: int,
        deformable_num: int,
        contact_id: List[List[int]], # order: deformable, rigid
        processor_num: int,
        attention_heads_num: int,
        mlp_ratio: int = 2,
        dropout: float = 0.,
        bias: bool = False,
        act: str = "gelu"
    ):
        super().__init__()
        self.load_num = load_num
        self.rigid_num = rigid_num
        self.deformable_num = deformable_num
        self.processor_num = processor_num
        self.use_edge = use_edge

        self.load_ffn = nn.ModuleList(
            [
                nn.Linear(load_dim, emb_dim)
                for _ in range(self.load_num)
            ]
        )

        self.rigid_ffn = nn.ModuleList(
            [
                nn.Linear(rigid_dim, emb_dim)
                for _ in range(self.rigid_num)
            ]
        )

        self.deformable_ffn = nn.ModuleList(
            [
                FFN(
                    input_dim = input_deformable_dim,
                    hidden_dim = emb_dim * 2,
                    output_dim = emb_dim,
                    layer_num = 0,
                    act = act
                ) for _ in range(self.deformable_num)
            ]
        )

        self.load_slice_blocks = nn.ModuleList(
            [
                SliceBlock(
                    emb_dim = emb_dim,
                    slice_num = slice_num
                ) for _ in range(self.load_num)
            ]
        )

        self.rigid_slice_blocks = nn.ModuleList(
            [
                SliceBlock(
                    emb_dim = emb_dim,
                    slice_num = slice_num
                ) for _ in range(self.rigid_num)
            ]
        )

        self.deformable_slice_blocks = nn.ModuleList(
            [
                SliceBlock(
                    emb_dim = emb_dim,
                    slice_num = slice_num,
                    use_edge = use_edge
                ) for _ in range(self.deformable_num)
            ]
        )

        self.processors = nn.ModuleList(
            [
                Processor(
                    emb_dim = emb_dim,
                    load_num = load_num,
                    rigid_num = rigid_num,
                    deformable_num = deformable_num,
                    contact_id = contact_id,
                    attention_heads_num = attention_heads_num,
                    mlp_ratio = mlp_ratio,
                    dropout = dropout,
                    bias = bias,
                    act = act
                ) for _ in range(self.processor_num)
            ]
        )

        self.deformable_deslice_blocks = nn.ModuleList(
            [
                DesliceBlock(
                    emb_dim = emb_dim,
                    mlp_ratio = mlp_ratio,
                    act = act
                ) for _ in range(self.deformable_num)
            ]
        )

        self.deformable_ffn_out = nn.ModuleList(
            [
                FFN(
                    input_dim = emb_dim,
                    hidden_dim = emb_dim * mlp_ratio,
                    output_dim = output_deformable_dim,
                    layer_num = 0,
                    act = act
                ) for _ in range(self.deformable_num)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = .02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, loads, rigid_solids, deformable_solids, deformable_edges = None):
        assert len(loads) == self.load_num
        assert len(rigid_solids) == self.rigid_num
        assert len(deformable_solids) == self.deformable_num

        load_features = [self.load_ffn[i](loads[i]) for i in range(self.load_num)]
        rigid_features = [self.rigid_ffn[i](rigid_solids[i]) for i in range(self.rigid_num)]
        deformable_features = [self.deformable_ffn[i](deformable_solids[i]) for i in range(self.deformable_num)]

        loads_tokens = [
            self.load_slice_blocks[i](load_features[i])[0]
            for i in range(self.load_num)
        ]
        
        rigid_tokens = [
            self.rigid_slice_blocks[i](rigid_features[i])[0]
            for i in range(self.rigid_num)
        ]

        deformable_tokens, deformable_weights = \
            map(
                lambda t: list(t),
                zip(
                    *[
                        self.deformable_slice_blocks[i](deformable_features[i]) if not self.use_edge else
                        self.deformable_slice_blocks[i]([deformable_features[i]] + list(deformable_edges[i]))
                        for i in range(self.deformable_num)
                    ]
                )
            )

        for processor in self.processors:
            deformable_tokens = processor(loads_tokens, rigid_tokens, deformable_tokens)

        updated_deformable_solids = [
            self.deformable_ffn_out[i](
                self.deformable_deslice_blocks[i](
                    deformable_tokens[i],
                    deformable_weights[i],
                    deformable_features[i]
                ) + deformable_features[i]
            ) for i in range(self.deformable_num)
        ]

        return updated_deformable_solids
    
