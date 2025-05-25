import torch
import torch.nn as nn

from typing import List
from .attention import AttentionLayer, FFN

class ContactBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_heads_num,
        dropout = 0.,
        bias = False,
        act = "gelu"
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(
            query_dim = emb_dim,
            key_dim = emb_dim,
            value_dim = emb_dim,
            output_dim = emb_dim,
            heads_num = attention_heads_num,
            mlp_ratio = 0,
            dropout = dropout,
            bias = bias,
            act = act,
            experts_num = 0
        )
        self.ln_x = nn.LayerNorm(emb_dim)
        self.ln_y = nn.LayerNorm(emb_dim)

    def forward(self, x, y):
        z = self.ln_x(x) + self.ln_y(y)
        return self.attention_layer(z, z, z)


class ContactModule(nn.Module):
    def __init__(
        self,
        emb_dim,
        contact_id: List[List[int]],
        attention_heads_num,
        dropout = 0.,
        bias = False,
        act = "gelu",
    ):
        super().__init__()
        self.contact_num = len(contact_id)
        self.contact_id = contact_id

        self.contact_blocks = nn.ModuleList(
            [
                ContactBlock(
                    emb_dim = emb_dim,
                    attention_heads_num = attention_heads_num,
                    dropout = dropout,
                    bias = bias,
                    act = act
                ) for _ in range(self.contact_num)
            ] 
        )

    def get_contact_pairs(self, solids_tokens):
        contact_pairs = [
            [
                solids_tokens[self.contact_id[i][0]],
                solids_tokens[self.contact_id[i][1]]
            ] for i in range(self.contact_num)
        ]
        return contact_pairs
    

    def forward(self, solids_tokens: List[torch.Tensor]):
        contact_pairs = self.get_contact_pairs(solids_tokens)
        contact_constraints = [
            contact_block(
                contact_pairs[i][0],
                contact_pairs[i][1]
            ) for i, contact_block in enumerate(self.contact_blocks)
        ]
        return contact_constraints
    
    
class InteractionAllocation(nn.Module):
    def __init__(
        self,
        emb_dim,
        mlp_ratio,
        interaction_type = "contact",
        act = "gelu",
    ):
        super().__init__()
        self.interaction_type = interaction_type
        self.fc_weight = nn.Linear(emb_dim, emb_dim)
        if interaction_type == "contact":
            self.fc = nn.Sequential(
                nn.LayerNorm(emb_dim),
                FFN(
                    input_dim = emb_dim,
                    hidden_dim = emb_dim * mlp_ratio,
                    output_dim = emb_dim,
                    layer_num = 0,
                    act = act
                )
            )
        

    def forward(self, interactions):
        interactions_stack = torch.stack(interactions, dim = -1).permute(0, 3, 1, 2)
        weights = self.fc_weight(interactions_stack).permute(0, 2, 3, 1).softmax(-1)
        x = (weights * interactions_stack.permute(0, 2, 3, 1)).sum(dim = -1)
        return self.fc(x) + x if self.interaction_type == "contact" else x
    

class LoadBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        load_num,
        contact_num,
        attention_heads_num,
        mlp_ratio,
        dropout = 0.,
        bias = False,
        act = "gelu"
    ):
        super().__init__()
        self.load_num = load_num
        if load_num > 1:
            self.loads_allocation = InteractionAllocation(
                emb_dim = emb_dim, 
                mlp_ratio = mlp_ratio, 
                interaction_type = "load",
                act = act
            )
        self.contact_num = contact_num
        if contact_num > 1:
            self.contact_constraints_allocation = InteractionAllocation(
                emb_dim = emb_dim, 
                mlp_ratio = mlp_ratio, 
                interaction_type = "contact",
                act = act
            )
        self.attention_layer = AttentionLayer(
            query_dim = emb_dim,
            key_dim = emb_dim,
            value_dim = emb_dim,
            output_dim = emb_dim,
            heads_num = attention_heads_num,
            mlp_ratio = mlp_ratio,
            dropout = dropout,
            bias = bias,
            act = act,
            experts_num = 1
        )

        self.ln_load = nn.LayerNorm(emb_dim)
        self.ln_contact_constraint = nn.LayerNorm(emb_dim)
        self.ln_deformable_solid = nn.LayerNorm(emb_dim)

    def forward(self, loads, deformable_solid, contact_constraints):
        if self.load_num > 1:
            load = self.loads_allocation(loads)
        else:
            load = loads[0]
        if self.contact_num > 1:
            contact_constraint = self.contact_constraints_allocation(contact_constraints)
        else:
            contact_constraint = contact_constraints[0]
        z = self.ln_load(load) + self.ln_contact_constraint(contact_constraint) + self.ln_deformable_solid(deformable_solid)
        return self.attention_layer(z, z, z)
    

class LoadModule(nn.Module):
    def __init__(
        self,
        emb_dim,
        load_num,
        contact_num,
        deformable_num,
        attention_heads_num,
        mlp_ratio,
        dropout = 0.,
        bias = False,
        act = "gelu"
    ):
        super().__init__()
        self.load_blocks = nn.ModuleList(
            [
                LoadBlock(
                    emb_dim = emb_dim,
                    load_num = load_num,
                    contact_num = contact_num,
                    attention_heads_num = attention_heads_num,
                    mlp_ratio = mlp_ratio,
                    dropout = dropout,
                    bias = bias,
                    act = act
                ) for _ in range(deformable_num)
            ]  
        )

    def forward(self, loads, deformable_solids, contact_constraints):
        return [
            load_block(loads, deformable_solids[i], contact_constraints)
            for i, load_block in enumerate(self.load_blocks)
        ]


class Processor(nn.Module):
    def __init__(
        self,
        emb_dim,
        load_num,
        rigid_num,
        deformable_num,
        contact_id: List[List[int]],
        attention_heads_num,
        mlp_ratio,
        dropout = 0.,
        bias = False,
        act = "gelu",
    ):
        super().__init__()
        self.load_num = load_num
        self.rigid_num = rigid_num
        self.deformable_num = deformable_num            

        self.contact_module = ContactModule(
            emb_dim = emb_dim,
            contact_id = contact_id,
            attention_heads_num = attention_heads_num,
            dropout = dropout,
            bias = bias,
            act = act
        )

        self.load_module = LoadModule(
            emb_dim = emb_dim,
            load_num = load_num,
            contact_num = len(contact_id),
            deformable_num = deformable_num,
            attention_heads_num = attention_heads_num,
            mlp_ratio = mlp_ratio,
            dropout = dropout,
            bias = bias,
            act = act
        )


    def forward(self, loads_tokens, rigid_tokens, deformable_tokens):
        assert len(loads_tokens) == self.load_num
        assert len(rigid_tokens) == self.rigid_num
        assert len(deformable_tokens) == self.deformable_num
        contact_constraints = self.contact_module(
            deformable_tokens + rigid_tokens
        )
        updated_deformable_solids_tokens = self.load_module(
            loads_tokens,
            deformable_tokens,
            contact_constraints
        )
        
        return updated_deformable_solids_tokens