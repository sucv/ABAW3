import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from models.temporal_convolutional_model import TemporalBlock


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.data.masked_fill_(mask, float('-inf'))
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the models)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, gate=None, mask=None, return_attention=False):

        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        if gate is not None:
            gate = gate[:, None, None, :].repeat(1, self.num_heads, seq_length, 1)
            q, k = q * gate, k * gate
        # # Determine value outputs
        # mask = np.triu(np.ones((q.shape[0], q.shape[1], q.shape[2], q.shape[2])), k=1).astype('bool')
        # mask = torch.from_numpy(mask).cuda()

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class MultimodalMultiheadAttention(nn.Module):
    def __init__(self, modalities, input_dim, modal_dim, num_heads):
        super().__init__()

        self.modalities = modalities

        assert modal_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = modal_dim
        self.num_heads = num_heads
        self.head_dim = modal_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.ModuleDict()

        for modal in modalities:
            self.qkv_proj[modal] = nn.Linear(input_dim[modal], 3*modal_dim)

        self.o_proj = nn.Linear(modal_dim*len(self.modalities), modal_dim*len(self.modalities))

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        for modal in self.modalities:
            nn.init.xavier_uniform_(self.qkv_proj[modal].weight)
            self.qkv_proj[modal].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x[self.modalities[0]].size()

        Q, K, V = [], [], []

        for modal in self.modalities:
            qkv = self.qkv_proj[modal](x[modal])

            # Separate Q, K, V from linear output
            qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 1, 3*self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3, 4) # [Batch, Head, SeqLen, Modal, Dims]
            q, k, v = qkv.chunk(3, dim=-1)

            # q = q.view(batch_size, )
            Q.append(q)
            K.append(k)
            V.append(v)

        Q = torch.cat(Q, dim=-2)
        K = torch.cat(K, dim=-2)
        V = torch.cat(V, dim=-2)

        # Determine value outputs
        values, attention = scaled_dot_product(Q, K, V, mask=mask)
        values += V
        values = values.permute(0, 2, 1, 3, 4) # [Batch, SeqLen, Head, Modal, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim*len(self.modalities))
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class MultiModalEncoderBlock(nn.Module):

    def __init__(self, modalities, input_dim, modal_dim, num_heads, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """

        # Attention layer
        super().__init__()
        self.self_attn = MultimodalMultiheadAttention(modalities, input_dim, modal_dim, num_heads)

        # Two-layer MLP
        mlp_input_dim = np.sum([dim for dim in input_dim.values()])
        output_dim = modal_dim * len(modalities)

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = self.dropout(attn_out)
        x = self.norm1(x)
        return x


class MultimodalTransformerEncoder(nn.Module):
    def __init__(self, modalities, input_dim, modal_dim, num_heads,  dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers = MultiModalEncoderBlock(modalities, input_dim, modal_dim, num_heads, dropout)

    def forward(self, x, mask=None):
        x = self.layers(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        _, attn_map = self.layers.self_attn(x, mask=mask, return_attention=True)
        attention_maps.append(attn_map)
        return attention_maps


class InterModalMultiheadAttention(nn.Module):
    def __init__(self, modalities, input_dim, modal_dim, num_heads):
        super().__init__()

        self.modalities = modalities

        assert modal_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = modal_dim
        self.num_heads = num_heads
        self.head_dim = modal_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.ModuleDict()

        for modal in modalities:
            self.qkv_proj[modal] = nn.Linear(input_dim[modal], 3*modal_dim)

        self.o_proj = nn.Linear(modal_dim*len(self.modalities), modal_dim*len(self.modalities))

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        for modal in self.modalities:
            nn.init.xavier_uniform_(self.qkv_proj[modal].weight)
            self.qkv_proj[modal].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x[self.modalities[0]].size()

        Q, K, V = [], [], []

        for modal in self.modalities:
            qkv = self.qkv_proj[modal](x[modal])

            # Separate Q, K, V from linear output
            qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 1, 3*self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3, 4) # [Batch, Head, SeqLen, Modal, Dims]
            q, k, v = qkv.chunk(3, dim=-1)

            # q = q.view(batch_size, )
            Q.append(q)
            K.append(k)
            V.append(v)

        Q = torch.cat(Q, dim=-2)
        K = torch.cat(K, dim=-2)
        V = torch.cat(V, dim=-2)

        # Determine value outputs
        values, attention = scaled_dot_product(Q, K, V, mask=mask)
        values += V
        values = values.permute(0, 2, 1, 3, 4) # [Batch, SeqLen, Head, Modal, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim*len(self.modalities))
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class IntraEncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gate=None, mask=None):
        # Attention part
        attn_out = self.self_attn(x, gate=gate, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class InterModalEncoderBlock(nn.Module):

    def __init__(self, modalities, input_dim, modal_dim, num_heads, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """

        # Attention layer
        super().__init__()
        self.self_attn = InterModalMultiheadAttention(modalities, input_dim, modal_dim, num_heads)

        # Two-layer MLP
        mlp_input_dim = np.sum([dim for dim in input_dim.values()])
        output_dim = modal_dim * len(modalities)

        self.linear_net = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        x = self.linear_net(x)
        x = self.norm2(x)

        return x


class IntraModalTransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([IntraEncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, gate=None, mask=None):
        for l in self.layers:
            x = l(x, gate=gate, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class InterModalTransformerEncoder(nn.Module):
    def __init__(self, modalities, input_dim, modal_dim, num_heads,  dropout=0.0):
        super().__init__()

        self.layers = InterModalEncoderBlock(modalities, input_dim, modal_dim, num_heads, dropout)

    def forward(self, x, mask=None):
        x = self.layers(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        _, attn_map = self.layers.self_attn(x, mask=mask, return_attention=True)
        attention_maps.append(attn_map)
        return attention_maps