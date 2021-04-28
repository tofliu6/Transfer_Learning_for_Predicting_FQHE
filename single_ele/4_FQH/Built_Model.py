import math
import torch
from torch import nn

import torch.nn.functional as F
from torch import Tensor

'''
class PositionalEncoding(object):
    pass


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout = 0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)

def inputseq_to_tensor(input: Tensor, output: Tensor):
'''

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor):
    temp = query.bmm(key.transpose(1,2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp/scale, dim = -1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: Tensor, key:Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim = -1)
        )


def position_encoding(
        seq_len: int, dim_model: int, device: torch.device = torch.device('cpu')
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_input: int = 5, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input)
    )


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 5,
            num_heads: int = 5,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout = dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout = dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


def linear_encoded(input: Tensor) -> Tensor:
    seq_width = input.size(-1)
    init_layer = nn.Linear(seq_width, 5)
    tensor = init_layer(input)
    tensor = tensor.reshape(1, -1, 5)
    #encode_matrix = torch.randn(seq_width, 15)
    #tensor = (input @ encode_matrix).reshape(1, -1, 15)
    return tensor


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            dim_model: int = 5,
            num_heads: int = 5,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor) -> Tensor:
        #add a linear encoded layer
        src = linear_encoded(src)
        # src = src.reshape(1, 1, -1)
        seq_len, dimension = src.size(-2), src.size(-1)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 5,
            num_heads: int = 5,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout = dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout = dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout = dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(memory, memory, tgt)
        return self.feed_forward(tgt)


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            dim_model: int = 5,
            num_heads: int = 5,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        # add a linear encoded layer
        # tgt = linear_encoded(tgt)
        tgt = tgt.reshape(1, -1, 5)
        seq_len, dimension = tgt.size(-2), tgt.size(-1)
        tgt += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return self.linear(tgt)

class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_model: int = 5,
            num_heads: int = 5,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers = num_encoder_layers,
            dim_model = dim_model,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers = num_decoder_layers,
            dim_model = dim_model,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
        )

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.decoder(tgt, self.encoder(src))

if __name__ == '__main__':
    src = torch.randn(10, 3)
    tgt = torch.randn(10, 5)
    out = Transformer()(src, tgt)
    print(out.shape)
