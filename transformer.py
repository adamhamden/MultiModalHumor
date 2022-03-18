import torch
import torch.nn as nn
import math
import numpy as np

"""
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""


def get_attention_mask(seq1, seq2):
    length = seq1.size(1)
    numpy_seq2 = seq2.detach().cpu().numpy()

    padding = np.where(~numpy_seq2.any(axis=2))
    mask = np.zeros((numpy_seq2.shape[:2]))
    mask[padding[0], padding[1]] = 1
    mask = torch.from_numpy(mask).unsqueeze(1).expand(-1, length, -1)
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000, dropout=.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config, temperature, dropout):
        super(ScaledDotProductAttention, self).__init__()

        self.config = config
        self.device = config['device']
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V, mask=None):

        Q.to(self.device)
        K.to(self.device)
        V.to(self.device)
        mask.to(self.device)

        attention = torch.matmul(Q, K.transpose(2, 3))
        attention /= self.temperature

        if mask is not None:
            attention = attention.masked_fill(mask, -np.inf)

        attention = self.softmax(attention)
        attention = self.dropout(attention)

        output = torch.matmul(attention, V)

        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, config, n_heads, d_model, d_key, d_value, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.config = config
        self.device = config['device']

        self.n_head = n_heads
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value

        self.W_Q = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_value, bias=False)

        self.attention = ScaledDotProductAttention(config=config, temperature=d_key ** .5, dropout=dropout)
        self.ff = nn.Linear(n_heads * d_value, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask):
        residual = Q
        batch_size = Q.size(0)

        Q.to(self.device)
        K.to(self.device)
        V.to(self.device)
        mask.to(self.device)

        Q_0 = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_key).transpose(1, 2)
        K_0 = self.W_K(K).view(batch_size, -1, self.n_head, self.d_key).transpose(1, 2)
        V_0 = self.W_V(V).view(batch_size, -1, self.n_head, self.d_value).transpose(1, 2)

        mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        q, attention = self.attention(Q_0, K_0, V_0, mask)
        q = q.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_value)
        q = self.ff(q)
        q = self.dropout(q)
        q += residual

        q = self.layer_norm(q)

        return q, attention


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_feedforward):
        super(PoswiseFeedForwardNet, self).__init__()

        self.ff = nn.Sequential(nn.Linear(d_model, d_feedforward, bias=False),
                                nn.ReLU(),
                                nn.Linear(d_feedforward, d_model)
                                )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        residual = input
        output = self.ff(input)
        output = self.layer_norm(output + residual)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, config, d_model, n_heads, d_feedforward, d_key, d_value):
        super(EncoderLayer, self).__init__()

        self.config = config
        self.device = config['device']

        self.self_attention = MultiHeadAttention(config=config, d_model=d_model, n_heads=n_heads, d_key=d_key, d_value=d_value)
        self.pos_ff = PoswiseFeedForwardNet(d_model=d_model, d_feedforward=d_feedforward)

    def forward(self, input, mask):

        input.to(self.device)
        mask.to(self.device)

        output, attention = self.self_attention(input, input, input, mask)
        output = self.pos_ff(output)

        return output, attention


class Encoder(nn.Module):
    def __init__(self, config, d_input, d_model, n_heads, n_layers, d_feedforward, d_key, d_value, dropout=0.1):
        super(Encoder, self).__init__()

        self.config = config
        self.device = config['device']

        self.input_encoder = nn.Linear(d_input, d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(config=config, d_model=d_model, n_heads=n_heads, d_feedforward=d_feedforward, d_key=d_key, d_value=d_value) for _ in range(n_layers)])

    def forward(self, input):

        input.to(self.device)

        output = self.input_encoder(input)
        output = self.positional_encoder(output.transpose(0, 1)).transpose(1, 0)

        attention_mask = get_attention_mask(input, input)
        attention_mask = attention_mask.type(torch.bool).to(self.device)

        self_attentions = []
        for layer in self.layers:
            output, self_attention = layer(output, attention_mask)
            self_attentions.append(self_attention)

        return output, self_attentions


class Transformer(nn.Module):
    def __init__(self, config, d_input, d_target, max_length, d_model, n_heads, n_layers, d_feedforward, d_key, d_value, dropout=0.1):
        super(Transformer, self).__init__()

        self.config = config
        self.device = config['device']

        self.encoder = Encoder(config=config,
                               d_input=d_input,
                               d_model=d_model,
                               n_heads=n_heads,
                               n_layers=n_layers,
                               d_feedforward=d_feedforward,
                               d_key=d_key,
                               d_value=d_value,
                               dropout=dropout
                               )

        self.ff = nn.Linear(d_model * max_length, d_target)

    def forward(self, input):

        input.to(self.device)

        output, self_attention = self.encoder(input)
        output = torch.reshape(output, (output.shape[0], -1))
        result = self.ff(output)

        return result
