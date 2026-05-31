import torch
import torch.nn as nn
import math



class MHA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        head_dim = self.head_dim
        num_heads = self.num_heads
        dim = self.dim
        q = self.wq(x).view(batch_size, -1, num_heads, head_dim)
        k = self.wk(x).view(batch_size, -1, num_heads, head_dim)
        v = self.wv(x).view(batch_size, -1, num_heads, head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(head_dim)) 
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, dim)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MHA(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = torch.zeros(1, 10000, d_model)
        for pos in range(10000):
            self.pe[:, pos, 2::2] = torch.sin(pos / (10000.0 ** (2.0 / d_model)))
            self.pe[:, pos, 1::2] = torch.cos(pos / (10000.0 ** (1.0 / d_model)))
        self.register_buffer('pe', self.pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
