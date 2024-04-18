import torch
import torch.nn.functional as F
from torch import nn
import math

class SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

    def forward(self, query, key, value, mask=None):
        B, _, N = query.size()

        # 将输入的Q、K、V拆分为多头
        query = query.view(B, self.num_heads, N, self.head_dim)
        key = key.view(B, self.num_heads, N, self.head_dim)
        value = value.view(B, self.num_heads, N, self.head_dim)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 如果提供了mask，将其应用到注意力分数上
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 计算加权的V
        output = torch.matmul(attn_weights, value)

        # 合并多头
        output = output.contiguous().view(B, -1, N)

        return output