import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class HierarchicalVisualEnhancement(nn.Module):
    """
    HVPE: Hierarchical Visual Feature Enhancement
    三路径 Transformer + 动态权重门控融合
    """
    
    def __init__(self, args, input_dim=2048, hidden_dim=768, num_heads=8, num_layers=3):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 初始投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Local Pathway - Window-based Attention
        self.local_transformer = WindowedTransformer(
            hidden_dim, num_heads, num_layers, window_size=7
        )
        
        # Global Pathway - Full Self-Attention
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=num_layers
        )
        
        # Semantic Pathway - MLP for semantic aggregation
        self.semantic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Dynamic Gating Network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # Attention Pooling
        self.attention_pool = AttentionPooling(hidden_dim)
        
    def forward(self, img_features):
        """
        img_features: [batch_size, input_dim] 或 [batch_size, H*W, input_dim]
        """
        # 初始投影
        x = self.input_proj(img_features)
        x = self.layer_norm(x)
        
        # 三路径处理
        F_local = self.local_transformer(x)
        F_global = self.global_transformer(x)
        F_semantic = self.semantic_mlp(F_global)
        
        # Pooling
        pool_local = self.attention_pool(F_local)
        pool_global = self.attention_pool(F_global)
        pool_semantic = self.attention_pool(F_semantic)
        
        # Gating Network
        concat_features = torch.cat([pool_local, pool_global, pool_semantic], dim=-1)
        gate_logits = self.gate_network(concat_features)
        weights = F.softmax(gate_logits, dim=-1)  # [batch_size, 3]
        
        # 加权融合
        pooled_features = torch.stack([pool_local, pool_global, pool_semantic], dim=1)
        F_v = torch.sum(weights.unsqueeze(-1) * pooled_features, dim=1)
        
        return F_v, weights


class WindowedTransformer(nn.Module):
    """Window-based Self-Attention for Local Pathway"""
    
    def __init__(self, hidden_dim, num_heads, num_layers, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.layers = nn.ModuleList([
            WindowedAttentionLayer(hidden_dim, num_heads, window_size)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class WindowedAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, window_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: [seq_len, batch_size, hidden_dim]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.squeeze(0) if x.size(0) == 1 else x


class AttentionPooling(nn.Module):
    """Learnable Attention Pooling"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            return x
        # x: [batch_size, seq_len, hidden_dim]
        attn_scores = torch.softmax(
            torch.matmul(x, self.query.unsqueeze(-1)).squeeze(-1),
            dim=1
        )
        pooled = torch.sum(attn_scores.unsqueeze(-1) * x, dim=1)
        return pooled


class VisualPrefixGenerator(nn.Module):
    """
    VMN: Vision Mapping Network
    将视觉特征映射为可学习的前缀 token 序列
    """
    
    def __init__(self, args, visual_dim=768, prefix_len=10, plm_dim=768):
        super().__init__()
        self.prefix_len = prefix_len
        self.plm_dim = plm_dim
        
        self.vmn = nn.Sequential(
            nn.Linear(visual_dim, plm_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(plm_dim * 2, prefix_len * plm_dim)
        )
        
    def forward(self, F_v):
        """
        F_v: [batch_size, visual_dim]
        return: [batch_size, prefix_len, plm_dim]
        """
        prefix = self.vmn(F_v)
        prefix = prefix.view(-1, self.prefix_len, self.plm_dim)
        return prefix