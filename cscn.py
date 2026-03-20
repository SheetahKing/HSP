import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalSemanticCalibration(nn.Module):
    """
    CSCN: Cross-Modal Semantic Calibration Network
    """
    
    def __init__(self, args, visual_dim=768, text_dim=768, shared_dim=512):
        super().__init__()
        self.args = args
        self.shared_dim = shared_dim
        
        # 视觉投影
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU()
        )
        
        # 文本投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU()
        )
        
        # 双线性交互矩阵
        self.interaction_matrix = nn.Parameter(torch.randn(shared_dim, shared_dim))
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(shared_dim * 2 + 1, shared_dim),
            nn.Sigmoid()
        )
        
        # 对比学习温度参数
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, F_v, T_e):
        """
        F_v: [batch_size, visual_dim]
        T_e: [batch_size, text_dim]
        """
        # 投影到共享语义空间
        S_v = self.visual_proj(F_v)
        S_t = self.text_proj(T_e)
        
        # 双线性交互
        I_vt = torch.sum(S_v * (S_t @ self.interaction_matrix), dim=-1, keepdim=True)
        
        # 门控融合
        gate_input = torch.cat([S_v, S_t, I_vt], dim=-1)
        g = self.gate_network(gate_input)
        
        S_shared = g * S_v + (1 - g) * S_t
        
        return S_v, S_t, S_shared
    
    def contrastive_loss(self, S_v, S_t):
        """
        InfoNCE 对比损失
        """
        S_v = F.normalize(S_v, dim=-1)
        S_t = F.normalize(S_t, dim=-1)
        
        sim_matrix = S_v @ S_t.T / self.temperature
        batch_size = sim_matrix.shape[0]
        
        labels = torch.arange(batch_size, device=S_v.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss