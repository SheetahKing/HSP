import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class EntityTypeAdaptiveFusion(nn.Module):
    """
    EAFC: Entity-Type Adaptive Fusion Controller
    类型感知融合 + MAML 元学习优化
    """
    
    def __init__(self, args, type_num=10, type_dim=128, shared_dim=512):
        super().__init__()
        self.args = args
        self.type_num = type_num
        self.type_dim = type_dim
        self.shared_dim = shared_dim
        
        # 类型嵌入
        self.type_embedding = nn.Embedding(type_num, type_dim)
        
        # 类型预测器（用于未知类型）
        self.type_predictor = nn.Sequential(
            nn.Linear(shared_dim, type_dim * 2),
            nn.GELU(),
            nn.Linear(type_dim * 2, type_num)
        )
        
        # 融合权重生成器
        self.fusion_gate = nn.Sequential(
            nn.Linear(type_dim, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, 2)
        )
        
        # Meta-learning 参数
        self.meta_lr = 0.01
        
    def forward(self, S_v, S_t, type_ids=None):
        """
        S_v: [batch_size, shared_dim]
        S_t: [batch_size, shared_dim]
        type_ids: [batch_size] 或 None
        """
        # 获取类型嵌入
        if type_ids is not None:
            type_emb = self.type_embedding(type_ids)
        else:
            # 未知类型：使用预测分布
            type_probs = F.softmax(self.type_predictor(S_v), dim=-1)
            type_emb = torch.matmul(
                type_probs,
                self.type_embedding.weight
            )
        
        # 生成融合权重
        gate_logits = self.fusion_gate(type_emb)
        weights = F.softmax(gate_logits, dim=-1)
        
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        
        # 自适应融合
        Z_eafc = alpha * S_v + beta * S_t
        
        return Z_eafc, weights
    
    def meta_adapt(self, support_data, query_data, inner_steps=1):
        """
        MAML 元学习适配
        support_data: (S_v, S_t, type_ids, labels)
        """
        # Inner Loop
        adapted_params = copy.deepcopy(list(self.fusion_gate.parameters()))
        
        for _ in range(inner_steps):
            S_v, S_t, type_ids, labels = support_data
            Z_eafc, _ = self.forward_with_params(S_v, S_t, type_ids, adapted_params)
            
            loss = F.mse_loss(Z_eafc, labels)
            grads = torch.autograd.grad(loss, adapted_params, create_graph=True)
            
            adapted_params = [
                p - self.meta_lr * g for p, g in zip(adapted_params, grads)
            ]
        
        # Outer Loop
        S_v, S_t, type_ids, labels = query_data
        Z_eafc, _ = self.forward_with_params(S_v, S_t, type_ids, adapted_params)
        query_loss = F.mse_loss(Z_eafc, labels)
        
        return query_loss
    
    def forward_with_params(self, S_v, S_t, type_ids, params):
        """使用指定参数进行前向传播（用于 meta-learning）"""
        type_emb = self.type_embedding(type_ids)
        
        # 手动应用参数
        gate_out = F.linear(type_emb, params[0], params[1])
        gate_out = F.gelu(gate_out)
        gate_logits = F.linear(gate_out, params[2], params[3])
        weights = F.softmax(gate_logits, dim=-1)
        
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        Z_eafc = alpha * S_v + beta * S_t
        
        return Z_eafc, weights