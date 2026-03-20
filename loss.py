import torch
import torch.nn as nn
import torch.nn.functional as F


class HSPLoss(nn.Module):
    """
    HSP 多任务联合优化损失
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 权重参数
        self.lambda_mr = nn.Parameter(torch.tensor(1.0))
        self.lambda_be = nn.Parameter(torch.tensor(1.0))
        self.lambda_bm = nn.Parameter(torch.tensor(1.0))
        self.lambda_cscn = nn.Parameter(torch.tensor(1.0))
        self.lambda_meta = nn.Parameter(torch.tensor(1.0))
        
        # 基础损失
        self.margin_ranking = nn.MarginRankingLoss(margin=args.margin)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, outputs, batch_data):
        """
        计算所有损失项
        """
        loss_dict = {}
        
        # 1. Embedding Alignment Loss (Margin Ranking)
        loss_dict['L_mr'] = self._mr_loss(
            outputs['Z_eafc'],
            batch_data['train_links'],
            batch_data.get('negatives')
        )
        
        # 2. Entailment Losses
        loss_dict['L_be'] = self._be_loss(
            outputs['nsp_logits'],
            batch_data['labels']
        )
        loss_dict['L_bm'] = self._bm_loss(
            outputs['nsp_logits'],
            batch_data['train_links'],
            batch_data.get('negatives')
        )
        
        # 3. Cross-Modal Contrastive Loss
        loss_dict['L_cscn'] = self._cscn_loss(
            outputs['S_v'],
            outputs['S_t']
        )
        
        # 4. Meta-Learning Loss (如有)
        loss_dict['L_meta'] = self._meta_loss(
            outputs,
            batch_data.get('meta_tasks')
        )
        
        # 加权总和
        total_loss = (
            self.lambda_mr * loss_dict['L_mr'] +
            self.lambda_be * loss_dict['L_be'] +
            self.lambda_bm * loss_dict['L_bm'] +
            self.lambda_cscn * loss_dict['L_cscn'] +
            self.lambda_meta * loss_dict['L_meta']
        )
        
        return total_loss, loss_dict
    
    def _mr_loss(self, Z_eafc, train_links, negatives=None):
        """Embedding Alignment Loss"""
        if negatives is None:
            # 简化版本：使用 batch 内负采样
            pos_pairs = Z_eafc[train_links[:, 0]] - Z_eafc[train_links[:, 1]]
            neg_idx = torch.randperm(Z_eafc.size(0))[:train_links.size(0)]
            neg_pairs = Z_eafc[train_links[:, 0]] - Z_eafc[neg_idx]
            
            pos_dist = torch.norm(pos_pairs, dim=-1)
            neg_dist = torch.norm(neg_pairs, dim=-1)
            
            target = torch.ones_like(pos_dist)
            loss = self.margin_ranking(neg_dist, pos_dist, target)
        else:
            loss = torch.tensor(0.0, device=Z_eafc.device)
        
        return loss
    
    def _be_loss(self, nsp_logits, labels):
        """Binary Entailment Loss"""
        if nsp_logits is None:
            return torch.tensor(0.0, device=labels.device)
        return self.bce_loss(nsp_logits.squeeze(), labels.float())
    
    def _bm_loss(self, nsp_logits, train_links, negatives=None):
        """Binary Margin Loss"""
        if nsp_logits is None:
            return torch.tensor(0.0, device=train_links.device)
        
        pos_logits = nsp_logits[train_links[:, 0]]
        if negatives is not None:
            neg_logits = nsp_logits[negatives]
            target = torch.ones_like(pos_logits)
            loss = self.margin_ranking(neg_logits, pos_logits, target)
        else:
            loss = torch.tensor(0.0, device=train_links.device)
        
        return loss
    
    def _cscn_loss(self, S_v, S_t):
        """Cross-Modal Contrastive Loss"""
        S_v = F.normalize(S_v, dim=-1)
        S_t = F.normalize(S_t, dim=-1)
        
        sim_matrix = S_v @ S_t.T / 0.07
        batch_size = sim_matrix.shape[0]
        labels = torch.arange(batch_size, device=S_v.device)
        
        return self.cross_entropy(sim_matrix, labels)
    
    def _meta_loss(self, outputs, meta_tasks):
        """Meta-Learning Loss"""
        if meta_tasks is None:
            return torch.tensor(0.0, device=outputs['Z_eafc'].device)
        
        # 简化实现
        return torch.tensor(0.0, device=outputs['Z_eafc'].device)