import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from hvpe import HierarchicalVisualEnhancement, VisualPrefixGenerator
from cscn import CrossModalSemanticCalibration
from eafc import EntityTypeAdaptiveFusion


class HSPModel(nn.Module):
    """
    HSP: Hierarchical Semantic Preservation Model
    整合 HVPE + CSCN + EAFC + PLM
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # PLM Backbone
        self.plm = AutoModel.from_pretrained(args.plm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
        self.plm_dim = self.plm.config.hidden_size
        
        # HVPE 模块
        self.hvpe = HierarchicalVisualEnhancement(
            args,
            input_dim=args.img_dim,
            hidden_dim=self.plm_dim
        )
        self.prefix_generator = VisualPrefixGenerator(
            args,
            visual_dim=self.plm_dim,
            prefix_len=args.prefix_len,
            plm_dim=self.plm_dim
        )
        
        # CSCN 模块
        self.cscn = CrossModalSemanticCalibration(
            args,
            visual_dim=self.plm_dim,
            text_dim=self.plm_dim,
            shared_dim=args.shared_dim
        )
        
        # EAFC 模块
        self.eafc = EntityTypeAdaptiveFusion(
            args,
            type_num=args.type_num,
            type_dim=args.type_dim,
            shared_dim=args.shared_dim
        )
        
        # 推理头
        self.nsp_head = nn.Linear(self.plm_dim, 2)
        self.mlm_head = nn.Linear(self.plm_dim, self.tokenizer.vocab_size)
        
        # 融合权重参数
        self.gamma = nn.Parameter(torch.tensor(0.5))
        
        # 文本表示的注意力查询向量
        self.text_query = nn.Parameter(torch.randn(self.plm_dim))
        
    def forward(self, batch_data, mode='train'):
        """
        前向传播
        
        Args:
            batch_data: dict, 包含以下键:
                - 'img_features': [batch, img_dim] 图像特征
                - 'text_ids': [batch, seq_len] 文本 token IDs
                - 'text_mask': [batch, seq_len] 注意力掩码
                - 'type_ids': [batch] 实体类型 IDs
                - 'triples': list 三元组列表
                - 'train_links': [N, 2] 训练对齐链接
            mode: str, 'train' 或 'inference'
        
        Returns:
            dict: 包含所有中间和最终表示
        """
        # 1. HVPE: 视觉特征增强
        F_v, visual_weights = self.hvpe(batch_data['img_features'])
        
        # 2. 生成视觉前缀
        prefix_tokens = self.prefix_generator(F_v)
        
        # 3. PLM 编码（带视觉前缀）
        text_emb = self.plm(
            input_ids=batch_data['text_ids'],
            attention_mask=batch_data['text_mask'],
            inputs_embeds=self._merge_prefix(prefix_tokens, batch_data['text_ids'])
        ).last_hidden_state
        
        # 4. 文本表示（Attention Pooling）
        T_e = self._attention_pooling(text_emb, batch_data['text_mask'])
        
        # 5. CSCN: 跨模态语义校准
        S_v, S_t, S_shared = self.cscn(F_v, T_e)
        
        # 6. EAFC: 类型自适应融合
        Z_eafc, fusion_weights = self.eafc(S_v, S_t, batch_data.get('type_ids'))
        
        # 7. 推理头（仅在推理模式）
        nsp_logits = None
        mlm_logits = None
        if mode == 'inference':
            cls_emb = text_emb[:, 0, :]
            nsp_logits = self.nsp_head(cls_emb)
        
        return {
            'F_v': F_v,
            'T_e': T_e,
            'S_v': S_v,
            'S_t': S_t,
            'S_shared': S_shared,
            'Z_eafc': Z_eafc,
            'visual_weights': visual_weights,
            'fusion_weights': fusion_weights,
            'nsp_logits': nsp_logits,
            'mlm_logits': mlm_logits,
            'text_emb': text_emb
        }
    
    def _merge_prefix(self, prefix_tokens, text_ids):
        """
        合并视觉前缀和文本嵌入
        
        Args:
            prefix_tokens: [batch, prefix_len, plm_dim] 视觉前缀
            text_ids: [batch, seq_len] 文本 token IDs
        
        Returns:
            [batch, prefix_len + seq_len, plm_dim] 合并后的嵌入
        """
        text_emb = self.plm.get_input_embeddings()(text_ids)
        merged = torch.cat([prefix_tokens, text_emb], dim=1)
        return merged
    
    def _attention_pooling(self, hidden_states, attention_mask):
        """
        文本表示的 Attention Pooling
        
        Args:
            hidden_states: [batch, seq_len, dim] PLM 输出
            attention_mask: [batch, seq_len] 注意力掩码
        
        Returns:
            [batch, dim] 池化后的文本表示
        """
        # 计算注意力分数
        scores = torch.softmax(hidden_states @ self.text_query.unsqueeze(-1), dim=1)
        
        # 应用掩码
        scores = scores * attention_mask.unsqueeze(-1)
        scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
        
        # 加权求和
        pooled = torch.sum(scores * hidden_states, dim=1)
        return pooled
    
    def compute_embedding_similarity(self, Z1, Z2):
        """
        计算嵌入相似度（用于候选检索）
        
        Args:
            Z1: [batch1, dim]
            Z2: [batch2, dim]
        
        Returns:
            [batch1, batch2] 相似度矩阵
        """
        Z1 = F.normalize(Z1, dim=-1)
        Z2 = F.normalize(Z2, dim=-1)
        return torch.matmul(Z1, Z2.T)
    
    def entailment_reasoning(self, batch_data, reason_type='NSP'):
        """
        双重蕴含推理
        
        Args:
            batch_data: dict 输入数据
            reason_type: str, 'NSP' 或 'MLM'
        
        Returns:
            final_score: 最终匹配分数
            outputs: 模型输出字典
        """
        outputs = self.forward(batch_data, mode='inference')
        
        if reason_type == 'NSP':
            # NSP-based reasoning
            align_prob = torch.softmax(outputs['nsp_logits'], dim=-1)[:, 1]
        else:
            # MLM-based reasoning
            cls_emb = outputs['text_emb'][:, 0, :]
            align_prob = torch.sigmoid(self.nsp_head(cls_emb).squeeze())
        
        # 结合嵌入相似度和蕴含概率
        # 注意：这里假设 batch 中实体对是成对排列的
        if outputs['Z_eafc'].size(0) >= 2:
            emb_sim = self.compute_embedding_similarity(
                outputs['Z_eafc'][::2],
                outputs['Z_eafc'][1::2]
            )
            if emb_sim.dim() == 2:
                emb_sim = torch.diag(emb_sim)
        else:
            emb_sim = torch.ones(1, device=outputs['Z_eafc'].device)
        
        final_score = self.gamma * emb_sim + (1 - self.gamma) * align_prob
        
        return final_score, outputs
    
    def get_entity_embeddings(self, batch_data):
        """
        获取实体的最终嵌入表示（用于推理阶段的批量检索）
        
        Args:
            batch_data: dict 输入数据
        
        Returns:
            Z_eafc: [batch, dim] 融合后的实体嵌入
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch_data, mode='inference')
            return F.normalize(outputs['Z_eafc'], dim=-1)
    
    def compute_alignment_loss(self, Z_eafc, train_links, temperature=0.05):
        """
        计算对齐损失（InfoNCE）
        
        Args:
            Z_eafc: [batch, dim] 实体嵌入
            train_links: [N, 2] 对齐链接
            temperature: float 温度参数
        
        Returns:
            loss: 对齐损失值
        """
        emb_left = Z_eafc[train_links[:, 0]]
        emb_right = Z_eafc[train_links[:, 1]]
        
        emb_left = F.normalize(emb_left, dim=-1)
        emb_right = F.normalize(emb_right, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = emb_left @ emb_right.T / temperature
        
        # InfoNCE 损失
        batch_size = sim_matrix.shape[0]
        labels = torch.arange(batch_size, device=Z_eafc.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def save_pretrained(self, save_path):
        """
        保存模型权重
        
        Args:
            save_path: str 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'args': self.args
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_pretrained(self, load_path, device='cpu'):
        """
        加载预训练权重
        
        Args:
            load_path: str 加载路径
            device: str 设备
        """
        checkpoint = torch.load(load_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {load_path}")


class HSPModelWithGNN(HSPModel):
    """
    HSP 模型扩展版本：添加图神经网络编码结构信息
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # GNN 编码器（可选）
        if hasattr(args, 'use_gnn') and args.use_gnn:
            self.use_gnn = True
            self.gnn_encoder = GNNEncoder(
                node_dim=self.plm_dim,
                hidden_dim=args.gnn_hidden_dim,
                num_layers=args.gnn_num_layers
            )
        else:
            self.use_gnn = False
    
    def forward(self, batch_data, mode='train'):
        """扩展的前向传播，支持图结构编码"""
        # 基础前向传播
        outputs = super().forward(batch_data, mode)
        
        # 可选：添加 GNN 编码
        if self.use_gnn and 'adj_matrix' in batch_data:
            gnn_emb = self.gnn_encoder(
                outputs['Z_eafc'],
                batch_data['adj_matrix']
            )
            outputs['Z_eafc'] = gnn_emb
        
        return outputs


class GNNEncoder(nn.Module):
    """
    简单的图神经网络编码器
    """
    
    def __init__(self, node_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for i in range(num_layers)
        ])
        
        self.final_proj = nn.Linear(hidden_dim, node_dim)
    
    def forward(self, node_features, adj_matrix):
        """
        Args:
            node_features: [num_nodes, dim]
            adj_matrix: sparse tensor [num_nodes, num_nodes]
        """
        x = node_features
        
        for layer in self.layers:
            # 图卷积：聚合邻居信息
            neighbor_agg = torch.sparse.mm(adj_matrix, x)
            x = layer(x + neighbor_agg)
        
        x = self.final_proj(x)
        return x