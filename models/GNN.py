import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GNN(nn.Module):
    """HVAC异常检测图神经网络 - 纯粹基于变量关联"""
    
    def __init__(self, args):
        super(GNN, self).__init__()
        
        self.use_attention = args.use_attention
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.num_class
        self.dropout = args.dropout
        self.num_node_features = 1
        
        if self.use_attention:
            # 使用图注意力网络
            self.conv1 = GATConv(self.num_node_features, self.hidden_dim, heads=4, dropout=self.dropout)
            self.conv2 = GATConv(self.hidden_dim * 4, self.hidden_dim, heads=4, dropout=self.dropout)
            self.conv3 = GATConv(self.hidden_dim * 4, self.hidden_dim, heads=1, dropout=self.dropout)
        else:
            # 使用图卷积网络
            self.conv1 = GCNConv(self.num_node_features, self.hidden_dim)
            self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 4, self.num_classes)
        )
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GNN前向传播
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # 图级别的池化（将节点特征聚合为图特征）
        x = global_mean_pool(x, batch)
        
        # 分类
        x = self.classifier(x)
        
        return x 

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.model = GNN(args)

    def forward(self, data):
        return self.model(data) 