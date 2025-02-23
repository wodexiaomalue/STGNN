import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 根据任务设置
edge_index = torch.tensor([[...],
                           [...]])


class GCN(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(c_in, 2 * d_model)
        self.conv2 = GCNConv(2 * d_model, c_in)
        self.dropout = nn.Dropout(p=dropout)
        self.edge = edge_index.cuda()
        node = self.edge.max() + 1
        self.linear = nn.Linear(node, d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv2(self.dropout(F.leaky_relu(self.conv1(x, self.edge))), self.edge)
        x = x.permute(0, 2, 1)
        return self.linear(x)
