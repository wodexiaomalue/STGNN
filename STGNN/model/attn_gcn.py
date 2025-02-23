import torch
import torch.nn as nn
#from torch_geometric.nn import GCNConv
import torch.nn.functional as F
# from utils.tools import load_mask
import math
from utils.tools import cheby


class Spatial_Attention(nn.Module):
    def __init__(self, device, num_of_vertices, dropout=0.1):
        super(Spatial_Attention, self).__init__()
        self.K = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(device))
        self.V = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(device))
        self.b = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(device))
        self.dropout = nn.Dropout(p=dropout)
        self.init_parameter()

    def forward(self, x):
        B, T, N, fin = x.shape
        x = x.reshape(-1, N, fin)
        product = torch.matmul(x, self.dropout(torch.matmul(self.K, x).transpose(1, 2)))
        score = torch.matmul(self.V, torch.tanh(product + self.b))
        S_Attn = F.softmax(score, dim=-1)
        return S_Attn.reshape((B, T, N, N)).contiguous()

    def init_parameter(self):
        nn.init.xavier_normal_(self.K)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.b)


class SpatialAttentionGCN(nn.Module):
    def __init__(self, c_in, d_model, num_of_vertices, adj, dropout=0.1, device='', K=3):
        super(SpatialAttentionGCN, self).__init__()
        self.SAT = Spatial_Attention(device, num_of_vertices, dropout)
        self.project = nn.Linear(c_in, d_model//2)
        self.Theta1_ = nn.Linear(d_model//2, d_model // 2)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_ = nn.Linear(num_of_vertices * (d_model//2), d_model)
        self.register_buffer('adj', adj)

        #-------ChebnetII--------#
        self.K = K
        self.temp = nn.Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x):
        B, L, N = x.shape
        x = x.unsqueeze(-1)
        x = self.project(x)

        coe_tmp = F.relu(self.temp)  # 伽马系数
        coe=coe_tmp.clone()
        # 构建 切比雪夫插值系数
        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)
        Tx_0 = x
        Tx_1 = torch.matmul(self.adj, x)
        A_x = coe[0] / 2 * Tx_0 + coe[1] * Tx_1
        for i in range(2, self.K+1):
            Tx_2 = torch.matmul(self.adj, Tx_1)
            Tx_2=2*Tx_2-Tx_0
            A_x=A_x+coe[i]*Tx_2 # 求和
            Tx_0,Tx_1 = Tx_1, Tx_2

        SA = self.SAT(x)
        S_x = torch.matmul(SA, x)
        # x = torch.cat((A_x, S_x), dim=-1)
        x = A_x + S_x
        x = self.dropout(F.leaky_relu(self.Theta1_(x))).view(B, L, -1)
        return self.dropout(F.leaky_relu(self.linear_(x)))


