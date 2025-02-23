import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn import GCN
from model.attn_gcn import SpatialAttentionGCN
from torch_geometric.nn import GCNConv
from utils.tools import load_norm_Laplacian
import os
import math


class GCN_(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(GCN_, self).__init__()
        self.gcnConv = GCN(c_in, d_model, dropout=dropout)
        self.gcnConv.apply(self.init_para)

    @staticmethod
    def init_para(m):
        if isinstance(m, GCNConv):
            nn.init.kaiming_normal_(m.lin.weight, mode='fan_in', nonlinearity='leaky_relu')
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        y = self.gcnConv(x)
        return y

class SATGCN(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, args=None):
        super(SATGCN, self).__init__()
        adj, self.N = load_norm_Laplacian(os.path.join(args.root_path, args.adj_path), args.device)
        self.SAT_GCN = SpatialAttentionGCN(c_in, d_model, num_of_vertices=self.N, adj=adj, dropout=dropout,
                                           device=args.device, K=args.K)
        self.SAT_GCN.apply(self.init_para)

    @staticmethod
    def init_para(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        y = self.SAT_GCN(x)
        return y


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular')  # 变换的是特征维度

        def parameter_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.tokenConv.apply(parameter_init)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x  # [32, 96, 12] -> [32, 96, 512]


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros((max_len, d_model), requires_grad=False).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)  # 5000，1
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # position * div_term
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)  # 4，512

    def forward(self, x):
        return self.embed(x)

class Fuse_attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(Fuse_attention, self).__init__()
        self.heads = n_heads
        d_values = d_model // n_heads
        self.qt_l = nn.Linear(d_model, d_values * n_heads)
        self.kt_k = nn.Linear(d_model, d_values * n_heads)
        self.vt_v = nn.Linear(d_model, d_values * n_heads)
        self.qx_l = nn.Linear(d_model, d_values * n_heads)
        self.kx_k = nn.Linear(d_model, d_values * n_heads)
        self.vx_v = nn.Linear(d_model, d_values * n_heads)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_values * n_heads, d_model)

    def forward(self, x, t):
        b, t_len, _ = t.shape
        _, x_len, _ = x.shape
        qt = self.qt_l(t).view(b, t_len, self.heads, -1).transpose(2, 1)
        kt = self.kt_k(t).view(b, t_len, self.heads, -1).transpose(2, 1)
        vt = self.vt_v(t).view(b, t_len, self.heads, -1).transpose(2, 1)
        qx = self.qx_l(x).view(b, x_len, self.heads, -1).transpose(2, 1)
        kx = self.kx_k(x).view(b, x_len, self.heads, -1).transpose(2, 1)
        vx = self.vx_v(x).view(b, x_len, self.heads, -1).transpose(2, 1)
        scoret = torch.matmul(qt, kx.permute(0, 1, 3, 2))/math.sqrt(t_len)
        scorex = torch.matmul(qx, kt.permute(0, 1, 3, 2))/math.sqrt(x_len)
        attnt = self.dropout(F.softmax(scoret, dim=-1))
        attnx = self.dropout(F.softmax(scorex, dim=-1))
        v_x = torch.matmul(attnx, vx).contiguous().view(b, x_len, -1)
        v_t = torch.matmul(attnt, vt).contiguous().view(b, x_len, -1)
        return self.dropout(F.leaky_relu(self.out(v_x + v_t)))

class Spatial_Encoder(nn.Module):
    def __init__(self, d_model, c_in=1, embed_type='fixed', freq='t', dropout=0.1, args=None):
        super(Spatial_Encoder, self).__init__()

        # self.scalar_Projection = TokenEmbedding(c_in=c_in, d_model=d_model)

        self.GCN_EProjection = SATGCN(c_in=args.n_fin, d_model=d_model, dropout=dropout, args=args)
        self.GCN_DProjection = SATGCN(c_in=args.n_fin, d_model=d_model, dropout=dropout, args=args)
        # self.GCN_EProjection = GCN_(c_in=args.seq_len, d_model=d_model, dropout=dropout)
        # self.GCN_DProjection = GCN_(c_in=args.pred_len + args.label_len, d_model=d_model, dropout=dropout)

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) \
            if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)
        self.node_embedding = nn.Embedding(self.GCN_EProjection.N, d_model)
        self.enc_len = args.seq_len
        # self.dec_len = args.label_len + args.pred_len
        # self.fuse1 = nn.Conv1d(self.enc_len * 3, self.enc_len, kernel_size=3, padding=1, padding_mode='circular')
        # self.fuse2 = nn.Conv1d(self.dec_len * 3, self.dec_len, kernel_size=3, padding=1, padding_mode='circular')
        # self.fuse_attn = Fuse_attention(d_model, args.n_heads, dropout)

    def forward(self, x, x_time):
        # x = self.scalar_Projection(x) + self.position_embedding(x) + self.temporal_embedding(x_time)
        # x = self.position_embedding(x) + self.temporal_embedding(x_time) + self.GCN_Projection(x)
        # x = self.scalar_Projection(x) +  self.position_embedding(x) + self.temporal_embedding(x_time) + self.GCN_Projection(x)
        b, c, _ = x.shape
        A = self.position_embedding(x)
        B = self.temporal_embedding(x_time)  # 32， 96，512
        C = self.GCN_EProjection(x) if c == self.enc_len else self.GCN_DProjection(x)
        return A+B+C+self.node_embedding

        # _, s, d = B.shape
        # y = B.expand(b, s, d)
        # fuse_attn = self.fuse_attn(D + y, C)
        # y = torch.cat((y, C, D), dim=-2)
        # y = self.fuse1(y) if c == self.enc_len else self.fuse2(y)
        # return self.dropout(y)
