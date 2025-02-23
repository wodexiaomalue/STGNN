import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from utils.tools import multi_locality

class CrossFullAttention(nn.Module):  # Decoder 中 cross用的attention
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(CrossFullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):

        B, L, H, E = queries.shape  # 32 72 8 64
        _, S, _, D = values.shape  # 32 48 8 64
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, S, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [32, 8, 72, 48] # 72 和 48 每个做
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):  # Encoder和Decoder 中 都用的self attention
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape  # 32 8 96 64
        _, _, L_Q, _ = Q.shape  # 32 8 96 64

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K,
                                          E)

        index_sample = torch.randint(L_K, (L_Q, sample_k))  # 构建96*25的随机数,96 个 k要和那些q做点积
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2),
                                  K_sample.transpose(-2, -1)).squeeze()  # 96个Q和25个K之间的关系 32 8 96 1 25 .squeeze
        q_uniform_distribution = torch.tensor(1/L_K)
        p_k_q = F.softmax(Q_K_sample.detach()/sqrt(E))
        M = self.JS_divergence(p_k_q, q_uniform_distribution)
        # find the Top_k query with sparisty measurement
        M_top = M.topk(n_top, sorted=False)[1]
        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape  # 32 8 96 64

        if not self.mask_flag:
            V_sum = V.mean(dim=-2) # 96，64 在96这个维度求均值有64个
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # 先把96个V都用均值来替换
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape # 32 6 96 64

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            #print(scores.shape)
        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores) # 32 8 25 96

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)# 对25个有Q的更新V，其余的没变还是均值  torch.matmul(attn, V) 32 8 25 64

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            #print(attns.shape)
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape  # 32， 96， 8， 64
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)  # 32，8，96，64
        keys = keys.transpose(2, 1)  # 32，8，96，64
        values = values.transpose(2, 1)  # 32，8，96，64
        Sample_k = self.factor * np.ceil(np.log(L_K)).astype(
            'int').item()  # c*ln(L_k)
        top_q = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        Sample_k = Sample_k if Sample_k < L_K else L_K
        top_q = top_q if top_q < L_Q else L_Q
        # 32 8 25 96
        scores_top, index = self._prob_QK(queries, keys, sample_k=Sample_k, n_top=top_q)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # get the context 处理V
        context = self._get_initial_context(values, L_Q)  # 考虑 对 96-25 个q没做处理，处理V
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        # context 更新后的V  32 8 96 64，attn 是分数
        return context.transpose(2, 1).contiguous(), attn  # 32 96 8 64

    def JS_divergence(self, P, Q):
        M = 0.5 * (P + Q) # 32 8 36 20
        Q = Q.expand(P.shape).to(P.device)
        kl_PM = self.KL_divergence(P, M)
        kl_QM = self.KL_divergence(Q, M)
        return 0.5 * (kl_PM + kl_QM)

    def KL_divergence(self, P, Q):
        kl = torch.div(P, Q)
        kl = torch.log(kl).permute(0, 1, 3, 2)
        kl = torch.matmul(P, kl)
        return kl.sum(-1)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False, flag=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)  # 512/8
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # 512, 64*8
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)

        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)  # 64*8, 512
        self.n_heads = n_heads
        self.mix = mix

        self.kernel_size = 3
        self.padding = (self.kernel_size - 1)
        # self.query_conv1Ds = nn.Conv1d(in_channels=d_model, out_channels=d_keys * n_heads, kernel_size=self.kernel_size, padding=self.padding)
        # self.key_conv1Ds = nn.Conv1d(in_channels=d_model, out_channels=d_keys * n_heads, kernel_size=self.kernel_size, padding=self.padding)

        self.flag = flag
        if flag == 'encoder':
            self.kernel_size = 3
            self.padding = (self.kernel_size - 1)//2
            self.query_conv1Ds = nn.Conv1d(in_channels=d_model, out_channels=d_keys * n_heads, kernel_size=self.kernel_size, padding=self.padding)
            self.key_conv1Ds = nn.Conv1d(in_channels=d_model, out_channels=d_keys * n_heads, kernel_size=self.kernel_size, padding=self.padding)

        if flag == 'decoder':
            if isinstance(self.inner_attention, ProbAttention):
                self.kernel_size = 3
                self.padding = (self.kernel_size - 1)
                self.query_conv1Ds = nn.Conv1d(in_channels=d_model, out_channels=d_keys * n_heads, kernel_size=self.kernel_size, padding=self.padding)
                self.key_conv1Ds = nn.Conv1d(in_channels=d_model, out_channels=d_keys * n_heads, kernel_size=self.kernel_size, padding=self.padding)
            else:
                self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # 512, 64*8
                self.key_projection = nn.Linear(d_model, d_keys * n_heads)

        self.q_k = multi_locality([3, 5], d_model, d_keys * n_heads)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape  # 32，96，512
        _, S, _ = keys.shape  # 32，96，512
        H = self.n_heads  # 8

        # if self.flag == 'encoder':
        #     queries = self.query_conv1Ds(queries.permute(0, 2, 1)).permute(0, 2, 1).view(B, L, H, -1)
        #     keys = self.key_conv1Ds(keys.permute(0, 2, 1)).permute(0, 2, 1).view(B, S, H, -1)
        # else:
        #     if isinstance(self.inner_attention, ProbAttention):
        #         queries = self.query_conv1Ds(queries.permute(0, 2, 1))[:, :, :-2].permute(0, 2, 1).view(B, L, H, -1)
        #         keys = self.key_conv1Ds(keys.permute(0, 2, 1))[:, :, :-2].permute(0, 2, 1).view(B, S, H, -1)
        #     else:
        #         queries = self.query_projection(queries).view(B, L, H, -1)
        #         keys = self.key_projection(keys).view(B, S, H, -1)

        # queries = self.query_conv1Ds(queries.permute(0, 2, 1))[:, :, :-self.padding].permute(0, 2, 1).view(B, L, H, -1)
        # keys = self.key_conv1Ds(keys.permute(0, 2, 1))[:, :, :-self.padding].permute(0, 2, 1).view(B, S, H, -1)

        if L==S:
            queries, keys = self.q_k(queries, keys)
            queries = queries.view(B, L, H, -1)
            keys = keys.view(B, L, H, -1)
        else:
            queries = self.query_projection(queries).view(B, L, H, -1)
            keys = self.key_projection(keys).view(B, S, H, -1)

        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


'''
# non-multilocal
        if L == S:
            # queries, keys = self.q_k(queries, keys)
            # queries = queries.view(B, L, H, -1)
            # keys = keys.view(B, L, H, -1)
            queries = self.query_projection(queries).view(B, L, H, -1)
            keys = self.key_projection(keys).view(B, S, H, -1)
        else:
            # queries = self.query_conv1Ds(queries.permute(0, 2, 1))[:, :, :-2].permute(0, 2, 1).view(B, L, H, -1)
            # keys = self.key_conv1Ds(keys.permute(0, 2, 1))[:, :, :-2].permute(0, 2, 1).view(B, S, H, -1)
            queries = self.query_projection(queries).view(B, L, H, -1)
            keys = self.key_projection(keys).view(B, S, H, -1)
            
        values = self.value_projection(values).view(B, S, H, -1)  # 32，96，8，64
'''
