import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.s_de import series_decomp

class MultiScaleFuse(nn.Module):
    def __init__(self, layer_num=None, length=None):
        super(MultiScaleFuse, self).__init__()

        # self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # pooling_list = []
        # if (l_num-2) > 0 :
        #     for i in range(l_num-1):
        #         list = []
        #         for j in range(l_num-2, 0, -1):
        #             list.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        Lout_layer = []
        Lin = length
        for i in range(layer_num - 1):
            Lout = (Lin + 1) // 2
            Lout_layer.append(Lout)
            Lin = Lout
        Last_out = Lout_layer[-1]
        assert Last_out >= 1
        self.model_list = nn.ModuleList()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        if layer_num < 3:
            self.model_list.append(nn.Conv1d(in_channels=Last_out,
                                             out_channels=Last_out,
                                             kernel_size=3,
                                             padding=padding,
                                             padding_mode='circular'))
        else:
            self.size = layer_num - 2
            for i in range(layer_num - 2):
                self.model_list.append(nn.Conv1d(in_channels=Lout_layer[i],
                                                 out_channels=Last_out,
                                                 kernel_size=3,
                                                 padding=padding,
                                                 padding_mode='circular'))
            self.model_list.append(nn.Conv1d(in_channels=Last_out,
                                             out_channels=Last_out,
                                             kernel_size=3,
                                             padding=padding,
                                             padding_mode='circular'))
        self.activation = F.gelu
        self.fuse = nn.Conv1d(in_channels=Last_out * layer_num, out_channels=Last_out, kernel_size=3, padding=padding, padding_mode='circular')
        self.norm = nn.BatchNorm1d(Last_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, layer_x):
        y = x
        if len(self.model_list) is not 1:
            # y = y + self.activation(self.model_list[-1](layer_x[-1]))
            out = self.model_list[-1](layer_x[-1])
            y = torch.cat((y, out), dim=-2)
            for i in range(self.size):
                out = self.activation(self.norm(self.model_list[i](layer_x[i])))
                # y = y + out
                y = torch.cat((y, out), dim=-2)
            return self.activation(self.fuse(y))
        else:
            y = y + self.activation(self.norm(self.model_list[-1](layer_x[-1])))
            return y


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')  # 512,512
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))  # 512-512 的1dconv
        x = self.norm(x)
        x = self.activation(x)  #
        x = self.maxPool(x)  # 32 512 48
        x = x.transpose(1, 2)  # 32 48 512
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", moving_avg=25):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)  # 512,2048
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.decomp = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]

        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )  #
        x = x + self.dropout(new_x)
        x, _ = self.decomp(x)
        y = x = self.norm1(x)  # [32, 96, 512]
        # print(y.transpose(-1,1).shape) #
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # conv1 512 2048
        # print(y.shape) # [32, 2048, 96]
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # conv2 2048 512
        # print(y.shape) # [32, 96, 512]
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, e_layers=None, args=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.M_f = MultiScaleFuse(e_layers, args.seq_len)

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        multiscale_value = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):  #
                x, attn = attn_layer(x,
                                     attn_mask=attn_mask)  # q k v
                x = conv_layer(x)  # 32，96，512-> 32 48 512
                multiscale_value.append(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)  #
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        x = self.M_f(x, multiscale_value)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
