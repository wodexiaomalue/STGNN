import torch
import torch.nn as nn
import torch.nn.functional as F
from model.spatial_encoder import Spatial_Encoder
from model.attention import CrossFullAttention, ProbAttention, AttentionLayer
from model.encoder import Encoder, EncoderLayer, ConvLayer
from model.decoder import Decoder, DecoderLayer
from model.s_de import series_decomp

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, out_len, args,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='t', activation='gelu',
                 distil=True, mix=True, output_attention=False):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.label_len = args.label_len
        self.attn_type = attn
        self.output_attention = output_attention

        kernel_size = args.moving_avg
        self.decomp = series_decomp(kernel_size)

        # spatial encoder
        self.L_SE = Spatial_Encoder(enc_in, d_model, embed, freq, dropout, args)
        self.R_SE = Spatial_Encoder(dec_in, d_model, embed, freq, dropout, args)

        # Attention
        LSAttn = ProbAttention if attn == 'prob' else CrossFullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(LSAttn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False, flag='encoder'),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    moving_avg=25
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
            e_layers=e_layers,
            args=args
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(LSAttn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix, flag='decoder'),
                    AttentionLayer(CrossFullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False, flag='decoder'),
                    d_model,
                    c_out,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    moving_avg=25
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out * args.vex_dim, bias=True)  # 512 12

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #print(x_enc.shape)  # torch.Size([32, 96, 12])
        #print(x_mark_enc.shape)  # torch.Size([32, 96, 4])
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        dec_out = self.R_SE(seasonal_init, x_mark_dec)
        enc_out = self.L_SE (x_enc, x_mark_enc)  # 32， 96， 512

        # enc_out = torch.cat([dec_out,enc_out],dim=-2)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # 32 48 512，

        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        seasonal_part = self.projection(seasonal_part)
        dec_out = trend_part + seasonal_part

        # dec_out = self.projection(dec_out)
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


