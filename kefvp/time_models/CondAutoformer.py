import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from condlayers.Embed import DataEmbedding, DataEmbedding_wo_pos
from condlayers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer, CondAutoCorrelationLayer, CondAutoCorrelation, Each_BATCH_AutoCorrelation
from condlayers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, CondDecoderLayer
import math
import numpy as np
from transformers_model.modules import TransformerBlock


class CondAutoformer(nn.Module):
    """
    """
    def __init__(self, configs, output_attention=False):
        super(CondAutoformer, self).__init__()
        # self.pred_len = 1    # 3
        self.pred_len = configs.duration
        self.output_attention = output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=self.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                CondDecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    CondAutoCorrelationLayer(
                        CondAutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    pred_len = self.pred_len
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
        self.cond_embed = nn.Linear(configs.text_indim, configs.d_model)
        nn.init.xavier_normal_(self.cond_embed.weight)
        
        self.cond_att = nn.ModuleList([
            TransformerBlock(configs.d_model),
            TransformerBlock(configs.d_model)
        ])
        

    def forward(self, x_enc, cond, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init, mean], dim=1)
        seasonal_init = torch.cat([seasonal_init, zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, seasonal_init)
        
        
        cond = self.cond_embed(cond)
        
        
        mask_idx = (cond[:, :, 0] != 0).float().unsqueeze(-1)
        
        attcond = self.cond_att[0](cond, mask_idx)
        attcond = self.cond_att[1](attcond, mask_idx)
        cond = attcond.mean(1).unsqueeze(1).repeat(1, self.pred_len, 1)

        dec_out[:, -self.pred_len:, :] = dec_out[:, -self.pred_len:, :] + cond.detach()
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init, cond=cond)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :].mean(1)  # [B, L, D]
            # return cond.mean(1)