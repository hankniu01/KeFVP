import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        
        ########## Seasonal ###########
        self.seasonal_enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed_type, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            mode='seasonal'
        )
        
        ########## Trend ###########
        self.trend_enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed_type, configs.freq,
                                                  configs.dropout)
        
        # Encoder
        self.trend_encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            mode='trend'
        )
        
        
    def forward(self, x_enc, x_mark_enc,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        seasonal_init, trend_init = self.decomp(x_enc)
        
        ########## Seasonal ###########
        seasonal_enc_out = self.seasonal_enc_embedding(seasonal_init, seasonal_init)
        seasonal_enc_out, sea_attns = self.seasonal_encoder(seasonal_enc_out, attn_mask=enc_self_mask)
       
        ########## Trend ###########
        trend_enc_out = self.trend_enc_embedding(trend_init, trend_init)
        trend_enc_out, tre_attns = self.seasonal_encoder(trend_enc_out, attn_mask=enc_self_mask)
        
        
        if self.output_attention:
            return seasonal_enc_out, trend_enc_out, sea_attns, tre_attns
        else:
            return seasonal_enc_out, trend_enc_out
