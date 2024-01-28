import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers_model import GraphConvolution, MultiHeadAttention, GraphChannelAttLayer, KumaLearner, AsymmetricGraphConvolution

class Net(torch.nn.Module):
    def __init__(self, data):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 200)
        self.conv2 = GCNConv(200, 100)
        self.hidden3 = nn.Linear(100,1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        gc1 = self.conv1(x, edge_index)
        gc2 = F.relu(gc1)
        gc3 = F.dropout(gc2, training=self.training)
        gc4 = self.conv2(gc3, edge_index)
        x = self.hidden3(gc4)

        return gc1, gc2, gc3, gc4, x
    

class Graphlearner(torch.nn.Module):
    def __init__(self, opt, in_dim=200, mem_dim=100, attention_heads=2, num_layers = 2):
        super(Graphlearner, self).__init__()
        self.opt = opt
        self.attention_heads = attention_heads
        self.num_layers = num_layers

        self.trans_ratio = nn.Linear(768, in_dim)

        self.graphlearner = nn.ModuleList([
            MultiHeadAttention(opt, attention_heads, in_dim, mem_dim, combination=self.opt.combination),
            MultiHeadAttention(opt, attention_heads, mem_dim, mem_dim, combination=self.opt.combination)])

        self.price_graphlearner = nn.ModuleList([
            MultiHeadAttention(opt, attention_heads, in_dim, mem_dim, combination=self.opt.combination),
            MultiHeadAttention(opt, attention_heads, in_dim, mem_dim, combination=self.opt.combination)])

        # self.sea_price_graphlearner = nn.ModuleList([
        #     MultiHeadAttention(opt, attention_heads, in_dim, mem_dim, combination=self.opt.combination),
        #     MultiHeadAttention(opt, attention_heads, in_dim, mem_dim, combination=self.opt.combination)])

        # self.earning_ratio_learner = nn.ModuleList([
        #     MultiHeadAttention(opt, attention_heads, in_dim, mem_dim, combination=self.opt.combination, asymmetric=True),
        #     MultiHeadAttention(opt, attention_heads, mem_dim, mem_dim, combination=self.opt.combination, asymmetric=True)])
        
        self.earning_ratio_learner = nn.ModuleList([KumaLearner(in_dim), 
                            KumaLearner(mem_dim)])

        self.combine = GraphChannelAttLayer(num_channel=2)
        
        self.gcns_text = nn.ModuleList([GraphConvolution(in_dim, mem_dim, layers=1),
                    GraphConvolution(mem_dim, mem_dim, layers=1)])
        
        self.gcns_price = nn.ModuleList([GraphConvolution(in_dim, mem_dim, layers=1),
                    GraphConvolution(mem_dim, mem_dim, layers=1)])

        # self.gcns_sea_price = nn.ModuleList([GraphConvolution(in_dim, mem_dim, layers=1),
        #             GraphConvolution(mem_dim, mem_dim, layers=1)])


        # self.gcns_ratio_earning = nn.ModuleList([AsymmetricGraphConvolution(in_dim, mem_dim, layers=1),
        #             AsymmetricGraphConvolution(mem_dim, in_dim, layers=1)])

        # self.gcns_raw = nn.ModuleList([GraphConvolution(in_dim, mem_dim, layers=1),
        #             GraphConvolution(mem_dim, mem_dim, layers=1)])
        
        # self.affine_text = nn.Parameter(torch.Tensor(mem_dim, mem_dim))
        # self.affine_price = nn.Parameter(torch.Tensor(mem_dim, mem_dim))
        
        self.hidden = nn.Linear(mem_dim*2, 1)
    
    def att_graphlearner(self, gidx, x, priceX, earningX, ratioX, adj):
        
        adj_text = self.graphlearner[gidx](x, x)
        
        adj_price = self.price_graphlearner[gidx](priceX, priceX)

        # adj_sea_price = self.sea_price_graphlearner[gidx](Sea_priceX, Sea_priceX)
        
        return adj_text, adj_price

    def forward(self, x_, priceX, ratioX, Sea_priceX, G_stock_earning, G_stock_earning_ratio, G_earning_ratio):
        
        # ratioX = self.trans_ratio(ratioX)
        # G_text_stock_earning_ratio, G_price_stock_earning_ratio  = G_stock_earning_ratio.clone(), G_stock_earning_ratio.clone()
        # allX =  torch.cat([x, ratioX], dim=0)
        # allX_text, allX_price = allX, allX

        earningX = x_[self.opt.num_stock:, :]

        # for j in range(self.num_layers):
        #     adj_earning_ratio = self.earning_ratio_learner[j](earningX, ratioX)   # num_earning * num_ratios
        #     adj_earning_ratio_ = adj_earning_ratio * G_earning_ratio
        #     earningX, ratioX = self.gcns_ratio_earning[j](earningX, ratioX, adj_earning_ratio_)
        
        # x_[self.opt.num_stock:, :] = earningX.detach()

        # x = x_[self.opt.num_stock:, :]
        # x_price = x_[self.opt.num_stock:, :]
        x = earningX
        x_price = earningX
        # x_sea_price = earningX

        priceX = priceX[self.opt.num_stock:, :]
        Sea_priceX = Sea_priceX[self.opt.num_stock:, :]

        for i in range(self.num_layers):
            # if i != 0:
                # x = allX_text[:self.opt.num_stock_earning, :]
                # earningX = allX_text[self.opt.num_stock:self.opt.num_stock_earning, :]
                # ratioX = allX_text[self.opt.num_stock_earning:, :]
            ########## learning earning to ratio###############
            adj_text, adj_price = self.att_graphlearner(i, x, priceX, earningX, ratioX, G_stock_earning)

            ############learning stock to earning###############
            # Gs = {}
            # adjs = {'text': adj_text, 'price': adj_price}
            # for k, g in {'text': G_text_stock_earning_ratio, 'price': G_price_stock_earning_ratio}.items():
            #     g[self.opt.num_stock:self.opt.num_stock_earning, self.opt.num_stock_earning:] = adj_earning_ratio
            #     g[self.opt.num_stock_earning:, self.opt.num_stock:self.opt.num_stock_earning] = adj_earning_ratio.transpose(1,0)
            #     g[:self.opt.num_stock_earning, :self.opt.num_stock_earning] = adjs[k]
            #     Gs[k] = g.clone()
            # G_text, G_price = Gs['text'], Gs['price']

            x = self.gcns_text[i](x, adj_text)
            x_price = self.gcns_price[i](x_price, adj_price)
            # x_sea_price = self.gcns_sea_price[i](x_sea_price, adj_sea_price)
        
        # gc1_raw = self.gcns_raw[1](gc0_raw, adj)
        
        # att_text = F.softmax(torch.mm(gc1_price@self.affine_price, gc1_text.transpose(1, 0)), dim=-1)
        # att_price = F.softmax(torch.mm(gc1_text@self.affine_text, gc1_price.transpose(1, 0)), dim=-1)
        # gc1_text, gc1_price = torch.mm(att_text, gc1_text), torch.mm(att_price, gc1_price)
        # gc1_text, gc1_price = F.dropout(gc1_text, 0.3, True), F.dropout(gc1_price, 0.3, True)

        # gc1 = torch.cat([x, x_price, x_sea_price], dim=-1)
        gc1 = torch.cat([x, x_price], dim=-1)

        x = self.hidden(gc1)
        # x = x[:self.opt.num_stock_earning, :]
        gc1 = torch.cat([x_[:self.opt.num_stock, :].detach(), gc1.detach()], dim=0)

        return x, gc1