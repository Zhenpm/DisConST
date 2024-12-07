import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch_scatter
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from .gat_conv import GATConv
import torch.nn.init as init

def mean_act(x):
    return torch.exp(x).clamp(1e-5, 1e6)
def disp_act(x):
    return F.softplus(x).clamp(1e-4, 1e4)

class DisConST(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(DisConST, self).__init__()
        self.num_layers = len(hidden_dims)
        if self.num_layers == 2:
            [in_dim, out_dim] = hidden_dims
            self.conv1 = GATConv(in_dim, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
            self.conv2 = GATConv(out_dim, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        else:
            [in_dim, num_hidden, out_dim] = hidden_dims
            self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                                dropout=0, add_self_loops=False, bias=False)
            self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                                dropout=0, add_self_loops=False, bias=False)
            self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                                dropout=0, add_self_loops=False, bias=False)
            self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                                dropout=0, add_self_loops=False, bias=False)                 

    def forward(self, features, edge_index, CL_graph):
        if self.num_layers == 2:
            h2 = F.elu(self.conv1(features, edge_index))
            self.conv2.lin_src.data = self.conv1.lin_src.transpose(0, 1)
            self.conv2.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
            h4 = F.elu(self.conv2(h2, edge_index, attention=True,
                            tied_attention=self.conv1.attentions))
            h4_p=h4
            h4_d=h4
        else:                   
            h1 = F.elu(self.conv1(features, edge_index))
            h2 = self.conv2(h1, edge_index, attention=False)
            self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
            self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
            self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
            self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
            h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                                tied_attention=self.conv1.attentions))
            h4 = self.conv4(h3, edge_index, attention=False)
            h4_p = self.conv4(h3, edge_index, attention=False)
            h4_d = self.conv4(h3, edge_index, attention=False)
        #h_4 = self.linear(h3) 
        sigmoid = nn.Sigmoid()   
        pi = sigmoid(h4_p)
        disp = disp_act(h4_d)
        mean = mean_act(h4)

        output = [mean, disp, pi]  # assuming output, disp, and pi are tensors
        h_pos = self.CSL(h2, edge_index)
        h_neg = self.CSL(h2, CL_graph)
        return h2, h_pos, h_neg, output, h4
    def CSL(self, h, edge_index):
        node_features = h.index_select(0, edge_index[1])
        # 使用平均聚合将邻居节点特征进行聚合
        h_agg = torch_scatter.scatter_mean(node_features, edge_index[0], dim=0)
        return h_agg

    def CSL_loss(self, h_anc, h_pos, h_neg, margin = 1.0):
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        csl_loss = triplet_loss(h_anc, h_pos, h_neg)
        return csl_loss

class AE_last(torch.nn.Module):
    def __init__(self,latent_dims):
        super(AE_last, self).__init__()
        [in_dim_m, latent_dim_m] = latent_dims
        self.encoder =  nn.Sequential(
            nn.Linear(in_dim_m,latent_dim_m),
            nn.ELU()
        )
        self.decoder =  nn.Sequential(
            nn.Linear(latent_dim_m,in_dim_m),
            nn.ELU()
        )

    def forward(self,features):
        h=self.encoder(features)
        h_rec=self.decoder(h)
        
        return h, h_rec

'''
        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        
        # 添加编码器层
        for i in range(num_layers-1):
            self.encoder_convs.append(GATConv(hidden_dims[i], hidden_dims[i+1], heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False))
        
        # 添加解码器层
        for i in range(num_layers-1, 0, -1):
            self.decoder_convs.append(GATConv(hidden_dims[i], hidden_dims[i-1], heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False))
encoder_outputs = []
        # 编码
        for conv in self.encoder_convs:
            x = F.elu(conv(x, edge_index))
            encoder_outputs.append(x)  # 保存每一层编码器的输出
        
        # 解码
        for i, conv in enumerate(self.decoder_convs):
            # 传递参数
            conv.lin_src.data = self.encoder_convs[num_layers-1-i].lin_src.transpose(0, 1)
            conv.lin_dst.data = self.encoder_convs[num_layers-1-i].lin_dst.transpose(0, 1)
            x = F.elu(conv(x, edge_index))
            '''

    
