import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from .utils import generate_csl_graph
from .DisConST import  AE_last, DisConST
from .loss import ZINB, ZINBLoss
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import networkx as nx
import random
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import DataParallel as DPG
from torch.nn.parallel import DistributedDataParallel as DDP


def train_last(adata,
        latent_dim = 32,                                                        # the dimensions of encoder
        embedding_gene='STgene',                                                # save the embedding of gene in adata.obsm[embedding_gene]
        embedding_cell='STcell',                                                # save the embedding of cell type proportion in adata.obsm[embedding_cell]
        n_epochs=1000,                                                          # epoch times
        lr=0.001,                                                               # learning rate
        embedding_both='STboth',                                                # save the embedding of gene in adata.obsm[embedding_both]
        weight_cell=1,                                                          # the weight of cell type proportion
        gradient_clipping=5.,                                                   # gradient clipping for the optimizer
        weight_decay=0.0001,                                                    # weight decay for the optimizer  
        show =True,                                                             # show some infomation
        random_seed=666,                                                        # random seed   
        save_embedding = None,                                                  # save embedding to the appointed path     
        save_model=None,                                                        # save model parameters to the appointed path 
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # the chosen device
):
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    stgene_matrix=adata.obsm[embedding_gene]
    stcell_matrix=adata.obsm[embedding_cell] * weight_cell
    assert stgene_matrix.shape[0]==stcell_matrix.shape[0],"the matrix has the same number of rows"
    combined_matrix = np.hstack((stgene_matrix, stcell_matrix))
    adata.obsm['STconcat'] = combined_matrix
    data = torch.FloatTensor(adata.obsm['STconcat'])
    data = data.to(device)
    model = AE_last(latent_dims=[data.shape[1],latent_dim]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('training for fusion...')
    for epoch in tqdm(range(0, n_epochs)):
        model.train()
        optimizer.zero_grad()
        z, rec=model(data)
        loss_mse =F.mse_loss(data, rec)
        loss=loss_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    with torch.no_grad():
        z,rec=model(data)
    model.eval()
    adata.obsm[embedding_both] = z.cpu().detach().numpy()
    # save cell embedding 
    if (isinstance(save_embedding, str)):
        np.save(save_embedding, adata.obsm[embedding_both], allow_pickle=True, fix_imports=True)
        print('Successfully save final embedding at {}.'.format(save_embedding)) if (show) else None

    if (isinstance(save_model, str)):
        torch.save(model, save_model)
        print('Successfully export final model at {}.'.format(save_model)) if (show) else None

    
    return adata

def train_DisConST(
        adata, 
        data='both',                                                            # 'gene'/'both' using only RNA-seq data or both RNA-seq and cell type proportion data
        hidden_dims_gene = [3000, 512, 30],                                     # the dimensions of encoder (gene)
        size_factors_gene = None,                                               # size factors of gene count matrix
        hidden_dims_cell = [120, 64, 16],                                       # the dimensions of encoder (cell type proportion)
        size_factors_cell = None,                                               # size factors of gene count matrix
        n_epochs=1000,                                                          # epoch times
        lr=0.001,                                                               # learning rate
        embedding_gene='STgene',                                                # save the embedding of gene in adata.obsm[embedding_gene]
        embedding_cell='STcell',                                                # save the embedding of cell type proportion in adata.obsm[embedding_cell]
        gradient_clipping=5.,                                                   # gradient clipping for the optimizer
        weight_decay=0.0001,                                                    # weight decay for the optimizer
        random_seed=666,                                                        # random seed
        save_embedding_gene = None,                                             # save embedding to the appointed path (gene)
        save_model_gene = None,                                                 # save model parameters to the appointed path (gene)
        save_embedding_cell = None,                                             # save embedding to the appointed path (cell)
        save_model_cell = None,                                                 # save model parameters to the appointed path (cell)
        kner = 5,                                                               # the number of nerigbors of graph for contrastive learning
        zinb_l_g = 1,                                                           # the weight of ZINB loss (gene)
        csl_l_g = 0.2,                                                          # the weight of contrastive learning loss (gene) 
        zinb_l_c = 1,                                                           # the weight of ZINB loss (cell)
        csl_l_c = 0.2,                                                          # the weight of contrastive learning loss (cell)
        show = True,                                                            # show some infomation
        latent_dim_last = 32,                                                   # the dimensions of encoder
        n_epochs_last=1000,                                                     # epoch times
        lr_last=0.001,                                                          # learning rate
        embedding_both='STboth',                                                # save the embedding of gene in adata.obsm[embedding_both]
        weight_cell=1,                                                          # the weight of cell type proportion
        gradient_clipping_last=5.,                                              # gradient clipping for the optimizer
        weight_decay_last=0.0001,                                               # weight decay for the optimizer  
        save_embedding_last = None,                                             # save embedding to the appointed path     
        save_model_last = None,                                                 # save model parameters to the appointed path 
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # the chosen device 
):
    assert ((data != 'both') or (data != 'gene')), 'ERROR:The parameter data can only be both or gene!'
    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    graph = adata.uns['graph']

    node_count = adata.n_obs
    graphcsl = generate_csl_graph(node_count, kner)
    edge_CSL = torch.LongTensor(np.array([graphcsl[0], graphcsl[1]])).to(device)
    graph = torch.LongTensor(np.array([graph[0],graph[1]])).to(device)
    data_gene = torch.FloatTensor(adata.X.todense()).to(device)#
    size_factors_gene=torch.FloatTensor(size_factors_gene).to(device)
    if data=='both':
        data_cell = torch.FloatTensor(adata.obsm['cell_type']).to(device) 
        size_factors_cell=torch.FloatTensor(size_factors_cell).to(device)

    model_g = DisConST(hidden_dims=hidden_dims_gene).to(device)
    #model_g=nn.DataParallel(model_g,device_ids=[0, 1])
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr, weight_decay=weight_decay)

    print('training RNA-seq data...')
    for epoch in tqdm(range(0, n_epochs)):
        model_g.train()
        optimizer_g.zero_grad()
        z_g, z_pos_g, z_neg_g, output_g,h4_g = model_g(data_gene, graph, edge_CSL)
        
        mean_g = output_g[0]
        disp_g = output_g[1]
        pi_g = output_g[2]
        #zinb_g = ZINB(pi_g, scale_factor=size_factors_gene, theta=disp_g)
        #zinb_loss_g = zinb_g.loss(data_gene, mean_g)
        zinb_g = ZINBLoss()
        zinb_loss_g = zinb_g(x=data_gene, mean = mean_g, disp = disp_g, pi = pi_g, scale_factor = size_factors_gene)
        csl_loss_g = model_g.CSL_loss(z_g, z_pos_g, z_neg_g)
        loss_g = zinb_l_g * zinb_loss_g + csl_l_g * csl_loss_g
        
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(model_g.parameters(), gradient_clipping)
        optimizer_g.step() 
    with torch.no_grad():
        z_g, _,_,_,_ = model_g(data_gene, graph, edge_CSL)
    model_g.eval()
    adata.obsm[embedding_gene] = z_g.cpu().detach().numpy()
    adata.obsm['predicted_gene'] = h4_g.cpu().detach().numpy()

    if (data == 'both'):
        model_c = DisConST(hidden_dims=hidden_dims_cell).to(device)
        optimizer_c = torch.optim.Adam(model_c.parameters(), lr=lr, weight_decay=weight_decay)
        print('training cell type proprotion data...')
        for epoch in tqdm(range(0, n_epochs)):
            model_c.train()
            optimizer_c.zero_grad()
            z_c, z_pos_c, z_neg_c, output_c,h4_c = model_c(data_cell, graph, edge_CSL)

            mean_c = output_c[0]
            disp_c = output_c[1]
            pi_c = output_c[2]

            #zinb_c = ZINB(pi_c, scale_factor=size_factors_cell, theta=disp_c)
            #zinb_loss_c = zinb_c.loss(data_cell,mean_c)
            zinb_c = ZINBLoss()
            zinb_loss_c = zinb_c(x=data_cell, mean = mean_c, disp = disp_c, pi = pi_c, scale_factor = size_factors_cell)
            csl_loss_c = model_c.CSL_loss(z_c, z_pos_c, z_neg_c)
            loss_c = zinb_l_c * zinb_loss_c + csl_l_c * csl_loss_c

            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(model_c.parameters(), gradient_clipping)
            optimizer_c.step() 
        with torch.no_grad():
            z_c, _, _ ,_,_= model_c(data_cell, graph, edge_CSL)
        model_c.eval()
        adata.obsm[embedding_cell] = z_c.cpu().detach().numpy()
        adata.obsm['predicted_cell'] = h4_c.cpu().detach().numpy()

    # save gene embedding 
    if (isinstance(save_embedding_gene, str)):
        np.save(save_embedding_gene, adata.obsm[embedding_gene], allow_pickle=True, fix_imports=True)
        print('Successfully save RNA-seq embedding at {}.'.format(save_embedding_gene)) if (show) else None

    # save gene model parameters
    if (isinstance(save_model_gene, str)):
        torch.save(model_g, save_model_gene)
        print('Successfully export RNA-seq model at {}.'.format(save_model_gene)) if (show) else None
    if (data == 'both'):
        # save cell embedding 
        if (isinstance(save_embedding_cell, str)):
            np.save(save_embedding_cell, adata.obsm[embedding_cell], allow_pickle=True, fix_imports=True)
            print('Successfully save cell type embedding at {}.'.format(save_embedding_cell)) if (show) else None

        # save cell model parameters
        if (isinstance(save_model_cell, str)):
            torch.save(model_c, save_model_cell)
            print('Successfully export cell type model at {}.'.format(save_model_cell)) if (show) else None
            
    if (data == 'both'):
        adata = train_last(adata, latent_dim=latent_dim_last, random_seed=random_seed, embedding_gene=embedding_gene,
         embedding_cell=embedding_cell, n_epochs=n_epochs_last,lr=lr_last,embedding_both=embedding_both,weight_cell = weight_cell,
          gradient_clipping = gradient_clipping_last,weight_decay = weight_decay_last,show = show, save_embedding = save_embedding_last, 
          save_model = save_model_last, device=device)
    return adata