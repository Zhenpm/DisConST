import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import scanpy as sc
import torch
import ot
import random
def refine_label(adata, radius=50, key='mclust', ref_label='ref_label'):
    '''
    radius:     refining radius
    key:        the predicted label saved in adata.obs[key]
    ref_label:  after refining, save the new label in adata.obs[ref_label]
    '''
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    adata.obs[ref_label] = np.array(new_type)
    
    return adata

def generate_csl_graph(num_nodes, kner):
    '''
    Generate a graph randomly.
    num_nodes:  the number of nodes in the graph
    kner:       the number of edges per node
    '''
    num_edges = num_nodes * kner
    edge_index = np.zeros((2, num_edges), dtype=int)
    edge_set = set()

    for i in range(num_nodes):
        indices = random.sample(range(num_nodes), kner)
        for j in indices:
            if j != i and (i, j) not in edge_set:
                edge_set.add((i, j))
                edge_index[:, len(edge_set)-1] = [i, j]

    sorted_index = np.argsort(edge_index[0])
    edge_index = edge_index[:, sorted_index]
    mask = (edge_index[0] == 0) & (edge_index[1] == 0)
    mask = ~mask
    edge_index = edge_index[:,mask]
    print("Negative spots selection completed!")
    return edge_index

def generate_spatial_graph(adata, radius=None, knears=None, self_loops:bool=False):
    '''
    radius: the radius when selecting neighors
    knears: the number of neighbors
    method: the method to generate graph: radius or knn
    '''
    #knears=0
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['row', 'col']
    if (radius):
        nbrs = NearestNeighbors(radius = radius).fit(coor)
        distance, indices = nbrs.radius_neighbors(coor, return_distance=True)
    elif (knears):
        nbrs = NearestNeighbors(n_neighbors=knears).fit(coor)
        distance, indices = nbrs.kneighbors(coor)
    else:
        raise ValueError("method Error:radius or knn!")
    edge_list = []
    if(self_loops):
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                # donnot add self-loop edge to the graph 
                edge_list.append([i, indices[i][j]])  
    else:
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                # donnot add self-loop edge to the graph 
                if (i != indices[i][j]):
                    edge_list.append([i, indices[i][j]])  
    edge_list = np.array(edge_list)
    #np.savetxt('edge_list2.txt', edge_list, fmt='%d')
    #new_edge_list = np.vstack((edge_list, np.flip(edge_list, axis=1)))
    #edge_list = new_edge_list
    print('graph includs edges:',len(edge_list))

    edge_list = edge_list.transpose()
    adata.uns['graph']=edge_list
    print("Graph construction completed!")

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STboth', random_seed=666, domain_obs='label'):

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    try:
        mclust_res = np.array(res[-2])
    except Exception as e:
        #print("using concat data")
        res = rmclust(adata.obsm['STconcat'], num_cluster, modelNames)
        mclust_res = np.array(res[-2])

    adata.obs[domain_obs] = mclust_res
    adata.obs[domain_obs] = adata.obs[domain_obs].astype('int')
    adata.obs[domain_obs] = adata.obs[domain_obs].astype('category')

    return adata

def cal_size_factor(adata, data='both', cell_obsm='cell_type', sf_gene='size_factor_g', sf_cell='size_factor_c'):
    '''
    cell_obsm:  the cell type proportion data saves in adata.obsm[cell_obsm]
    sf_gene:    save the size factor of gene in adata.obs[sf_gene]
    sf_cell:    save the size factor of cell type proportion in adata.obs[sf_cell]
    '''
    if data=='both':
        adata.obs['total_counts_c'] = adata.obsm['cell_type'].sum(axis=1)
        total_counts_c = adata.obs['total_counts_c'].values
        median_size_factor_c = np.median(total_counts_c)
        size_factors_c = total_counts_c / median_size_factor_c
        adata.obs[sf_cell] = size_factors_c
        adata.obs[sf_cell] = adata.obs[sf_cell].astype('float32')

    adata.obs['total_counts_g'] = adata.X.sum(axis=1)
    total_counts_g = adata.obs['total_counts_g'].values
    median_size_factor_g = np.median(total_counts_g)
    size_factors_g = total_counts_g / median_size_factor_g
    adata.obs[sf_gene] = size_factors_g
    adata.obs[sf_gene] = adata.obs[sf_gene].astype('float32')
    print("Size factor calculation completed!")
    return adata

def preprocess(adata, data='both', cell_obsm='cell_type', sf_gene='size_factor_g', sf_cell='size_factor_c',n_top_genes=3000, target_sum_gene=1e4, target_sum_cell=200):
    '''
    cell_obsm:  the cell type proportion data saves in adata.obsm[cell_obsm]
    sf_gene:    save the size factor of gene in adata.obs[sf_gene]
    sf_cell:    save the size factor of cell type proportion in adata.obs[sf_cell]
    '''
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    adata=cal_size_factor(adata, data, cell_obsm=cell_obsm, sf_gene=sf_gene, sf_cell=sf_cell)
    sc.pp.normalize_total(adata, target_sum=target_sum_gene)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.log1p(adata)
    if data=='both':
        total_counts_c = adata.obs['total_counts_c'].values
        for i in range(adata.obsm[cell_obsm].shape[0]):
            adata.obsm[cell_obsm][i,:]=adata.obsm[cell_obsm][i,:]/total_counts_c[i]*target_sum_cell
        adata.obsm[cell_obsm]=np.log(adata.obsm[cell_obsm]+1)
        adata.obsm[cell_obsm] = adata.obsm[cell_obsm].astype('float32')
    print("Data preprocessing completed!")
    return adata
