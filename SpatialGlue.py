

# ------------------------------ Model CODE ----------------------

import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3):
    """
    Construct neighbor graphs, including feature graph and spatial graph.
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
       n_neighbors=6
    # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2

    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2

    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

    return data

def pca(adata, use_reps=None, n_comps=10):

    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray())
       else:
          feat_pca = pca.fit_transform(adata.X)

    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):

    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode= "connectivity", metric="correlation", include_self=False):

    """Constructing feature neighbor graph according to expresss profiles"""

    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2=kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    #print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""

    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)

    adj_spatial_omics1 = adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()

    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2>1, 1, adj_spatial_omics2)

    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1) # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)

    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())

    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1>1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2>1, 1, adj_feature_omics2)

    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1) # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)

    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }

    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def fix_seed(seed):
    #seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder_overall(Module):
    """
    Overall encoder.
    """
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act

        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)

        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)

    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2):
        # graph1
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)

        # graph2
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        # within-modality attention aggregation layer
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)

        # between-modality attention aggregation layer
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        # consistency encoding
        emb_latent_omics1_across_recon = self.encoder_omics2(self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2), adj_spatial_omics2)
        emb_latent_omics2_across_recon = self.encoder_omics1(self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1), adj_spatial_omics1)

        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'emb_latent_omics1_across_recon':emb_latent_omics1_across_recon,
                   'emb_latent_omics2_across_recon':emb_latent_omics2_across_recon,
                   'alpha_omics1':alpha_omics1,
                   'alpha_omics2':alpha_omics2,
                   'alpha':alpha_omics_1_2
                   }

        return results


class Encoder(Module):
    """
    Modality-specific GNN encoder.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class Decoder(Module):
    """
    Modality-specific GNN decoder.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class AttentionLayer(Module):
    """
    Attention layer.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha
import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter
    import pandas as pd

    np.random.seed(random_seed)

    robjects.r.library("mclust")

    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)

    rmclust = robjects.r["Mclust"]

    # Get the data
    X = np.array(adata.obsm[used_obsm], dtype=np.float64)

    print("Input shape:", X.shape)

    # Convert to DataFrame with column names
    df = pd.DataFrame(
        X,
        columns=[f'PC{i+1}' for i in range(X.shape[1])]
    )

    with localconverter(default_converter + pandas2ri.converter):
        res = rmclust(
            df,
            G=num_cluster,
            modelNames=modelNames
        )

    if hasattr(res, 'rx2'):
        mclust_res = np.array(res.rx2('classification'))
    elif hasattr(res, 'getbyname'):
        mclust_res = np.array(res.getbyname('classification'))
    else:
        mclust_res = np.array(res['classification'])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int').astype('str')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    return adata


def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20, random_seed=2020):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.
    random_seed : int, optional
        Random seed for clustering. The default is 2020.

    Returns
    -------
    None.

    """

    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == 'mclust':
       if use_pca:
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters, random_seed=random_seed)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters, random_seed=random_seed)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca:
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca:
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!."

    return res

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values

  """
  import pandas as pd

  df = pd.DataFrame(columns=[modality1, modality2, 'label'])
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2)

  plt.tight_layout(w_pad=0.05)
  plt.show()
import torch
from tqdm import tqdm
import torch.nn.functional as F


class Train_SpatialGlue:
    def __init__(self,
        data,
        datatype = 'SPOTS',
        device= torch.device('cpu'),
        epochval=None,
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600,
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 5, 1, 1]
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.

        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        self.loss_history = []

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        print("epochval -->", epochval)

        if self.datatype == 'SPOTS':
           self.epochs = 600
           self.weight_factors = [1,5,1,1]

        elif self.datatype == 'Stereo-CITE-seq':
           self.epochs = 1500
           self.weight_factors = [1,10,1,10]

        elif self.datatype == '10x':
           self.epochs = 200
           self.weight_factors = [1,5,1,10]

        elif self.datatype == 'Spatial-epigenome-transcriptome':
           self.epochs = 1600
           self.weight_factors = [1,5,1,1]

        if epochval is not None:
            self.epochs = epochval



    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

            # correspondence loss
            self.loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            self.loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

            loss = self.weight_factors[0]*self.loss_recon_omics1 + self.weight_factors[1]*self.loss_recon_omics2 + self.weight_factors[2]*self.loss_corr_omics1 + self.weight_factors[3]*self.loss_corr_omics2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch == 0:
               print(f'Epoch {epoch} - Initial Loss: {loss.item():.4f}')

            self.loss_history.append(loss.item())

        print(f"Model training finished! Final Loss: {loss.item():.4f}\n")

        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'SpatialGlue': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy(),
                  'loss_history': self.loss_history}

        return output










# ------------------------------ Run Configuration ----------------------

choices = [
    ("10x_human_lymph_node_A1",
     "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC"),
    ("10x_human_lymph_node_D1",
     "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz"),
    ("Mouse_Brain_E11_S1",
     "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo"),
    ("Mouse_Brain_E13_S1",
     "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud"),
    ("Mouse_Brain_E15_S1",
     "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD"),
    ("Mouse_Brain_E18_S1",
     "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB"),
]

import argparse
parser = argparse.ArgumentParser(description="Run SpatialGlue multimodal spatial clustering benchmark")
parser.add_argument('--dataset', type=int, default=None, help="Single dataset index to run (0 to 5)")
parser.add_argument('--dataset_indices', type=int, nargs='+', default=None, help="List of dataset indices (e.g. --dataset_indices 0 1)")
parser.add_argument('--seed', type=int, default=None, help="Single seed to run (e.g. --seed 0)")
parser.add_argument('--seeds', type=int, nargs='+', default=None, help="List of seeds to run (e.g. --seeds 42 0 7)")
parser.add_argument('--output_dir', type=str, default=None, help="Custom output directory")
parser.add_argument('--api_url', type=str, default="https://model-performance.vercel.app/api/experiments/upload", help="Dashboard API URL for uploading experiment runs")
args, unknown = parser.parse_known_args()

# Parse dataset selection
if args.dataset is not None:
    DATASET_INDICES = [args.dataset]
elif args.dataset_indices is not None:
    DATASET_INDICES = args.dataset_indices
else:
    DATASET_INDICES = [0, 1, 2, 3, 4, 5]

# Parse seed selection
if args.seed is not None:
    SEEDS = [args.seed]
elif args.seeds is not None:
    SEEDS = args.seeds
else:
    SEEDS = [42, 0, 7, 123, 1234, 2022, 2023, 2024, 1337]

MODEL_NAME = "SpatialGlue"
OUTPUT_DIR = args.output_dir if args.output_dir else os.getenv("OUTPUT_DIR", f"{MODEL_NAME}_outputs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
VISUALS_DIR = os.path.join(OUTPUT_DIR, "visuals")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

# Set R_HOME and path before any rpy2 operations
import os
import subprocess
import time

if 'R_HOME' not in os.environ:
    for candidate in ['/usr/lib/R', '/usr/local/lib/R', '/opt/R']:
        if os.path.exists(candidate):
            os.environ['R_HOME'] = candidate
            break
    if 'R_HOME' not in os.environ:
        try:
            r_home_det = subprocess.check_output(['R', 'RHOME'], text=True).strip()
            if r_home_det and os.path.exists(r_home_det):
                os.environ['R_HOME'] = r_home_det
        except Exception:
            os.environ['R_HOME'] = '/usr/lib/R'

if 'R_HOME' in os.environ and os.path.exists(os.path.join(os.environ['R_HOME'], 'bin')):
    r_bin = os.path.join(os.environ['R_HOME'], 'bin')
    if r_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = r_bin + ':' + os.environ.get('PATH', '')

# Import rpy2 and setup mclust
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, default_converter
import rpy2.robjects.conversion as cv

cv.set_conversion(default_converter + pandas2ri.converter)

# Suppress warnings
robjects.r.options(warn=-1)

# Install mclust only if it is not already installed
robjects.r("""
if (!requireNamespace("mclust", quietly = TRUE)) {
    install.packages("mclust", repos="https://cloud.r-project.org")
}
library(mclust)
""")

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def compute_chaos(spatial_coords, labels):
    """Spatial CHAOS score (Shang & Zhou, Nature Communications 2022). Lower is better."""
    coords = np.asarray(spatial_coords, dtype=np.float64)
    labels = np.asarray(labels)
    n_spots = len(labels)
    if n_spots <= 1:
        return 0.0
    coords_norm = StandardScaler().fit_transform(coords)
    total_dist = 0.0
    for cl in np.unique(labels):
        idx = np.where(labels == cl)[0]
        if len(idx) <= 1:
            continue
        sub_coords = coords_norm[idx]
        nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(sub_coords)
        dists, _ = nbrs.kneighbors(sub_coords)
        total_dist += float(np.sum(dists[:, 1]))
    return float(total_dist / n_spots)


def compute_pas(spatial_coords, labels, k=10):
    """Percentage of Abnormal Spots (Shang & Zhou, Nature Communications 2022). Lower is better."""
    coords = np.asarray(spatial_coords, dtype=np.float64)
    labels = np.asarray(labels)
    n_spots = len(labels)
    k_eff = min(k, n_spots - 1)
    if k_eff <= 0:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean').fit(coords)
    _, indices = nbrs.kneighbors(coords)
    neighbor_labels = labels[indices[:, 1:]]
    diff_frac = np.mean(neighbor_labels != labels[:, None], axis=1)
    return float(np.mean(diff_frac > 0.5))


def compute_morans_i(spatial_coords, labels, k=10):
    """Permutation-invariant mean Moran's I across one-hot domain indicators. Higher is better."""
    coords = np.asarray(spatial_coords, dtype=np.float64)
    labels = np.asarray(labels)
    n_spots = len(labels)
    k_eff = min(k, n_spots - 1)
    if k_eff <= 0:
        return 0.0
    W = kneighbors_graph(coords, n_neighbors=k_eff, mode='connectivity', include_self=False)
    W = W.maximum(W.T).tocsr()
    W_sum = float(W.sum())
    if W_sum <= 0.0:
        return 0.0
    moran_vals = []
    for cl in np.unique(labels):
        y = (labels == cl).astype(np.float64)
        y_c = y - y.mean()
        denom = float(np.dot(y_c, y_c))
        if denom <= 1e-12:
            continue
        num = float(y_c @ (W @ y_c))
        moran_vals.append((n_spots / W_sum) * (num / denom))
    return float(np.mean(moran_vals)) if len(moran_vals) > 0 else 0.0


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
        val = float(obj)
        return 0.0 if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, float):
        return 0.0 if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    return obj


def export_dashboard_experiment(
    model_id: str,
    model_name: str,
    dataset_name: str,
    seed: int,
    metrics_dict: dict,
    hyperparameters: dict,
    duration_seconds: float,
    best_epoch: int = 200,
    embeddings_data: Optional[dict] = None,
    output_dir: str = "results",
    api_url: Optional[str] = "https://model-performance.vercel.app/api/experiments/upload",
    architecture: str = "Spatial Multi-Omics Dual-Attention GNN (SpatialGlue)",
):
    """Export standardized JSON artifacts and send directly to Dashboard via API."""
    import json
    import urllib.request

    exp_dir = os.path.join(output_dir, "experiments", model_id, dataset_name, f"seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    safe_metrics = make_json_serializable(metrics_dict)
    safe_hyperparameters = make_json_serializable(hyperparameters)
    safe_embeddings = make_json_serializable(embeddings_data or {})
    best_score = float(safe_metrics.get("Silhouette", 0.0))

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "dataset": dataset_name,
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_score": best_score,
            "durationSeconds": float(duration_seconds),
            "metrics": safe_metrics,
        }, f, indent=2)

    with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "dataset": dataset_name,
            "seed": int(seed),
            "durationSeconds": float(duration_seconds),
            "hyperparameters": safe_hyperparameters,
        }, f, indent=2)

    if api_url:
        try:
            payload = {
                "modelId": model_id,
                "modelName": model_name,
                "datasetId": dataset_name,
                "datasetName": dataset_name.replace("_", " ").title(),
                "seed": int(seed),
                "bestEpoch": int(best_epoch),
                "bestScore": best_score,
                "durationSeconds": float(duration_seconds),
                "finalMetrics": safe_metrics,
                "history": [],
                "embeddingsData": safe_embeddings,
                "modelMetadata": {
                    "architecture": architecture,
                    "hyperparameters": safe_hyperparameters,
                },
            }
            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    print(f"🚀 Sent {model_name} ({dataset_name}, seed={seed}, {duration_seconds:.2f}s) to Dashboard API!")
        except Exception as e:
            print(f"ℹ️ Direct API push skipped ({e}). Data saved locally in {exp_dir}.")


all_results = []
all_dataset_adata = {}
tool = 'mclust'  # mclust, leiden, and louvain
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\n[Hardware] Compute Engine: {device}")
if torch.cuda.is_available():
    print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
else:
    print("[Hardware] No GPU detected. Running on CPU.\n")

for dataset_idx in DATASET_INDICES:
    dataset_name, folder_url = choices[dataset_idx]

    print("\n" + "#"*80)
    print(f" STARTING DATASET: {dataset_name} ".center(80, "#"))
    print("#"*80)

    # ------------------ Download & Path Configuration ------------------
    base = f"data/{dataset_name}"
    os.makedirs(base, exist_ok=True)

    rna_path = os.path.join(base, "adata_RNA.h5ad")

    # Human datasets have ADT; Mouse datasets have ATAC
    if dataset_name.startswith("10x"):
        other_path = os.path.join(base, "adata_ADT.h5ad")
        annotation_path = os.path.join(base, "annotation.csv")
        gt_column = "manual-anno"
        data_type = '10x'
    else:
        other_path = os.path.join(base, "adata_ATAC.h5ad")
        annotation_path = os.path.join(base, "anno.csv")
        gt_column = "cluster"
        data_type = 'Spatial-epigenome-transcriptome'

    # Check if files already exist locally before triggering download
    if not os.path.exists(rna_path) or not os.path.exists(other_path) or not os.path.exists(annotation_path):
        print(f"Downloading dataset files into: {base}")
        gdown_cmd = ".venv/bin/gdown" if os.path.exists(".venv/bin/gdown") else "gdown"
        os.system(f'{gdown_cmd} --folder "{folder_url}" --output "{base}" --remaining-ok')
        # If gdown created a subfolder, flatten files into base
        for root, dirs, files in os.walk(base):
            if root != base:
                for f in files:
                    src = os.path.join(root, f)
                    dst = os.path.join(base, f)
                    if not os.path.exists(dst):
                        import shutil
                        shutil.move(src, dst)
    else:
        print(f"Dataset files already exist at {base}. Skipping download.")

    # ------------------ Load AnnData & Annotations ------------------
    adata_omics1 = sc.read_h5ad(rna_path)
    adata_omics2 = sc.read_h5ad(other_path)

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    anno_df = pd.read_csv(annotation_path, index_col=0)
    adata_omics1.obs['ground_truth'] = anno_df[gt_column]
    adata_omics2.obs['ground_truth'] = anno_df[gt_column]

    # ------------------ Preprocessing ------------------
    fix_seed(2022)
    # RNA preprocessing
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    if not dataset_name.startswith("10x"):
        sc.pp.filter_cells(adata_omics1, min_genes=200)

    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]

    # Protein or ATAC preprocessing
    if dataset_name.startswith("10x"):
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        adata_omics1.obsm["feat"] = pca(
            adata_omics1_high,
            n_comps=adata_omics2.n_vars - 1
        )
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm["feat"] = pca(
            adata_omics2,
            n_comps=adata_omics2.n_vars - 1
        )
    else:
        adata_omics1.obsm["feat"] = pca(
            adata_omics1_high,
            n_comps=50
        )
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        if "X_lsi" not in adata_omics2.obsm:
            sc.pp.highly_variable_genes(
                adata_omics2,
                flavor="seurat_v3",
                n_top_genes=3000
            )
            lsi(
                adata_omics2,
                use_highly_variable=False,
                n_components=51
            )
        adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()

    # ------------------ Spatial Neighbor Graph Construction ------------------
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

    # ------------------ Seed Iterations ------------------
    dataset_results = []
    n_ground_truth = int(adata_omics1.obs["ground_truth"].dropna().nunique())
    print(f"Number of ground truth classes: {n_ground_truth}")
    n_cluster = n_ground_truth

    best_ari = -1.0
    best_seed = None
    best_adata = None

    for seed in SEEDS:
        print(f"\n" + "-"*60)
        print(f" Dataset: {dataset_name} | Seed: {seed} ".center(60, "-"))
        print("-"*60)

        seed_start_time = time.time()

        # 1. Set seed
        fix_seed(seed)

        # 2. Model Training
        # We need a fresh deep copy of the data dictionary for each seed run to avoid side-effects
        import copy
        data_copy = {
            'adata_omics1': data['adata_omics1'].copy(),
            'adata_omics2': data['adata_omics2'].copy()
        }

        # Train model
        model = Train_SpatialGlue(data_copy, datatype=data_type, device=device, random_seed=seed)
        output = model.train()

        adata = data_copy['adata_omics1'].copy()
        adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
        adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
        adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()
        adata.obsm['alpha'] = output['alpha']
        adata.obsm['alpha_omics1'] = output['alpha_omics1']
        adata.obsm['alpha_omics2'] = output['alpha_omics2']

        # 3. Clustering
        clustering(adata, key='SpatialGlue', add_key='SpatialGlue', n_clusters=n_cluster, method=tool, use_pca=True, random_seed=seed)

        duration_sec = round(time.time() - seed_start_time, 2)

        # 4. Score Generating
        valid_mask = adata.obs['ground_truth'].notna()
        y_true = adata.obs.loc[valid_mask, 'ground_truth'].astype(str)
        y_pred_eval = adata.obs.loc[valid_mask, 'SpatialGlue'].astype(str)
        y_pred = adata.obs['SpatialGlue'].astype(str)

        ari = adjusted_rand_score(y_true, y_pred_eval)
        nmi = normalized_mutual_info_score(y_true, y_pred_eval)
        ami = adjusted_mutual_info_score(y_true, y_pred_eval)
        homogeneity = homogeneity_score(y_true, y_pred_eval)
        v_measure = v_measure_score(y_true, y_pred_eval)

        # Calculate Silhouette Score using robust LabelEncoder (filtered by valid_mask)
        joint_feat = adata.obsm['SpatialGlue'][valid_mask.values]
        le = LabelEncoder()
        y_pred_int = le.fit_transform(y_pred_eval)

        try:
            sil_score = float(silhouette_score(joint_feat, y_pred_int))
        except Exception:
            sil_score = float('nan')

        try:
            chi_score = float(calinski_harabasz_score(joint_feat, y_pred_int))
        except Exception:
            chi_score = float('nan')

        try:
            dbi_score = float(davies_bouldin_score(joint_feat, y_pred_int))
        except Exception:
            dbi_score = float('nan')

        # Spatial Coherence Metrics: Moran's I (↑), PAS (↓), CHAOS (↓)
        try:
            sc_coords = adata.obsm['spatial']
            morans_i = compute_morans_i(sc_coords, y_pred, k=10)
            pas_score = compute_pas(sc_coords, y_pred, k=10)
            chaos_score = compute_chaos(sc_coords, y_pred)
        except Exception:
            morans_i = float('nan')
            pas_score = float('nan')
            chaos_score = float('nan')

        print(f"\nResult for dataset: {dataset_name} | seed: {seed}")
        print(f"ARI: {ari:.4f} | NMI: {nmi:.4f} | AMI: {ami:.4f} | Silhouette: {sil_score:.4f} | DBI: {dbi_score:.4f}")
        print(f"Moran's I: {morans_i:.4f} | PAS: {pas_score:.4f} | CHAOS: {chaos_score:.4f} | CHI: {chi_score:.2f} | Duration: {duration_sec:.2f}s")

        # Store seed results
        res_dict = {
            'dataset': dataset_name,
            'seed': seed,
            'ARI': ari,
            'NMI': nmi,
            'AMI': ami,
            'Homogeneity': homogeneity,
            'V-measure': v_measure,
            'Silhouette': sil_score,
            'CHI': chi_score,
            'DBI': dbi_score,
            'Morans_I': morans_i,
            'PAS': pas_score,
            'CHAOS': chaos_score,
            'Duration': duration_sec
        }
        dataset_results.append(res_dict)
        all_results.append(res_dict)

        # Track best seed AnnData for cross-dataset visualizations
        if ari > best_ari or best_adata is None:
            best_ari = ari
            best_seed = seed
            best_adata = adata.copy()
            best_adata.obsm['spatial'] = adata_omics1.obsm['spatial']
            best_adata.obsm[f'{MODEL_NAME}_emb'] = output['SpatialGlue'].copy()
            best_adata.obs[f'{MODEL_NAME}_domain'] = adata.obs['SpatialGlue'].copy()
            best_adata.obs['domain'] = adata.obs['SpatialGlue'].copy()
            if 'alpha' in output and output['alpha'] is not None:
                # Modality 1 (RNA) attention weight
                best_adata.obs['mean_rna_attention'] = output['alpha'][:, 0]
            best_adata.uns['selected_seed'] = best_seed

        # ------------------ Plotting & Visualization ------------------
        fig, ax_list = plt.subplots(1, 3, figsize=(11, 3))
        sc.pp.neighbors(adata, use_rep='SpatialGlue', n_neighbors=10)
        sc.tl.umap(adata)

        if 'anno' not in adata.obs.columns and 'ground_truth' in adata.obs.columns:
            adata.obs['anno'] = adata.obs['ground_truth']

        sc.pl.umap(adata, color='SpatialGlue', ax=ax_list[0], title='SpatialGlue', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color='SpatialGlue', ax=ax_list[1], title='SpatialGlue', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color='anno', ax=ax_list[2], title='anno', s=20, show=False)
        plt.tight_layout(w_pad=0.3)

        seed_plots_dir = os.path.join(VISUALS_DIR, "seed_plots", dataset_name)
        os.makedirs(seed_plots_dir, exist_ok=True)
        os.makedirs(f"plots/{dataset_name}", exist_ok=True)
        plt.savefig(f"{seed_plots_dir}/umap_seed_{seed}.png", dpi=150)
        plt.savefig(f"plots/{dataset_name}/umap_seed_{seed}.png", dpi=150)
        plt.close(fig)

        # ------------------ Export to Database / Dashboard ------------------
        try:
            sample_sils = silhouette_samples(joint_feat, y_pred_int).tolist()
        except Exception:
            sample_sils = []

        umap_coords = adata.obsm.get('X_umap', None)
        spatial_coords = adata.obsm.get('spatial', None)
        embeddings_data = {
            "umapCoordinates": umap_coords.tolist() if umap_coords is not None else [],
            "spatialCoordinates": spatial_coords.tolist() if spatial_coords is not None else [],
            "predictedLabels": y_pred.tolist(),
            "groundTruthLabels": adata.obs['ground_truth'].astype(str).tolist(),
            "sampleSilhouettes": sample_sils,
        }
        metrics_payload = {
            "ARI": float(ari),
            "NMI": float(nmi),
            "AMI": float(ami),
            "Homogeneity": float(homogeneity),
            "V-measure": float(v_measure),
            "Silhouette": float(sil_score),
            "CHI": float(chi_score),
            "DBI": float(dbi_score),
            "Morans_I": float(morans_i),
            "PAS": float(pas_score),
            "CHAOS": float(chaos_score),
        }
        hp_payload = {
            "datatype": data_type,
            "clustering_tool": tool,
        }
        export_dashboard_experiment(
            model_id="SpatialGlue",
            model_name=MODEL_NAME,
            dataset_name=dataset_name,
            seed=seed,
            metrics_dict=metrics_payload,
            hyperparameters=hp_payload,
            duration_seconds=duration_sec,
            best_epoch=200,
            embeddings_data=embeddings_data,
            output_dir=RESULTS_DIR,
            api_url=args.api_url,
        )

    # Store best AnnData
    if best_adata is not None:
        all_dataset_adata[dataset_name] = best_adata
        best_h5ad_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_adata_{dataset_name}.h5ad")
        best_adata.write_h5ad(best_h5ad_path)
        print(f"[Saved] Best-ARI seed ({best_seed}) AnnData saved to: {best_h5ad_path}")

    # Write dataset summary statistics and save csv for this dataset
    df_ds = pd.DataFrame(dataset_results)
    print("\n" + "="*80)
    print(f" RESULTS SUMMARY FOR {dataset_name} ".center(80, "="))
    print("="*80)
    print(df_ds.to_string(index=False))
    print("-"*80)
    print("Summary Statistics:")
    print(df_ds.describe().loc[['mean', 'std']])
    print("="*80)

    # Generate text summary with Mean ± SD format for this dataset
    summary_lines = [f"Summary Stats for {dataset_name}:"]
    metrics_list = ['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette', 'CHI', 'DBI', 'Duration']
    for metric in metrics_list:
        mean_val = df_ds[metric].mean()
        std_val = df_ds[metric].std()
        summary_lines.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    dataset_summary_str = "\n".join(summary_lines)
    print(dataset_summary_str)
    print("="*80)

    # Save to CSV and summary file
    with open(os.path.join(RESULTS_DIR, f"summary_{dataset_name}.txt"), "w") as f:
        f.write(dataset_summary_str)

    df_ds.to_csv(os.path.join(RESULTS_DIR, f"metrics_{dataset_name}.csv"), index=False)
    # Also save to root results/ for compatibility
    os.makedirs("results", exist_ok=True)
    df_ds.to_csv(f"results/SpatialGlue_{dataset_name}_results.csv", index=False)
    df_ds.to_csv(f"results/metrics_{dataset_name}.csv", index=False)
    with open(f"results/summary_{dataset_name}.txt", "w") as f:
        f.write(dataset_summary_str)
    print(f"Saved dataset metrics to '{RESULTS_DIR}/metrics_{dataset_name}.csv'")
    print(f"Saved dataset summary to '{RESULTS_DIR}/summary_{dataset_name}.txt'")

    # Generate publication-quality boxplot + stripplot for this dataset
    plt.figure(figsize=(14, 6))
    df_ds_long = df_ds.melt(
        id_vars=['seed'],
        value_vars=['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette'],
        var_name='Metric',
        value_name='Score'
    )
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(
        x='Metric',
        y='Score',
        data=df_ds_long,
        palette='Set2',
        showfliers=False,
        width=0.5
    )
    sns.stripplot(
        x='Metric',
        y='Score',
        data=df_ds_long,
        color='black',
        alpha=0.6,
        jitter=0.2,
        size=5,
        ax=ax
    )
    plt.title(f"Variability of Metrics across {len(SEEDS)} Seeds ({dataset_name})", fontsize=14, pad=15)
    plt.xlabel("Metric", fontsize=12, labelpad=10)
    plt.ylabel("Score", fontsize=12, labelpad=10)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    boxplot_path = os.path.join(VISUALS_DIR, f"boxplot_{dataset_name}.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.savefig(f"results/boxplot_{dataset_name}.png", dpi=300)
    plt.close()
    print(f"Saved variability boxplot to '{boxplot_path}'\n")

# Write overall summary to file
df_all = pd.DataFrame(all_results)
df_all.to_csv(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_benchmark_results.csv"), index=False)
df_all.to_csv(os.path.join(RESULTS_DIR, "metrics_all_datasets.csv"), index=False)
df_all.to_csv(f"results/{MODEL_NAME}_all_results.csv", index=False)
df_all.to_csv("results/metrics_all_datasets.csv", index=False)

# Compile global text summary
global_summary_lines = [
    "="*80,
    f"GLOBAL SUMMARY: MEAN ± STD ACROSS {len(SEEDS)} SEEDS ({MODEL_NAME})",
    "="*80
]

for d_name in df_all['dataset'].unique():
    df_d = df_all[df_all['dataset'] == d_name]
    global_summary_lines.append(f"\nDataset: {d_name}")
    global_summary_lines.append("-" * len(f"Dataset: {d_name}"))
    for metric in ['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette', 'CHI', 'DBI', 'Duration']:
        mean_val = df_d[metric].mean()
        std_val = df_d[metric].std()
        global_summary_lines.append(f"{metric} = {mean_val:.4f} ± {std_val:.4f}")
global_summary_lines.append("\n" + "="*80)

global_summary_str = "\n".join(global_summary_lines)
print(global_summary_str)

summary_stats_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_summary_statistics.txt")
with open(summary_stats_path, "w") as f:
    f.write(global_summary_str)
with open("results/summary_statistics.txt", "w") as f:
    f.write(global_summary_str)

print("\n" + "="*80)
print(" ALL DATASETS COMPLETED ".center(80, "="))
print("="*80)
print(df_all.groupby('dataset')[['ARI', 'NMI', 'Silhouette', 'CHI', 'DBI', 'Duration']].mean())
print("="*80)
print(f"Saved combined metrics to '{RESULTS_DIR}/{MODEL_NAME}_benchmark_results.csv'")
print(f"Saved global summary statistics to '{summary_stats_path}'")

# Generate comparative global boxplot across all datasets
df_all_long = df_all.melt(
    id_vars=['dataset', 'seed'],
    value_vars=['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette'],
    var_name='Metric',
    value_name='Score'
)

fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
axes = axes.flatten()
metrics_list = ['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette']

sns.set_theme(style="whitegrid")
for i, metric in enumerate(metrics_list):
    df_metric = df_all_long[df_all_long['Metric'] == metric]
    sns.boxplot(
        ax=axes[i],
        x='dataset',
        y='Score',
        data=df_metric,
        palette='Set3',
        showfliers=False
    )
    sns.stripplot(
        ax=axes[i],
        x='dataset',
        y='Score',
        data=df_metric,
        color='black',
        alpha=0.5,
        jitter=0.15,
        size=4
    )
    axes[i].set_title(f"{metric} Comparison", fontsize=14)
    axes[i].set_xlabel("Dataset", fontsize=12)
    axes[i].set_ylabel("Score", fontsize=12)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].set_ylim(0, 1.05)

plt.suptitle(f"{MODEL_NAME}: Performance Comparison Across All Datasets & {len(SEEDS)} Seeds", fontsize=18, y=0.98)
plt.tight_layout()
global_plot_path = os.path.join(VISUALS_DIR, f"{MODEL_NAME}_global_boxplot_comparison.png")
plt.savefig(global_plot_path, dpi=300)
plt.savefig("results/global_boxplot_comparison.png", dpi=300)
plt.close()
print(f"Saved global comparison boxplot to '{global_plot_path}'")

# -------------------------------------------------------------------------------------------------
# EXECUTE FULL VISUALIZATION SUITE (Exactly matching visualization-add.txt)
# -------------------------------------------------------------------------------------------------
try:
    from visualizations import run_all_visualizations
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from visualizations import run_all_visualizations

print("\n" + "="*85)
print(f" GENERATING PUBLICATION VISUALIZATION SUITE ({MODEL_NAME}) ".center(85, "="))
print("="*85)
run_all_visualizations(
    model_name=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    df_results=df_all,
    loaded_adatas=all_dataset_adata,
    choices=choices,
    dataset_indices=DATASET_INDICES
)

# -------------------------------------- End ---------------------------------------










