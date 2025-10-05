import torch
import scanpy as sc
import numpy as np
from scipy import sparse
from typing import Tuple
from torch import Tensor
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity


def RNA_guided_imputation(rna_data, atac_data, k=8, weighted=True):

    pca = PCA(n_components=min(15, rna_data.shape[1]))
    rna_pca = pca.fit_transform(rna_data)
    similarity = cosine_similarity(rna_pca)
    neighbors = np.argsort(similarity, axis=1)[:, -k:]
    imputed_atac = np.zeros_like(atac_data)

    return imputed_atac



class TensorDataSetWithIndex(TensorDataset):
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor):
        super(TensorDataSetWithIndex, self).__init__(*tensors)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), index
def add_gaussian_noise(data, sigma=0.1):
    noise = torch.randn_like(data) * sigma
    return data + noise

def prepare_dataloader(args):
    # Load and preprocess RNA data
    scrna_adata = sc.read_h5ad(args.data_path + args.scrna_data)
    if sparse.issparse(scrna_adata.X):
        scrna_adata.X = scrna_adata.X

    if args.scrna_preprocess == "Standard":
        sc.pp.normalize_total(scrna_adata, target_sum=1e4)
        sc.pp.log1p(scrna_adata)
    elif args.scrna_preprocess == "TFIDF":
        tfidf = TfidfTransformer()
        scrna_adata.X = tfidf.fit_transform(scrna_adata.X).toarray()
    else:
        raise NotImplementedError
    sc.pp.scale(scrna_adata)
    scrna_adata.obs["Domain"] = args.scrna_data[:-5]
    scrna_label = scrna_adata.obs["cell_type"]
    scrna_label_int = scrna_label.rank(method="dense", ascending=True).astype(int) - 1
    scrna_label = scrna_label.values
    scrna_label_int = scrna_label_int.values
    label_map = {k: scrna_label[scrna_label_int == k][0] for k in range(scrna_label_int.max() + 1)}

    # Load and preprocess ATAC data
    scatac_adata = sc.read_h5ad(args.data_path + args.scatac_data)
    if sparse.issparse(scatac_adata.X):
        scatac_adata.X = scatac_adata.X

    scatac_adata.X = RNA_guided_imputation(
        rna_data=scrna_adata.X.toarray() if sparse.issparse(scrna_adata.X) else scrna_adata.X,
        atac_data=scatac_adata.X.toarray() if sparse.issparse(scatac_adata.X) else scatac_adata.X,
        k=8
    )

    if args.scatac_preprocess == "Standard":
        sc.pp.normalize_total(scatac_adata, target_sum=1e4)
        sc.pp.log1p(scatac_adata)
    elif args.scatac_preprocess == "TFIDF":
        tfidf = TfidfTransformer()
        scatac_adata.X = tfidf.fit_transform(scatac_adata.X).toarray()
    else:
        raise NotImplementedError

    sc.pp.scale(scatac_adata)
    scatac_adata.obs["Domain"] = args.scatac_data[:-5]

    # Prepare PyTorch Data
    scrna_data = torch.from_numpy(scrna_adata.X).float()
    scrna_label_int = torch.from_numpy(scrna_label_int).long()
    scatac_data = torch.from_numpy(scatac_adata.X).float()
    scatac_index = torch.arange(scatac_data.shape[0]).long()

    # Prepare PyTorch Dataset and DataLoader
    scrna_dataset = TensorDataset(scrna_data, scrna_label_int)
    scrna_dataloader_train = DataLoader(
        dataset=scrna_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    scrna_dataloader_eval = DataLoader(
        dataset=scrna_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    scatac_dataset = TensorDataSetWithIndex(scatac_data, scatac_index)
    scatac_dataloader_train = DataLoader(
        dataset=scatac_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    scatac_dataloader_eval = DataLoader(
        dataset=scatac_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    gene_num = scrna_data.shape[1]
    type_num = torch.unique(scrna_label_int).shape[0]

    return (
        scrna_dataset,
        scrna_dataloader_train,
        scrna_dataloader_eval,
        scatac_dataset,
        scatac_dataloader_train,
        scatac_dataloader_eval,
        gene_num,
        type_num,
        label_map,
        scrna_adata,
        scatac_adata,
    )


def adjacency(X, K=8):
    adj = kneighbors_graph(
        X.cpu().numpy(),
        K,
        mode="connectivity",
        include_self=True,
    ).toarray()
    adj = adj * adj.T
    return adj


def partition_data(
    predictions,
    prob_feature,
    prob_logit,
    scrna_dataset,
    scatac_dataset,
    args,
):
    reliable_index = (prob_feature > args.reliability_threshold) & (
        prob_logit > args.reliability_threshold
    )
    unreliable_index = ~reliable_index
    reliable_samples = scatac_dataset.tensors[0][reliable_index]
    reliable_predictions = predictions[reliable_index]
    scrna_data = torch.cat((scrna_dataset.tensors[0], reliable_samples))
    scrna_type = torch.cat(
        (scrna_dataset.tensors[1], reliable_predictions)
    )
    scrna_dataset = TensorDataset(scrna_data, scrna_type)
    unreliable_samples = scatac_dataset.tensors[0][unreliable_index]
    unreliable_index = scatac_dataset.tensors[1][unreliable_index]
    scatac_dataset = TensorDataSetWithIndex(unreliable_samples, unreliable_index)

    print(
        "rna_data_size:",
        scrna_dataset.__len__(),
        "atac_data_size:",
        scatac_dataset.__len__(),
    )

    # DataLoader
    scrna_dataloader_train = DataLoader(
        dataset=scrna_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    scrna_dataloader_eval = DataLoader(
        dataset=scrna_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    scatac_dataloader_train = DataLoader(
        dataset=scatac_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    scatac_dataloader_eval = DataLoader(
        dataset=scatac_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return (
        scrna_dataloader_train,
        scrna_dataloader_eval,
        scatac_dataloader_train,
        scatac_dataloader_eval,
        scrna_dataset,
        scatac_dataset,
    )
