
import numpy as np
import time
import torch
from torch import nn
from tqdm import tqdm

@torch.no_grad()
def get_assignments(X, centroids, chunk_size=None, H_inv_diag=None):
    """
    X: G x N x D
    centroids: G x K x D
    """
    if H_inv_diag is None:
        H_inv_diag = torch.ones(X.shape[-1]).to(X.device)
    elif H_inv_diag.ndim > 2:  # should then be 1 x N x D
        assert (
            H_inv_diag.shape[0] == 1
            and H_inv_diag.shape[1] == X.shape[1]
            and H_inv_diag.shape[2] == X.shape[2]
        ), f"{H_inv_diag.shape, X.shape}"
        H_inv_diag = H_inv_diag.unsqueeze(2)  # 1 x N x 1 x D

    if chunk_size is None:
        X_chunks = [X]
        H_inv_diag_chunks = [H_inv_diag]
    else:
        X_chunks = torch.split(X, chunk_size, dim=1)
        if H_inv_diag.ndim > 1:
            H_inv_diag_chunks = torch.split(H_inv_diag, chunk_size, dim=1)
        else:
            H_inv_diag_chunks = [H_inv_diag] * len(X_chunks)

    centroids = centroids.unsqueeze(1)  # G x 1 x K x D

    assignments = []
    for X, H_inv_diag in zip(X_chunks, H_inv_diag_chunks):
        X = X.unsqueeze(2)  # G x N' x 1 x D

        dist = ((X - centroids).pow(2) * H_inv_diag).sum(-1)

        assignments.append(dist.argmin(-1))  # G x N'
    assignments = torch.concat(assignments, dim=1)

    return assignments  # G x N

@torch.no_grad()
def kmeans_m_step_3(
    centroids: torch.Tensor,
    n_centroids: int,
    assignments: torch.LongTensor,
    X: torch.Tensor,
    H_inv_diag=None,
):
    """
    X: G x N x D
    centroids: G x K x D
    assignments: G x N
    H_inv_diag: 1 x N x D
    """
    crange = torch.arange(0, n_centroids).to(centroids.device)

    # G x N x 1 == 1 x 1 x K --> G x N x K
    assignments_expanded = (assignments.unsqueeze(-1) == crange.view(1, 1, -1)).to(X.dtype)

    if H_inv_diag is None:
        norm = 1.0 / torch.clip(assignments_expanded.sum(1), min=1)  # G x K
        clusters_for_centroid = torch.einsum("gnd,gnk,gk->gkd", X, assignments_expanded, norm)
    else:
        norm = 1.0 / torch.clip(
            torch.einsum("gnk,nd->gkd", assignments_expanded, H_inv_diag[0]), min=1e-10
        )
        clusters_for_centroid = torch.einsum(
            "gnd,nd,gnk,gkd->gkd", X, H_inv_diag[0], assignments_expanded, norm
        )

    centroids.copy_(clusters_for_centroid)

@torch.no_grad()
def kpp_parallel_sampled(data: torch.Tensor, k: int):
    G, N, D = data.shape

    if N * D < 32768 * 2:
        split_data = data.split(16)
    elif N * D * k < 32768 * 2 * 16:
        split_data = data.split(4)
    else:
        split_data = data.split(1)

    all_init = []

    for data in split_data:
        init = torch.zeros((data.shape[0], k, data.shape[-1]), dtype=torch.float16).to(
            data.device
        )  # G, K, D
        all_dists = torch.cdist(data.to(torch.float16), data.to(torch.float16), p=2)  # G, N, N
        init[:, 0] = data[:, 0]

        D2 = torch.zeros(data.shape[0], k, N).to(data.device)
        D2[:, 0] = all_dists[:, 0]

        for i in range(1, k):
            dists = D2[:, :i].amin(dim=1)  # G, N
            dists = (dists / dists.sum(-1, keepdims=True)).cumsum(-1)  # G, N

            v = torch.rand_like(dists[:, :1])  # G, 1

            idx = torch.clip(torch.searchsorted(dists, v).unsqueeze(-1), 0, N - 1)  # G, 1, 1

            D2[:, i : i + 1] = torch.gather(all_dists, dim=1, index=idx.expand(-1, 1, N))
            init[:, i : i + 1] = torch.gather(data, dim=1, index=idx.expand(-1, 1, D))
        all_init.append(init)
    return torch.concatenate(all_init)

@torch.no_grad()
def mahalanobis_init(X, n_centroids):
    """
    X: G x N x D
    centroids: G x K x D
    """
    vq_dim = X.shape[-1]
    mu = X.mean(1).unsqueeze(1)
    Xcentered = X - mu

    Sigma = torch.bmm(Xcentered.transpose(1, 2), Xcentered)  # G x D x D
    Lambda = torch.linalg.inv(Sigma)

    dists = (torch.bmm(Xcentered, Lambda) * Xcentered).sum(-1)  # G x N
    sorted_dists = torch.argsort(dists, dim=1)  # G x N
    idx = torch.round(torch.linspace(0, Xcentered.shape[1] - 1, n_centroids)).long()  # K
    idx = (
        sorted_dists[:, idx].unsqueeze(-1).expand(-1, -1, vq_dim)
    )  # G x K --> G x K x 1 --> G x K x D

    return torch.gather(X, dim=1, index=idx)

def kmeans_vq(
    X,
    centroids,
    iters=100,
    assignment_chunk_size=None,
    H_inv_diag=None,
):
    assert iters > 0
    n_centroids = centroids.shape[1]
    assignments = None
    for iter in range(iters):
        # E-step
        assignments = get_assignments(
            X, centroids, chunk_size=assignment_chunk_size, H_inv_diag=H_inv_diag
        )

        # M-step: gather all values for each centroid and compute means
        # Centroids is shape G x D x K; assignments is shape G x N
        kmeans_m_step_3(centroids, n_centroids, assignments, X, H_inv_diag=H_inv_diag)
    return assignments

@torch.no_grad()
def find_params(
    X: torch.Tensor, 
    kmeans_init_method: str = "mahalanobis",
    n_centroids: int = 16,
    kmeans_iters: int = 100,
):
    assert len(X.shape) == 3
    vq_dim = X.shape[-1]
    if kmeans_init_method == "cdf":
        assert vq_dim == 1
        X, _ = torch.sort(X, 1)
        idx = torch.round(torch.linspace(0, X.shape[1] - 1, n_centroids)).long()
        centroids = X[:, idx].clone()  # G x K x 1
    elif kmeans_init_method == "kpp":
        centroids = kpp_parallel_sampled(X, n_centroids)
    elif kmeans_init_method == "mahalanobis":
        centroids = mahalanobis_init(X, n_centroids)
    else:
        raise ValueError(f"Unkown k-means init method: {kmeans_init_method}")
    
    assignments = kmeans_vq(
        X,
        centroids,
        iters=kmeans_iters,
    )
    
    return assignments, centroids
