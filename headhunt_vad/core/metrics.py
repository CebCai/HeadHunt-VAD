"""Saliency metrics for attention head analysis."""

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel


def calculate_lda_score(
    pos_data: np.ndarray,
    neg_data: np.ndarray,
    reg: float = 1e-6,
) -> float:
    """
    Computes the Linear Discriminant Analysis (LDA) score using Fisher's criterion.

    Args:
        pos_data: Normal class samples of shape (n_pos, dim).
        neg_data: Anomaly class samples of shape (n_neg, dim).
        reg: Regularization for covariance matrix.

    Returns:
        LDA score (higher indicates better separability).
    """
    if len(pos_data) < 2 or len(neg_data) < 2:
        return 0.0

    d = pos_data.shape[1]

    # Class means
    mu_p = np.mean(pos_data, axis=0)
    mu_n = np.mean(neg_data, axis=0)

    # Class covariances
    cov_p = np.cov(pos_data, rowvar=False)
    cov_n = np.cov(neg_data, rowvar=False)

    # Handle 1D case
    if cov_p.ndim == 0:
        cov_p = np.array([[cov_p]])
        cov_n = np.array([[cov_n]])

    # Within-class scatter matrix
    S_w = (len(pos_data) - 1) * cov_p + (len(neg_data) - 1) * cov_n
    S_w += np.eye(d) * reg

    try:
        S_w_inv = np.linalg.pinv(S_w)
        w = S_w_inv @ (mu_p - mu_n)
    except np.linalg.LinAlgError:
        return 0.0

    # Fisher criterion: between-class scatter / within-class scatter
    between_class_scatter = (w.T @ (mu_p - mu_n)) ** 2
    within_class_scatter = w.T @ S_w @ w

    if within_class_scatter < 1e-9:
        return 0.0

    return float(between_class_scatter / within_class_scatter)


def calculate_kl_divergence(
    pos_data: np.ndarray,
    neg_data: np.ndarray,
    reg: float = 1e-10,
) -> float:
    """
    Computes the symmetric Kullback-Leibler divergence using Ledoit-Wolf covariance estimation.

    Args:
        pos_data: Positive class samples of shape (n_pos, dim).
        neg_data: Negative class samples of shape (n_neg, dim).
        reg: Regularization for numerical stability.

    Returns:
        Symmetric KL divergence.
    """
    if len(pos_data) < 2 or len(neg_data) < 2:
        return 0.0

    d = pos_data.shape[1]

    # Use Ledoit-Wolf for stable covariance estimation
    try:
        cov1 = LedoitWolf().fit(pos_data).covariance_ + np.eye(d) * reg
        cov2 = LedoitWolf().fit(neg_data).covariance_ + np.eye(d) * reg
    except Exception:
        return 0.0

    mu1 = np.mean(pos_data, axis=0)
    mu2 = np.mean(neg_data, axis=0)

    try:
        # Log determinants
        _, ldc1 = np.linalg.slogdet(cov1)
        _, ldc2 = np.linalg.slogdet(cov2)

        # Inverse covariances
        inv_cov1 = np.linalg.inv(cov1)
        inv_cov2 = np.linalg.inv(cov2)
    except np.linalg.LinAlgError:
        return 0.0

    # KL(P || Q)
    kl_pq = 0.5 * (
        ldc2 - ldc1 - d +
        np.trace(inv_cov2 @ cov1) +
        (mu2 - mu1).T @ inv_cov2 @ (mu2 - mu1)
    )

    # KL(Q || P)
    kl_qp = 0.5 * (
        ldc1 - ldc2 - d +
        np.trace(inv_cov1 @ cov2) +
        (mu1 - mu2).T @ inv_cov1 @ (mu1 - mu2)
    )

    # Symmetric KL
    return float(np.clip(0.5 * (kl_pq + kl_qp), 0, 1e6))


def calculate_mmd(
    pos_data: np.ndarray,
    neg_data: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD) with RBF kernel.

    Args:
        pos_data: Positive class samples of shape (n_pos, dim).
        neg_data: Negative class samples of shape (n_neg, dim).
        gamma: RBF kernel parameter. If None, uses median heuristic.

    Returns:
        MMD score.
    """
    if len(pos_data) < 2 or len(neg_data) < 2:
        return 0.0

    # Estimate gamma using median heuristic
    if gamma is None:
        X = np.vstack([pos_data, neg_data])
        dists_sq = np.sum(
            (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2,
            axis=-1
        )
        median_dist_sq = np.median(dists_sq[dists_sq > 0])
        gamma = 1.0 / (2 * median_dist_sq) if median_dist_sq > 0 else 1.0

    # Compute kernel matrices
    K_pp = rbf_kernel(pos_data, pos_data, gamma=gamma).mean()
    K_qq = rbf_kernel(neg_data, neg_data, gamma=gamma).mean()
    K_pq = rbf_kernel(pos_data, neg_data, gamma=gamma).mean()

    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    mmd_sq = K_pp + K_qq - 2 * K_pq

    return float(np.clip(mmd_sq, 0, None))


def calculate_nmi(
    pos_data: np.ndarray,
    neg_data: np.ndarray,
    random_state: int = 42,
) -> float:
    """
    Computes Normalized Mutual Information (NMI) based on K-Means clustering.

    Args:
        pos_data: Positive class samples of shape (n_pos, dim).
        neg_data: Negative class samples of shape (n_neg, dim).
        random_state: Random state for KMeans.

    Returns:
        NMI score in [0, 1].
    """
    if len(pos_data) < 2 or len(neg_data) < 2:
        return 0.0

    # Combine data
    X = np.vstack([pos_data, neg_data])
    y_true = np.hstack([np.zeros(len(pos_data)), np.ones(len(neg_data))])

    # Perform k-means clustering
    try:
        kmeans = KMeans(n_clusters=2, random_state=random_state, n_init="auto")
        y_pred = kmeans.fit_predict(X)
    except Exception:
        return 0.0

    # Calculate NMI
    return float(normalized_mutual_info_score(y_true, y_pred))


def compute_all_metrics(
    pos_data: np.ndarray,
    neg_data: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute all four saliency metrics.

    Args:
        pos_data: Positive class samples.
        neg_data: Negative class samples.

    Returns:
        Tuple of (lda_score, kl_score, mmd_score, nmi_score).
    """
    lda = calculate_lda_score(pos_data, neg_data)
    kl = calculate_kl_divergence(pos_data, neg_data)
    mmd = calculate_mmd(pos_data, neg_data)
    nmi = calculate_nmi(pos_data, neg_data)

    return lda, kl, mmd, nmi
