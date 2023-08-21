import torch

def cdist(X, Y):
    """
    Pairwise distance between rows of X and rows of Y.

    :param X: First input matrix
    :type X: th.Tensor
    :param Y: Second input matrix
    :type Y: th.Tensor
    :return: Matrix containing pairwise distances between rows of X and rows of Y
    :rtype: th.Tensor
    """
    xyT = X @ torch.t(Y)
    x2 = torch.sum(X**2, dim=1, keepdim=True)
    y2 = torch.sum(Y**2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + torch.t(y2)
    return d

def _atleast_epsilon(X, eps=1e-9):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return torch.where(X < eps, X.new_tensor(eps), X)

def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=1e-9):
    """
    Compute a Gaussian kernel matrix from a distance matrix.

    :param dist: Disatance matrix
    :type dist: th.Tensor
    :param rel_sigma: Multiplication factor for the sigma hyperparameter
    :type rel_sigma: float
    :param min_sigma: Minimum value for sigma. For numerical stability.
    :type min_sigma: float
    :return: Kernel matrix
    :rtype: th.Tensor
    """
    # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
    dist = torch.nn.functional.relu(dist)
    a = torch.median(dist)
    sigma2 = rel_sigma * torch.median(dist)
    # Disable gradient for sigma
    sigma2 = sigma2.detach()
    sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
    k = torch.exp(- dist / (2 * sigma2))
    return k

def d_cs(A, K, n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = torch.t(A) @ K @ A
    dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=1e-9**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / torch.sqrt(dnom_squared))
    return d

def triu(X):
    # Sum of strictly upper triangular part
    return torch.sum(torch.triu(X, diagonal=1))