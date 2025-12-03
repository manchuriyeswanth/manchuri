import torch
import numpy as np

def compute_basis_from_activations(activations, r):
    """
    activations: numpy array or torch tensor of shape [N, d_model] (samples x d)
    returns: basis B of shape (d_model, r), orthonormal columns (torch.Tensor)
    """
    if isinstance(activations, torch.Tensor):
        A = activations.detach().cpu().numpy()
    else:
        A = np.array(activations)
    # center
    A = A - A.mean(axis=0, keepdims=True)
    # compute top-r left singular vectors via SVD on A (we want column-space)
    # Use econ SVD on (N x d): A = U S Vt; Vt has right singular vectors (d)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T  # d x d
    B = V[:, :r]  # d x r
    return torch.from_numpy(B.astype(np.float32))
