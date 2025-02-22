
# Convert saliency map to potential map using GMM
from sklearn.mixture import GaussianMixture
import torch

def saliency_to_potential_EM(saliency_map, n_components):
    """
    Estimates the potential map from a saliency map using a Gaussian Mixture Model (GMM) via EM.

    Args:
        saliency_map (Tensor): Input saliency map (HxW).
        n_components (int): Number of Gaussian components for the GMM.

    Returns:
        Tensor: Continuous potential map U(x) =  log p(x).
    """
    device = saliency_map.device
    saliency_flat = saliency_map.cpu().numpy().flatten().reshape(-1, 1)  # Flatten & reshape for sklearn

    # Fit GMM using Expectation-Maximization
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", max_iter=100)
    gmm.fit(saliency_flat)

    # Compute log-likelihood p(x)
    log_probs = gmm.score_samples(saliency_flat)  # log p(x)

    #print(log_probs)

    # Convert log-likelihood to potential U(x) = log p(x)
    potential_map = torch.tensor(log_probs, dtype=torch.float32, device=device)
    potential_map = potential_map.view(saliency_map.shape)  # Reshape to original map size

    return potential_map
