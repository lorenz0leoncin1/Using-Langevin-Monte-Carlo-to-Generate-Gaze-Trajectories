
# Convert saliency map to potential map using GMM
import traceback
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


class GMM_Potential(nn.Module):
    def __init__(self, n_components):
        """
        Gaussian Mixture Model (GMM) potential function.

        Args:
            n_components (int): Number of Gaussian components in the mixture.
        """
        super(GMM_Potential, self).__init__()
        self.n_components = n_components
        
        # Gaussian parameters
        self.means = nn.Parameter(torch.randn(n_components))  # μ_k: Means of the Gaussians
        self.log_vars = nn.Parameter(torch.zeros(n_components))  # log(σ_k^2): Log-variances for stability
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)  # π_k: Normalized mixture weights

    def forward(self, x):
        """
        Computes the log-likelihood of the Gaussian Mixture Model (GMM) 
        and uses it as a potential function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Log-likelihood of the GMM for the given input.
        """
        x = x.unsqueeze(-1)  # Reshape for broadcasting
        
        # Compute Gaussian densities
        gaussians = -0.5 * ((x - self.means) ** 2) / self.log_vars.exp()  # (x - μ_k)^2 / σ_k^2
        gaussians -= 0.5 * self.log_vars  # Normalization term
        
        # Combine with mixture weights using log-sum-exp for numerical stability
        log_probs = torch.logsumexp(gaussians + self.weights.log(), dim=-1)
        
        return log_probs

def saliency_to_potential(saliency_map, n_components, lr=0.1, epochs=100):
    """
    Estimates the potential map from a saliency map using a Gaussian Mixture Model (GMM) in PyTorch.

    Args:
        saliency_map (Tensor): Input saliency map.
        n_components (int): Number of Gaussian components for the GMM.
        lr (float, optional): Learning rate for optimization. Default is 0.1.
        epochs (int, optional): Number of optimization steps. Default is 100.

    Returns:
        Tensor: Continuous and differentiable potential map.
    """
    device = saliency_map.device
    model = GMM_Potential(n_components).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    saliency_flat = saliency_map.flatten()

    # Train the GMM by maximizing the log-likelihood
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = -model(saliency_flat).mean()  # Maximize log-likelihood
        loss.backward()
        optimizer.step()

    # The resulting potential map is continuous and differentiable
    potential_map = model(saliency_map)
    return potential_map

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

def potential(potential_map, z):
    """
    Computes the potential at given coordinates using differentiable bilinear interpolation.

    Args:
        potential_map (Tensor): Precomputed potential map (assumed to have shape [B, C, H, W]).
        z (Tensor): Coordinates where the potential is evaluated (shape [N, 2] with (x, y) positions).

    Returns:
        Tensor: Interpolated potential values.
    """
    x, y = z[:, 0], z[:, 1]

    # Clip coordinates to stay within valid interpolation bounds
    x = torch.clamp(x, 0, potential_map.shape[2] - 2)  # -2 ensures valid interpolation
    y = torch.clamp(y, 0, potential_map.shape[3] - 2)

    # Compute integer grid points surrounding the target coordinates
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    # Compute bilinear interpolation weights
    wx1 = x - x0.float()
    wx0 = 1 - wx1
    wy1 = y - y0.float()
    wy0 = 1 - wy1

    # Perform bilinear interpolation
    val = (potential_map[:, :, x0, y0] * (wx0 * wy0).unsqueeze(1) +
           potential_map[:, :, x1, y0] * (wx1 * wy0).unsqueeze(1) +
           potential_map[:, :, x0, y1] * (wx0 * wy1).unsqueeze(1) +
           potential_map[:, :, x1, y1] * (wx1 * wy1).unsqueeze(1))

    return val.sum()

def unadjusted_langevin_algorithm(potential_map, init_point, n_steps, step_size, burn_in, ratio):
    """
    Implements the Unadjusted Langevin Algorithm (ULA) with a differentiable gradient.

    Args:
        potential_map (Tensor): Precomputed potential map.
        init_point (tuple or list): Initial (x, y) coordinates.
        n_steps (int): Number of Langevin steps.
        step_size (float): Step size for the Langevin update.
        burn_in (int): Number of initial samples to discard.
        ratio (float): Scaling factor for the final trajectory.

    Returns:
        np.ndarray: Sampled trajectory after burn-in.
    """                             
    Zi = torch.tensor(init_point, dtype=torch.float32, requires_grad=True).view(1, -1)
    
    # Define the limits of the downsampled image
    height, width = potential_map.shape[2], potential_map.shape[3]

    samples = []
    
    for i in tqdm(range(n_steps + burn_in), desc="ULA Simulation"):
        Zi.requires_grad_(True)
        
        # Compute potential and its gradient
        u = potential(potential_map, Zi)
        grad = torch.autograd.grad(u, Zi)[0]

        # Langevin update step
        noise = torch.randn_like(Zi) * np.sqrt(2 * step_size)
        Zi = Zi.detach() - step_size * grad + noise
        
        # Clip coordinates to stay within valid map bounds
        Zi_clipped = torch.clone(Zi)
        Zi_clipped[:, 0] = torch.clamp(Zi_clipped[:, 0], 0, height - 1)
        Zi_clipped[:, 1] = torch.clamp(Zi_clipped[:, 1], 0, width - 1)
        
        # Store the scaled position
        scaled_point = (Zi_clipped * ratio).detach().numpy()
        samples.append(scaled_point)
        
        Zi = Zi_clipped  # Update the current position
    
    return np.concatenate(samples, 0)[burn_in:]


def log_Q(potential_map, potential, z_prime, z, step_size):
    """
    Computes the log-ratio of the proposal distribution Q in MALA.

    Args:
        potential_map (Tensor): Precomputed potential map.
        potential (function): Function to compute potential values.
        z_prime (Tensor): Proposed sample.
        z (Tensor): Current sample.
        step_size (float): Step size for the Langevin update.

    Returns:
        Tensor: Log probability ratio.
    """
    z.requires_grad_()
    grad = torch.autograd.grad(potential(potential_map, z).mean(), z)[0]
    return -(torch.norm(z_prime - z + step_size * grad, p=2, dim=1) ** 2) / (4 * step_size)


def metropolis_adjusted_langevin_algorithm(potential_map, init_point, n_steps, step_size, burn_in, ratio):
    """
    Implements the Metropolis-Adjusted Langevin Algorithm (MALA).

    Args:
        potential_map (Tensor): Precomputed potential map.
        init_point (tuple or list): Initial (x, y) coordinates.
        n_steps (int): Number of Langevin steps.
        step_size (float): Step size for the Langevin update.
        burn_in (int): Number of initial samples to discard.
        ratio (tuple): Scaling factors for x and y.

    Returns:
        list: Accepted trajectory points after burn-in.
    """
    trajectory = []
    
    Zi = torch.tensor(init_point, dtype=torch.float32, requires_grad=True).view(1, -1)  # Ensure shape (1, 2)
    
    for step in tqdm(range(n_steps + burn_in), desc="MALA Simulation"):
        try:
            Zi.requires_grad_()
            
            # Compute potential and gradient
            u = potential(potential_map, Zi).mean()
            grad = torch.autograd.grad(u, Zi)[0]
            
            # Propose a new point using the Langevin update
            prop_Zi = Zi.detach() - step_size * grad + np.sqrt(2 * step_size) * torch.randn(1, 2)

            # Clip proposed point to map boundaries
            prop_Zi[:, 0] = torch.clamp(prop_Zi[:, 0], 0, potential_map.shape[2] - 1)
            prop_Zi[:, 1] = torch.clamp(prop_Zi[:, 1], 0, potential_map.shape[3] - 1)

            # Compute acceptance probability
            current_potential = potential(potential_map, Zi).mean()
            proposed_potential = potential(potential_map, prop_Zi).mean()
            
            log_ratio = (-proposed_potential + current_potential 
                         + log_Q(potential_map, potential, Zi, prop_Zi, step_size) 
                         - log_Q(potential_map, potential, prop_Zi, Zi, step_size))

            acceptance_ratio = torch.exp(log_ratio)

            # Accept or reject the proposal
            if torch.rand(1) < acceptance_ratio:
                Zi = prop_Zi  # Accept the proposal
                
            # Save the current position after burn-in
            if step >= burn_in:
                noisy_point = Zi.clone().detach().view(-1).numpy()  # Ensure shape (2,)
                noisy_point[0] *= ratio[1]  # Scale x coordinate
                noisy_point[1] *= ratio[0]  # Scale y coordinate
                trajectory.append(tuple(noisy_point))
                
        except Exception as e:
            print(f"Error during MALA: {e}")
            traceback.print_exc()  # Print full error traceback
            break
    
    return trajectory


def log_Q_cauchy(potential_map, potential, z_prime, z, step_size, gamma):
    """
    Computes the log-ratio of the proposal distribution Q using a Cauchy distribution.

    Args:
        potential_map (Tensor): Precomputed potential map.
        potential (function): Function to compute potential values.
        z_prime (Tensor): Proposed sample.
        z (Tensor): Current sample.
        step_size (float): Step size for the Langevin update.
        gamma (float): Scale parameter of the Cauchy distribution.

    Returns:
        Tensor: Log probability ratio of the proposal.
    """
    z.requires_grad_()
    grad = torch.autograd.grad(potential(potential_map, z).mean(), z)[0]

    # Compute log-likelihood of the proposal under the Cauchy distribution
    diff = z_prime - z + step_size * grad
    log_q = -torch.log(torch.pi * gamma) - torch.log(1 + (diff / gamma) ** 2).sum()
    
    return log_q


def metropolis_adjusted_langevin_algorithm_cauchy(potential_map, init_point, n_steps, step_size, burn_in, ratio, gamma):
    """
    Implements the Metropolis-Adjusted Langevin Algorithm (MALA) using a Cauchy proposal distribution.

    Args:
        potential_map (Tensor): Precomputed potential map.
        init_point (tuple or list): Initial (x, y) coordinates.
        n_steps (int): Number of Langevin steps.
        step_size (float): Step size for the Langevin update.
        burn_in (int): Number of initial samples to discard.
        ratio (tuple): Scaling factors for x and y.
        gamma (float): Scale parameter of the Cauchy distribution.

    Returns:
        list: Accepted trajectory points after burn-in.
    """
    trajectory = []

    Zi = torch.tensor(init_point, dtype=torch.float32, requires_grad=True).view(1, -1)  # Ensure shape (1, 2)

    for step in tqdm(range(n_steps + burn_in), desc="MALA Cauchy Simulation"):
        try:
            Zi.requires_grad_()
            
            # Compute potential and gradient
            u = potential(potential_map, Zi).mean()
            grad = torch.autograd.grad(u, Zi)[0]

            # Propose a new point using Langevin dynamics with Cauchy noise
            noise = gamma * torch.tan(torch.pi * (torch.rand(1, 2) - 0.5))  # Sample from standard Cauchy
            prop_Zi = Zi.detach() - step_size * grad + noise

            # Clip proposed point to map boundaries
            prop_Zi[:, 0] = torch.clamp(prop_Zi[:, 0], 0, potential_map.shape[2] - 1)
            prop_Zi[:, 1] = torch.clamp(prop_Zi[:, 1], 0, potential_map.shape[3] - 1)
            
            # Compute potentials for current and proposed points
            current_potential = potential(potential_map, Zi).mean()
            proposed_potential = potential(potential_map, prop_Zi).mean()
            
            # Compute log acceptance ratio
            log_ratio = (-proposed_potential + current_potential 
                         + log_Q_cauchy(potential_map, potential, Zi, prop_Zi, step_size, gamma) 
                         - log_Q_cauchy(potential_map, potential, prop_Zi, Zi, step_size, gamma))

            acceptance_ratio = torch.exp(log_ratio)

            # Accept or reject the proposed point
            if torch.rand(1) < acceptance_ratio:
                Zi = prop_Zi  # Accept the proposal
                
            # Save the current position after burn-in
            if step >= burn_in:
                noisy_point = Zi.clone().detach().view(-1).numpy()  # Ensure shape (2,)
                noisy_point[0] *= ratio[1]  # Scale x coordinate
                noisy_point[1] *= ratio[0]  # Scale y coordinate
                
                trajectory.append(tuple(noisy_point))
                
        except Exception as e:
            print(f"Error during MALA Cauchy: {e}")
            traceback.print_exc()  # Print full error traceback
            break
    
    return trajectory