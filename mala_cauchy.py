import torch
import numpy as np
from tqdm import tqdm
import traceback

class MALA_Cauchy:
    def __init__(self, potential_map, init_point, n_steps, step_size, burn_in, ratio, gamma):
        """
        Initializes the MALA_Cauchy class.

        Args:
            potential_map (Tensor): Precomputed potential map.
            init_point (tuple or list): Initial (x, y) coordinates.
            n_steps (int): Number of Langevin steps.
            step_size (float): Step size for the Langevin update.
            burn_in (int): Number of initial samples to discard.
            ratio (tuple): Scaling factors for x and y.
            gamma (float): Scale parameter of the Cauchy distribution.
        """
        self.potential_map = potential_map
        self.init_point = torch.tensor(init_point, dtype=torch.float32).view(1, -1)
        self.n_steps = n_steps
        self.step_size = step_size
        self.burn_in = burn_in
        self.ratio = ratio
        self.gamma = gamma
        self.trajectory = []


    def potential(self, z):
        """
        Computes the potential at given coordinates using bilinear interpolation.

        Args:
            z (Tensor): Coordinates where the potential is evaluated (shape [N, 2]).

        Returns:
            Tensor: Interpolated potential values.
        """
        x, y = z[:, 0], z[:, 1]

        # Clip coordinates to stay within valid interpolation bounds
        x = torch.clamp(x, 0, self.potential_map.shape[2] - 2)
        y = torch.clamp(y, 0, self.potential_map.shape[3] - 2)

        # Compute integer grid points surrounding the target coordinates
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        x1, y1 = x0 + 1, y0 + 1

        # Compute bilinear interpolation weights
        wx1, wy1 = x - x0.float(), y - y0.float()
        wx0, wy0 = 1 - wx1, 1 - wy1

        # Perform bilinear interpolation
        val = (
            self.potential_map[:, :, x0, y0] * (wx0 * wy0).unsqueeze(1) +
            self.potential_map[:, :, x1, y0] * (wx1 * wy0).unsqueeze(1) +
            self.potential_map[:, :, x0, y1] * (wx0 * wy1).unsqueeze(1) +
            self.potential_map[:, :, x1, y1] * (wx1 * wy1).unsqueeze(1)
        )

        return val.sum()

    def Q_cauchy(self, z_prime, z):
        """
        Computes the proposal distribution Q(z'|z) for MALA-Cauchy.
        
        Args:
            potential: potential function (U).
            z_prime: proposed position z'.
            z: current position z.
            step_size: step size of the algorithm.
            gamma: dispersion parameter (step).
            lambda_scale: scale parameter (lambda) for the Cauchy distribution.
        
        Returns:
            Q: value of the proposal distribution Q(z'|z).
        """
        # Compute the gradient of the potential
        z.requires_grad_()
        grad = torch.autograd.grad(self.potential(z).mean(), z)[0]
        
        # Calculate the distance between z' and z, plus the gradient term
        diff = z_prime - z + self.step_size * grad
        
        # Compute the Euclidean norm (similar to squared distance)
        diff_squared = torch.mean(diff ** 2, dim=1)
        
        # Calculate Q(z'|z) using the Cauchy distribution
        Q = 1 / (np.pi * self.gamma * (1 + (diff_squared / self.gamma)**2))
        
        return Q


    def simulate_scanpath(self):
        """
        Runs the Metropolis-Adjusted Langevin Algorithm (MALA) with a Cauchy proposal.

        Returns:
            list: Accepted trajectory points after burn-in.
        """

        Zi = self.init_point.clone().requires_grad_(True)
        for step in range(self.n_steps + self.burn_in):
        #for step in tqdm(range(self.n_steps + self.burn_in), desc="MALA Cauchy Simulation"):
            try:
                Zi.requires_grad_()
                
                # Compute potential and gradient
                u = self.potential(Zi).mean()
                grad = torch.autograd.grad(u, Zi)[0]

                # Propose a new point using Langevin dynamics with Cauchy noise
                noise = torch.distributions.Cauchy(0, self.gamma).sample((1, 2))# Sample from standard Cauchy
                prop_Zi = Zi.detach() - self.step_size * grad + noise

                # Clip proposed point to map boundaries
                prop_Zi[:, 0] = torch.clamp(prop_Zi[:, 0], 0, self.potential_map.shape[2] - 1)
                prop_Zi[:, 1] = torch.clamp(prop_Zi[:, 1], 0, self.potential_map.shape[3] - 1)
                
                # Compute potentials for current and proposed points
                current_potential = self.potential(Zi).mean()
                proposed_potential = self.potential(prop_Zi).mean()
                
                # Compute log acceptance ratio
                log_ratio = (-proposed_potential + current_potential 
                             + self.Q_cauchy(Zi, prop_Zi) 
                             - self.Q_cauchy(prop_Zi, Zi))

                acceptance_ratio = torch.exp(log_ratio)
    
                # Accept or reject the proposed point
                if torch.rand(1) < acceptance_ratio:
                    Zi = prop_Zi  # Accept the proposal
                    
                # Save the current position after burn-in
                if step >= self.burn_in:
                    noisy_point = Zi.clone().detach().view(-1).numpy()  # Ensure shape (2,)
                    noisy_point[0] *= self.ratio[1]  # Scale x coordinate
                    noisy_point[1] *= self.ratio[0]  # Scale y coordinate
                    
                    self.trajectory.append(tuple(noisy_point))
                
            except Exception as e:
                print(f"Error during MALA Cauchy: {e}")
                traceback.print_exc()  # Print full error traceback
                break
    
        return np.array(self.trajectory)
