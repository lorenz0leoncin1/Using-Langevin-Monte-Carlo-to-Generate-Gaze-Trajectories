import torch
import numpy as np
from tqdm import tqdm

class MALA:
    def __init__(self, potential_map, init_point, n_steps, step_size, burn_in, ratio):
        """
        Initializes the MALA simulation.

        Args:
            potential_map (Tensor): Precomputed potential map.
            init_point (tuple or list): Initial (x, y) coordinates.
            n_steps (int): Number of Langevin steps.
            step_size (float): Step size for the Langevin update.
            burn_in (int): Number of initial samples to discard.
            ratio (tuple): Scaling factors for x and y.
        """
        self.potential_map = potential_map
        self.init_point = init_point
        self.n_steps = n_steps
        self.step_size = step_size
        self.burn_in = burn_in
        self.ratio = ratio
        self.trajectory = []

    def potential(self, potential_map, z):
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

    def Q_norm(self, potential_map, potential, z_prime, z, step_size):
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

    def simulate_scanpath(self):
        """
        Runs the MALA simulation and returns the trajectory.

        Returns:
            list: Accepted trajectory points after burn-in.
        """
        Zi = torch.tensor(self.init_point, dtype=torch.float32, requires_grad=True).view(1, -1)  # Ensure shape (1, 2)
        
        for step in range(self.n_steps + self.burn_in):
        #for step in tqdm(range(self.n_steps + self.burn_in), desc="MALA Simulation"):
            try:
                Zi.requires_grad_()

                # Compute potential and gradient
                u = self.potential(self.potential_map, Zi).mean()
                grad = torch.autograd.grad(u, Zi)[0]

                # Propose a new point using the Langevin update
                prop_Zi = Zi.detach() - self.step_size * grad + np.sqrt(2 * self.step_size) * torch.randn(1, 2)

                # Clip proposed point to map boundaries
                prop_Zi[:, 0] = torch.clamp(prop_Zi[:, 0], 0, self.potential_map.shape[2] - 1)
                prop_Zi[:, 1] = torch.clamp(prop_Zi[:, 1], 0, self.potential_map.shape[3] - 1)

                # Compute acceptance probability
                current_potential = self.potential(self.potential_map, Zi).mean()
                proposed_potential = self.potential(self.potential_map, prop_Zi).mean()

                log_ratio = (-proposed_potential + current_potential 
                             + self.Q_norm(self.potential_map, self.potential, Zi, prop_Zi, self.step_size) 
                             - self.Q_norm(self.potential_map, self.potential, prop_Zi, Zi, self.step_size))

                acceptance_ratio = torch.exp(log_ratio)

                # Accept or reject the proposal
                if torch.rand(1) < acceptance_ratio:
                    Zi = prop_Zi  # Accept the proposal

                # Save the current position after burn-in
                if step >= self.burn_in:
                    noisy_point = Zi.clone().detach().view(-1).numpy()  # Ensure shape (2,)
                    noisy_point[0] *= self.ratio[1]  # Scale x coordinate
                    noisy_point[1] *= self.ratio[0]  # Scale y coordinate
                    self.trajectory.append(tuple(noisy_point))

            except Exception as e:
                print(f"Error during MALA: {e}")
                break
        
        return np.array(self.trajectory)
