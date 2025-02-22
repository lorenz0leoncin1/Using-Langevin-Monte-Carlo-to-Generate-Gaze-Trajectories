import torch
import numpy as np
from tqdm import tqdm

class ULA:
    def __init__(self, potential_map, init_point, n_steps, step_size, burn_in, ratio):
        """
        Initializes the ULA class.

        Args:
            potential_map (Tensor): Precomputed potential map.
            init_point (tuple or list): Initial (x, y) coordinates.
            n_steps (int): Number of Langevin steps.
            step_size (float): Step size for the Langevin update.
            burn_in (int): Number of initial samples to discard.
            ratio (float): Scaling factor for the final trajectory.
        """
        self.potential_map = potential_map
        self.init_point = torch.tensor(init_point, dtype=torch.float32).view(1, -1)
        self.n_steps = n_steps
        self.step_size = step_size
        self.burn_in = burn_in
        self.ratio = ratio
        self.height, self.width = potential_map.shape[2], potential_map.shape[3]

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
        x = torch.clamp(x, 0, self.height - 2)
        y = torch.clamp(y, 0, self.width - 2)

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

    def simulate_scanpath(self):
        """
        Runs the Unadjusted Langevin Algorithm (ULA) simulation.

        Returns:
            np.ndarray: Sampled trajectory after burn-in.
        """
        Zi = self.init_point.clone().requires_grad_(True)
        samples = []
        for step in range(self.n_steps + self.burn_in):
        #for _ in tqdm(range(self.n_steps + self.burn_in), desc="ULA Simulation"):
            Zi.requires_grad_(True)

            # Compute potential and its gradient
            u = self.potential(Zi)
            grad = torch.autograd.grad(u, Zi)[0]

            # Langevin update step
            noise = torch.randn_like(Zi) * np.sqrt(2 * self.step_size)
            Zi = Zi.detach() - self.step_size * grad + noise

            # Clip coordinates to stay within valid map bounds
            Zi[:, 0] = torch.clamp(Zi[:, 0], 0, self.height - 1)
            Zi[:, 1] = torch.clamp(Zi[:, 1], 0, self.width - 1)
            
            if step >= self.burn_in:
                scaled_point = Zi.clone().detach().view(-1).numpy()
                scaled_point[0] *= self.ratio[1]
                scaled_point[1] *= self.ratio[0]
                samples.append(scaled_point)

        return np.array(samples)
