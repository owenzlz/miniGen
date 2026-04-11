"""
2D Synthetic datasets for diffusion model experiments.

Supports various distributions: Gaussian mixture, Swiss roll, moons, circles,
checkerboard, spirals, pinwheel, rings, S-curve.
Great for visualizing and understanding diffusion dynamics.
"""
import math
import torch
from typing import Literal


class Synthetic2DDataset(torch.utils.data.Dataset):
    """
    2D synthetic dataset with various distributions.

    Each sample is a 2D point with an optional class label.
    """

    def __init__(
        self,
        distribution: Literal[
            "gaussian_mixture", "swiss_roll", "moons", "circles",
            "checkerboard", "spirals", "pinwheel", "rings", "s_curve",
        ] = "gaussian_mixture",
        num_samples: int = 100000,
        num_classes: int = 8,
        seed: int = 0,
    ):
        """
        Args:
            distribution: Type of distribution to sample from.
            num_samples: Number of samples in the dataset.
            num_classes: Number of classes (modes) for gaussian_mixture.
            seed: Random seed for reproducibility.
        """
        self.distribution = distribution
        self.num_samples = num_samples
        self.num_classes = num_classes

        # Pre-generate all samples for consistency
        generator = torch.Generator().manual_seed(seed)

        if distribution == "gaussian_mixture":
            self.points, self.labels = self._generate_gaussian_mixture(
                num_samples, num_classes, generator
            )
        elif distribution == "swiss_roll":
            self.points, self.labels = self._generate_swiss_roll(
                num_samples, generator
            )
        elif distribution == "moons":
            self.points, self.labels = self._generate_moons(
                num_samples, generator
            )
        elif distribution == "circles":
            self.points, self.labels = self._generate_circles(
                num_samples, generator
            )
        elif distribution == "checkerboard":
            self.points, self.labels = self._generate_checkerboard(
                num_samples, generator
            )
        elif distribution == "spirals":
            self.points, self.labels = self._generate_spirals(
                num_samples, generator
            )
        elif distribution == "pinwheel":
            self.points, self.labels = self._generate_pinwheel(
                num_samples, generator
            )
        elif distribution == "rings":
            self.points, self.labels = self._generate_rings(
                num_samples, generator
            )
        elif distribution == "s_curve":
            self.points, self.labels = self._generate_s_curve(
                num_samples, generator
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Normalize to [-1, 1] for compatibility with diffusion (which clips to [-1, 1])
        # Store normalization params for denormalization during visualization
        self.data_min = self.points.min(dim=0).values
        self.data_max = self.points.max(dim=0).values
        self.data_range = self.data_max - self.data_min
        self.data_center = (self.data_max + self.data_min) / 2

        # Normalize: map [min, max] -> [-1, 1]
        self.points = 2 * (self.points - self.data_min) / self.data_range - 1

    def _generate_gaussian_mixture(
        self, n: int, num_modes: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from a Gaussian mixture arranged in a circle."""
        if num_modes <= 0:
            num_modes = 8
        # Assign each sample to a mode
        labels = torch.randint(0, num_modes, (n,), generator=generator)

        # Mode centers arranged in a circle
        angles = 2 * math.pi * torch.arange(num_modes) / num_modes
        radius = 2.0
        centers_x = radius * torch.cos(angles)
        centers_y = radius * torch.sin(angles)

        # Sample from Gaussians centered at each mode
        std = 0.2
        noise = torch.randn(n, 2, generator=generator) * std

        points = torch.stack([
            centers_x[labels] + noise[:, 0],
            centers_y[labels] + noise[:, 1],
        ], dim=1)

        return points, labels

    def _generate_swiss_roll(
        self, n: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from a 2D Swiss roll."""
        t = 1.5 * math.pi * (1 + 2 * torch.rand(n, generator=generator))
        x = t * torch.cos(t)
        y = t * torch.sin(t)

        # Add noise
        noise = 0.1 * torch.randn(n, 2, generator=generator)
        points = torch.stack([x, y], dim=1) / 5.0 + noise  # Scale down

        # Labels based on t value (discretize into classes)
        labels = ((t - t.min()) / (t.max() - t.min()) * (self.num_classes - 1)).long()

        return points, labels

    def _generate_moons(
        self, n: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from two interleaving half circles."""
        n_per_moon = n // 2

        # First moon
        theta1 = math.pi * torch.rand(n_per_moon, generator=generator)
        x1 = torch.cos(theta1)
        y1 = torch.sin(theta1)

        # Second moon (shifted and flipped)
        theta2 = math.pi * torch.rand(n - n_per_moon, generator=generator)
        x2 = 1 - torch.cos(theta2)
        y2 = 1 - torch.sin(theta2) - 0.5

        points = torch.cat([
            torch.stack([x1, y1], dim=1),
            torch.stack([x2, y2], dim=1),
        ], dim=0)

        labels = torch.cat([
            torch.zeros(n_per_moon, dtype=torch.long),
            torch.ones(n - n_per_moon, dtype=torch.long),
        ])

        # Add noise and center
        noise = 0.05 * torch.randn(n, 2, generator=generator)
        points = points + noise
        points = points - points.mean(dim=0)

        # Shuffle
        perm = torch.randperm(n, generator=generator)
        points = points[perm]
        labels = labels[perm]

        return points, labels

    def _generate_circles(
        self, n: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from concentric circles."""
        n_outer = n // 2
        n_inner = n - n_outer

        # Outer circle
        theta_outer = 2 * math.pi * torch.rand(n_outer, generator=generator)
        r_outer = 1.0 + 0.05 * torch.randn(n_outer, generator=generator)
        x_outer = r_outer * torch.cos(theta_outer)
        y_outer = r_outer * torch.sin(theta_outer)

        # Inner circle
        theta_inner = 2 * math.pi * torch.rand(n_inner, generator=generator)
        r_inner = 0.5 + 0.05 * torch.randn(n_inner, generator=generator)
        x_inner = r_inner * torch.cos(theta_inner)
        y_inner = r_inner * torch.sin(theta_inner)

        points = torch.cat([
            torch.stack([x_outer, y_outer], dim=1),
            torch.stack([x_inner, y_inner], dim=1),
        ], dim=0)

        labels = torch.cat([
            torch.zeros(n_outer, dtype=torch.long),
            torch.ones(n_inner, dtype=torch.long),
        ])

        # Shuffle
        perm = torch.randperm(n, generator=generator)
        points = points[perm]
        labels = labels[perm]

        return points, labels

    def _generate_checkerboard(
        self, n: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate uniform samples in alternating filled squares of a 4x4 grid."""
        grid_size = 4
        # Enumerate filled squares in a checkerboard pattern
        filled = []
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:
                    filled.append((i, j))

        n_cells = len(filled)
        per_cell = n // n_cells
        remainder = n - per_cell * n_cells

        all_points = []
        all_labels = []
        for idx, (i, j) in enumerate(filled):
            count = per_cell + (1 if idx < remainder else 0)
            x = torch.rand(count, generator=generator) + i
            y = torch.rand(count, generator=generator) + j
            all_points.append(torch.stack([x, y], dim=1))
            all_labels.append(torch.full((count,), idx, dtype=torch.long))

        points = torch.cat(all_points, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Shuffle
        perm = torch.randperm(n, generator=generator)
        return points[perm], labels[perm]

    def _generate_spirals(
        self, n: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate two interleaving Archimedean spirals."""
        n_per = n // 2

        # Spiral 1
        t1 = torch.linspace(0, 3 * math.pi, n_per)
        r1 = t1 / (3 * math.pi)  # radius grows from 0 to 1
        x1 = r1 * torch.cos(t1)
        y1 = r1 * torch.sin(t1)

        # Spiral 2 (rotated by pi)
        n2 = n - n_per
        t2 = torch.linspace(0, 3 * math.pi, n2)
        r2 = t2 / (3 * math.pi)
        x2 = r2 * torch.cos(t2 + math.pi)
        y2 = r2 * torch.sin(t2 + math.pi)

        points = torch.cat([
            torch.stack([x1, y1], dim=1),
            torch.stack([x2, y2], dim=1),
        ], dim=0)
        labels = torch.cat([
            torch.zeros(n_per, dtype=torch.long),
            torch.ones(n2, dtype=torch.long),
        ])

        # Add noise
        noise = 0.02 * torch.randn(n, 2, generator=generator)
        points = points + noise

        # Shuffle
        perm = torch.randperm(n, generator=generator)
        return points[perm], labels[perm]

    def _generate_pinwheel(
        self, n: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate rotated elongated Gaussian clusters in a pinwheel pattern."""
        num_blades = 5
        per_blade = n // num_blades
        remainder = n - per_blade * num_blades

        radial_std = 0.3
        tangential_std = 0.05

        all_points = []
        all_labels = []
        for k in range(num_blades):
            count = per_blade + (1 if k < remainder else 0)
            # Elongated Gaussian in local frame
            radial = torch.randn(count, generator=generator) * radial_std + 0.5
            tangential = torch.randn(count, generator=generator) * tangential_std

            angle = 2 * math.pi * k / num_blades
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x = radial * cos_a - tangential * sin_a
            y = radial * sin_a + tangential * cos_a

            all_points.append(torch.stack([x, y], dim=1))
            all_labels.append(torch.full((count,), k, dtype=torch.long))

        points = torch.cat(all_points, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Shuffle
        perm = torch.randperm(n, generator=generator)
        return points[perm], labels[perm]

    def _generate_rings(
        self, n: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from multiple evenly-spaced concentric rings."""
        num_rings = 5
        per_ring = n // num_rings
        remainder = n - per_ring * num_rings

        all_points = []
        all_labels = []
        for k in range(num_rings):
            count = per_ring + (1 if k < remainder else 0)
            radius = 0.2 + 0.2 * k  # radii: 0.2, 0.4, 0.6, 0.8, 1.0
            theta = 2 * math.pi * torch.rand(count, generator=generator)
            r = radius + 0.02 * torch.randn(count, generator=generator)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            all_points.append(torch.stack([x, y], dim=1))
            all_labels.append(torch.full((count,), k, dtype=torch.long))

        points = torch.cat(all_points, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Shuffle
        perm = torch.randperm(n, generator=generator)
        return points[perm], labels[perm]

    def _generate_s_curve(
        self, n: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate S-shaped distribution (sigmoid curve with noise)."""
        t = 3 * math.pi * (torch.rand(n, generator=generator) - 0.5)  # [-1.5pi, 1.5pi]
        x = torch.sin(t)
        y = t
        points = torch.stack([x, y], dim=1)

        # Add noise
        noise = 0.05 * torch.randn(n, 2, generator=generator)
        points = points + noise

        # Labels based on t value
        labels = (t > 0).long()

        return points, labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "tenPoints": self.points[idx],
            "intLabel": self.labels[idx].item(),
        }


def build_synthetic2d_dataloader(
    distribution: str = "gaussian_mixture",
    num_samples: int = 100000,
    num_classes: int = 8,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Build 2D synthetic dataloader.

    Args:
        distribution: Type of distribution ("gaussian_mixture", "swiss_roll", "moons",
            "circles", "checkerboard", "spirals", "pinwheel", "rings", "s_curve").
        num_samples: Number of samples in the dataset.
        num_classes: Number of classes/modes.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        seed: Random seed.

    Returns:
        DataLoader yielding batches with 'tenPoints' and 'intLabel'.
    """
    dataset = Synthetic2DDataset(
        distribution=distribution,
        num_samples=num_samples,
        num_classes=num_classes,
        seed=seed,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
