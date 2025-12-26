"""
Hill Climbing Algorithm Module
Simple hill climbing implementation for TSP.

Notes
-----
- Intentionally basic (random 2-swap neighborhood, first-improvement acceptance).
- Records the best-so-far tour at every iteration so the visualizer can
  show stagnation and stop cleanly.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from tsp_reader import calculate_tour_length


class HillClimbing:
    """Basic Hill Climbing algorithm for TSP."""

    def __init__(self, dist_matrix, max_iterations: int = 1000, seed: Optional[int] = None):
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
        self.max_iterations = int(max_iterations)
        self.rng = random.Random(seed)

        # Stores (tour, length) at each iteration (best-so-far)
        self.history: List[Tuple[List[int], float]] = []

    def solve(
        self,
        initial_tour: Optional[List[int]] = None,
        stagnation_patience: int = 150,
    ) -> Tuple[List[int], float]:
        """
        Solve TSP using hill climbing with random swaps.

        Parameters
        ----------
        initial_tour:
            Optional starting permutation. If None, a random tour is created.
        stagnation_patience:
            Stop if no improvement occurs for this many iterations.

        Returns
        -------
        (best_tour, best_length)
        """
        if initial_tour is None:
            current_tour = list(range(self.n_cities))
            self.rng.shuffle(current_tour)
        else:
            if len(initial_tour) != self.n_cities:
                raise ValueError("initial_tour length must match number of cities")
            current_tour = list(initial_tour)

        current_length = calculate_tour_length(current_tour, self.dist_matrix)

        self.history = [(current_tour.copy(), float(current_length))]

        no_improve = 0

        for _ in range(self.max_iterations):
            i, j = self.rng.sample(range(self.n_cities), 2)
            candidate = current_tour.copy()
            candidate[i], candidate[j] = candidate[j], candidate[i]

            cand_len = calculate_tour_length(candidate, self.dist_matrix)

            if cand_len < current_length:
                current_tour = candidate
                current_length = cand_len
                no_improve = 0
            else:
                no_improve += 1

            # Record best-so-far for animation (even if unchanged)
            self.history.append((current_tour.copy(), float(current_length)))

            if no_improve >= int(stagnation_patience):
                break

        return current_tour, float(current_length)
