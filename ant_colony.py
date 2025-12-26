"""
Ant Colony Optimization Module
Ant Colony Optimization (ACO) for TSP.

Notes
-----
- Classic Ant System (AS) implementation with pheromone evaporation and
  probabilistic path construction (alpha/beta parameters).
- Supports 'seeding': if an initial tour is provided, it adds a small pheromone
  bias to that route to help guide the initial search.
- Records the global best tour at every iteration to drive the animation
  and detect stagnation.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from tsp_reader import calculate_tour_length


class AntColonyOptimization:
    """Ant Colony Optimization (ACO) for solving TSP."""

    def __init__(
        self,
        dist_matrix,
        n_ants: int = 30,
        n_iterations: int = 300,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.5,
        Q: float = 100.0,
        seed: Optional[int] = None,
    ):
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)

        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.evaporation_rate = float(evaporation_rate)
        self.Q = float(Q)

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.pheromones = np.ones((self.n_cities, self.n_cities), dtype=float)

        # Stores (best_tour, best_length) at each iteration (best-so-far)
        self.history: List[Tuple[List[int], float]] = []

    def _deposit_pheromone_for_tour(self, tour: List[int], length: float) -> None:
        deposit = self.Q / max(length, 1e-12)
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i + 1) % len(tour)]
            self.pheromones[a, b] += deposit
            self.pheromones[b, a] += deposit

    def construct_tour(self) -> List[int]:
        """Construct a tour using probabilistic transitions based on pheromones and heuristics."""
        tour: List[int] = []
        available = set(range(self.n_cities))

        current = self.rng.randrange(self.n_cities)
        tour.append(current)
        available.remove(current)

        while available:
            avail_list = list(available)

            # Heuristic: eta = 1 / distance (avoid division by zero)
            dists = np.array([self.dist_matrix[current, j] for j in avail_list], dtype=float)
            eta = 1.0 / np.maximum(dists, 1e-12)

            tau = np.array([self.pheromones[current, j] for j in avail_list], dtype=float)

            weights = (tau ** self.alpha) * (eta ** self.beta)
            s = float(weights.sum())
            if s <= 0.0 or not np.isfinite(s):
                next_city = self.rng.choice(avail_list)
            else:
                probs = weights / s
                next_city = int(self.np_rng.choice(avail_list, p=probs))

            tour.append(next_city)
            available.remove(next_city)
            current = next_city

        return tour

    def update_pheromones(self, all_tours: List[List[int]], all_lengths: List[float]) -> None:
        """Evaporate and deposit pheromones."""
        self.pheromones *= (1.0 - self.evaporation_rate)

        for tour, length in zip(all_tours, all_lengths):
            self._deposit_pheromone_for_tour(tour, float(length))

        # Keep pheromone values in a reasonable range
        np.clip(self.pheromones, 1e-12, 1e6, out=self.pheromones)

    def solve(
            self,
            initial_tour: Optional[List[int]] = None,
            stagnation_patience: int = 60,
        ) -> Tuple[List[int], float]:
            """
            Solve TSP using ACO.

            Parameters
            ----------
            initial_tour:
                Shared starting route (visual baseline + optional initial pheromone bias).
            stagnation_patience:
                Stop if best distance doesn't improve for this many iterations.

            Returns
            -------
            (best_tour, best_length)
            """
            # Reset pheromones and history for a clean run
            self.pheromones = np.ones((self.n_cities, self.n_cities), dtype=float)
            self.history = []

            if initial_tour is not None:
                if len(initial_tour) != self.n_cities:
                    raise ValueError("initial_tour length must match number of cities")
                best_tour = list(initial_tour)
                best_length = float(calculate_tour_length(best_tour, self.dist_matrix))
                # Small initial bias so the algorithm is 'seeded' with the same baseline.
                self._deposit_pheromone_for_tour(best_tour, best_length)

                # Baseline (frame 0)
                self.history.append((best_tour.copy(), best_length))
            else:
                best_tour = None
                best_length = float("inf")

            no_improve = 0

            for _it in range(self.n_iterations):
                improved_this_iter = False
                all_tours: List[List[int]] = []
                all_lengths: List[float] = []

                for _ in range(self.n_ants):
                    tour = self.construct_tour()
                    length = float(calculate_tour_length(tour, self.dist_matrix))
                    all_tours.append(tour)
                    all_lengths.append(length)

                    if length < best_length:
                        best_tour = tour
                        best_length = length
                        improved_this_iter = True

                self.update_pheromones(all_tours, all_lengths)

                if best_tour is None:
                    best_tour = all_tours[int(np.argmin(all_lengths))]
                    best_length = float(min(all_lengths))

                # Record best-so-far for animation (even if unchanged)
                self.history.append((list(best_tour), float(best_length)))

                if improved_this_iter:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= int(stagnation_patience):
                        break

            return list(best_tour), float(best_length)
