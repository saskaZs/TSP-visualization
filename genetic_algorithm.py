"""
Genetic Algorithm Module
Genetic algorithm implementation for TSP.

Notes
-----
- Standard GA implementation using Tournament Selection and Ordered Crossover (OX).
- Uses Elitism: the best individual from the previous generation is always preservation.
- Records the best-so-far tour at every generation so the visualizer can
  show progress smoothly and handle stagnation.

"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from tsp_reader import calculate_tour_length


class GeneticAlgorithm:
    """Genetic Algorithm (GA) for solving TSP."""

    def __init__(
        self,
        dist_matrix,
        population_size: int = 100,
        generations: int = 500,
        mutation_rate: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)

        self.population_size = int(population_size)
        self.generations = int(generations)
        self.mutation_rate = float(mutation_rate)

        self.rng = random.Random(seed)

        # Stores (best_tour, best_length) at each generation (best-so-far)
        self.history: List[Tuple[List[int], float]] = []

    def _random_tour(self) -> List[int]:
        tour = list(range(self.n_cities))
        self.rng.shuffle(tour)
        return tour

    def create_population(self, initial_tour: Optional[List[int]] = None) -> List[List[int]]:
        """Create initial population, ensuring `initial_tour` is included if provided."""
        population: List[List[int]] = []
        seen = set()

        if initial_tour is not None:
            if len(initial_tour) != self.n_cities:
                raise ValueError("initial_tour length must match number of cities")
            t0 = list(initial_tour)
            population.append(t0)
            seen.add(tuple(t0))

        # Fill rest randomly (use initial_tour as a base to shuffle, if given)
        while len(population) < self.population_size:
            if initial_tour is None:
                t = self._random_tour()
            else:
                t = list(initial_tour)
                self.rng.shuffle(t)
            key = tuple(t)
            if key in seen:
                continue
            seen.add(key)
            population.append(t)

        return population

    def selection(self, population: List[List[int]], tournament_k: int = 5) -> List[int]:
        """Tournament selection."""
        k = min(int(tournament_k), len(population))
        candidates = self.rng.sample(population, k)
        return min(candidates, key=lambda t: calculate_tour_length(t, self.dist_matrix))

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Ordered crossover (OX)."""
        size = len(parent1)
        start, end = sorted(self.rng.sample(range(size), 2))

        child = [-1] * size
        child[start : end + 1] = parent1[start : end + 1]

        p2_idx = 0
        for i in range(size):
            if child[i] != -1:
                continue
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
        return child

    def mutate(self, tour: List[int]) -> List[int]:
        """Swap mutation."""
        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    def solve(
        self,
        initial_tour: Optional[List[int]] = None,
        stagnation_patience: int = 80,
    ) -> Tuple[List[int], float]:
        """
        Solve TSP using a GA.

        Parameters
        ----------
        initial_tour:
            Shared starting route (for fair comparison/visual baseline).
        stagnation_patience:
            Stop if best distance doesn't improve for this many generations.

        Returns
        -------
        (best_tour, best_length)
        """
        population = self.create_population(initial_tour=initial_tour)

        # Baseline (frame 0): force the shared initial tour if given
        if initial_tour is not None:
            best_tour = list(initial_tour)
            best_length = float(calculate_tour_length(best_tour, self.dist_matrix))
        else:
            best_tour = min(population, key=lambda t: calculate_tour_length(t, self.dist_matrix))
            best_length = float(calculate_tour_length(best_tour, self.dist_matrix))

        self.history = [(best_tour.copy(), best_length)]

        no_improve = 0

        for _gen in range(self.generations):
            new_population: List[List[int]] = []

            # Elitism: keep current best
            new_population.append(best_tour.copy())

            # Fill the rest
            while len(new_population) < self.population_size:
                p1 = self.selection(population)
                p2 = self.selection(population)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

            # Evaluate current generation best
            current_best = min(population, key=lambda t: calculate_tour_length(t, self.dist_matrix))
            current_length = float(calculate_tour_length(current_best, self.dist_matrix))

            if current_length < best_length:
                best_tour = current_best
                best_length = current_length
                no_improve = 0
            else:
                no_improve += 1

            self.history.append((best_tour.copy(), best_length))

            if no_improve >= int(stagnation_patience):
                break

        return best_tour, best_length
