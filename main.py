"""
TSP Solver - Main Module
Entry point for solving TSP with different algorithms and modern visualizations.

This version runs in an interactive menu loop

Guarantees
----------
- All algorithms start from the SAME initial random tour (shared baseline).
- The dashed route in the animation is always that shared initial tour.
- Animations do NOT restart
- Final tours are printed for each algorithm.

"""

from __future__ import annotations

import random
from typing import List

from tsp_reader import read_tsp_file, calculate_distance_matrix
from hill_climbing import HillClimbing
from genetic_algorithm import GeneticAlgorithm
from ant_colony import AntColonyOptimization
from visualizer import TSPVisualizer


def _format_tour(tour: List[int]) -> str:
    if not tour:
        return ""
    return " -> ".join(map(str, tour + [tour[0]]))


def _make_initial_tour(n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def main() -> None:
    
    tsp_filename = "berlin52.tsp"  
    user_seed = None               
    fullscreen = True              
    show_labels = False           

    print("=" * 70)
    print("TSP SOLVER - ALGORITHM VISUALIZER")
    print("=" * 70)

    # Read TSP file once
    print(f"\n[*] Reading: {tsp_filename}")
    try:
        coords = read_tsp_file(tsp_filename)
        print(f"[*] Loaded {len(coords)} cities")
        
    except FileNotFoundError:
        print(f"[!] Error: TSP file '{tsp_filename}' not found. Put it next to main.py.")
        return

    # Distance matrix once
    print("[*] Calculating distance matrix...")
    dist_matrix = calculate_distance_matrix(coords)
    print(f"[*] Distance matrix computed ({len(coords)}x{len(coords)})")

    visualizer = TSPVisualizer(coords)
    
    # Shared initial tour is kept for the whole session by default.
    # (So if you run algorithms one-by-one, they remain comparable.)
    seed = user_seed if user_seed is not None else random.randrange(1_000_000_000)
    initial_tour = _make_initial_tour(len(coords), seed)

    def print_initial_info() -> None:
        initial_len = float(
            sum(
                dist_matrix[initial_tour[i], initial_tour[(i + 1) % len(initial_tour)]]
                for i in range(len(initial_tour))
            )
        )
        print(f"\n[*] Shared initial tour seed: {seed}")
        print(f"[*] Shared initial distance: {initial_len:.2f}")

    print_initial_info()

    while True:
        # Menu loop so closing the animation window does NOT end the program
        print("\nSelect algorithm to visualize:")
        print("1. Hill Climbing (Basic)")
        print("2. Genetic Algorithm")
        print("3. Ant Colony Optimization")
        print("4. Run all algorithms (single comparison animation)")
        print("5. New initial random tour (new seed)")
        print("\n0. Exit")

        choice = input("\nEnter your choice (0-5): ").strip()
        if choice == "0":
            print("Exiting...")
            return

        if choice == "5":
            seed = random.randrange(1_000_000_000)
            initial_tour = _make_initial_tour(len(coords), seed)
            print_initial_info()
            continue

        try:
            if choice == "1":
                print("\n[*] Running Hill Climbing...")
                hc = HillClimbing(dist_matrix, max_iterations=1500, seed=seed + 1)
                hc_tour, hc_len = hc.solve(initial_tour=initial_tour, stagnation_patience=180)
                print(f"[*] Hill Climbing final distance: {hc_len:.2f}")
                print(f"[*] Hill Climbing final tour: {_format_tour(hc_tour)}")

                print("\n[*] Creating animation...")
                _anim = visualizer.animate_single_algorithm(
                    hc.history,
                    "Hill Climbing",
                    color="#FF6B6B",
                    interval=35,
                    initial_tour=initial_tour,
                    show_city_labels=show_labels,
                    fullscreen=fullscreen,
                )

            elif choice == "2":
                print("\n[*] Running Genetic Algorithm...")
                ga = GeneticAlgorithm(
                    dist_matrix,
                    population_size=120,
                    generations=700,
                    mutation_rate=0.03,
                    seed=seed + 2,
                )
                ga_tour, ga_len = ga.solve(initial_tour=initial_tour, stagnation_patience=120)
                print(f"[*] Genetic Algorithm final distance: {ga_len:.2f}")
                print(f"[*] Genetic Algorithm final tour: {_format_tour(ga_tour)}")

                print("\n[*] Creating animation...")
                _anim = visualizer.animate_single_algorithm(
                    ga.history,
                    "Genetic Algorithm",
                    color="#4ECDC4",
                    interval=35,
                    initial_tour=initial_tour,
                    show_city_labels=show_labels,
                    fullscreen=fullscreen,
                )

            elif choice == "3":
                print("\n[*] Running Ant Colony Optimization...")
                aco = AntColonyOptimization(
                    dist_matrix,
                    n_ants=35,
                    n_iterations=450,
                    alpha=1.0,
                    beta=2.5,
                    evaporation_rate=0.45,
                    Q=120.0,
                    seed=seed + 3,
                )
                aco_tour, aco_len = aco.solve(initial_tour=initial_tour, stagnation_patience=90)
                print(f"[*] Ant Colony final distance: {aco_len:.2f}")
                print(f"[*] Ant Colony final tour: {_format_tour(aco_tour)}")

                print("\n[*] Creating animation...")
                _anim = visualizer.animate_single_algorithm(
                    aco.history,
                    "Ant Colony Optimization",
                    color="#7C5CFC",
                    interval=35,
                    initial_tour=initial_tour,
                    show_city_labels=show_labels,
                    fullscreen=fullscreen,
                )

            elif choice == "4":
                print("\n[*] Running all algorithms (same initial tour)...")

                print("\n  [1/3] Hill Climbing...")
                hc = HillClimbing(dist_matrix, max_iterations=1500, seed=seed + 1)
                hc_tour, hc_len = hc.solve(initial_tour=initial_tour, stagnation_patience=180)
                print(f"    - Distance: {hc_len:.2f}  | steps: {len(hc.history) - 1}")

                print("\n  [2/3] Genetic Algorithm...")
                ga = GeneticAlgorithm(dist_matrix, population_size=120, generations=700, mutation_rate=0.03, seed=seed + 2)
                ga_tour, ga_len = ga.solve(initial_tour=initial_tour, stagnation_patience=120)
                print(f"    - Distance: {ga_len:.2f}  | steps: {len(ga.history) - 1}")

                print("\n  [3/3] Ant Colony Optimization...")
                aco = AntColonyOptimization(dist_matrix, n_ants=35, n_iterations=450, alpha=1.0, beta=2.5, evaporation_rate=0.45, Q=120.0, seed=seed + 3)
                aco_tour, aco_len = aco.solve(initial_tour=initial_tour, stagnation_patience=90)
                print(f"    - Distance: {aco_len:.2f}  | steps: {len(aco.history) - 1}")

                print("\n" + "=" * 70)
                print("FINAL TOURS")
                print("=" * 70)
                print(f"Hill Climbing:     {hc_len:.2f}\n{_format_tour(hc_tour)}\n")
                print(f"Genetic Algorithm: {ga_len:.2f}\n{_format_tour(ga_tour)}\n")
                print(f"Ant Colony:        {aco_len:.2f}\n{_format_tour(aco_tour)}\n")

                best_name, best_len = min(
                    [("Hill Climbing", hc_len), ("Genetic Algorithm", ga_len), ("Ant Colony", aco_len)],
                    key=lambda x: x[1],
                )
                print(f"Best algorithm: {best_name} ({best_len:.2f})")
                print("=" * 70)

                print("\n[*] Creating ONE comparison animation...")
                _anim = visualizer.animate_comparison(
                    histories={
                        "Hill Climbing": hc.history,
                        "Genetic Algorithm": ga.history,
                        "Ant Colony": aco.history,
                    },
                    colors={
                        "Hill Climbing": "#FF6B6B",
                        "Genetic Algorithm": "#4ECDC4",
                        "Ant Colony": "#7C5CFC",
                    },
                    interval=35,
                    initial_tour=initial_tour,
                    show_city_labels=show_labels,
                    fullscreen=fullscreen,
                )

            else:
                print("Invalid choice!")
                continue

        except KeyboardInterrupt:
            # If user interrupts from terminal, don't crash: return to menu.
            print("\n[!] Interrupted. Returning to menu...")
            continue

        # After the window is closed, we return here and continue the menu loop.
        print("\n[+] Animation closed. Returning to menu...")


if __name__ == "__main__":
    main()