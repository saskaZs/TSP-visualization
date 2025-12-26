# TSP Algorithm Visualizer 🚀

A Python-based interactive tool to visualize and compare heuristic algorithms solving the **Traveling Salesperson Problem (TSP)**.

This project implements three distinct nature-inspired optimization approaches:
- **Hill Climbing**
- **Genetic Algorithm**
- **Ant Colony Optimization**

It renders their search process in real-time using `matplotlib`, demonstrating how different algorithms navigate the search space of this classic NP-hard problem.

![Screenshot](https://github.com/user-attachments/assets/8495679a-7bec-4b4f-95b3-8b1c7cce3451)  

## 🚀 Features

- **Interactive Menu**: Choose which algorithm to run or compare them all side-by-side.
- **Fair Comparison**: Uses a shared random seed system – all algorithms start from the **exact same** initial random tour.
- **Real-time Visualization**: Watch the "best-so-far" solution evolve step-by-step.
- **TSPLIB Support**: Reads standard `.tsp` files (default dataset: `berlin52`).
- **Comparison Mode**: Visualize all three algorithms running simultaneously on the same graph.

## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/tsp-visualizer.git
cd tsp-visualizer

# Install dependencies
pip install numpy matplotlib

# Run the application
python main.py
```

**Controls:** Follow the on-screen menu to select an algorithm. The visualization window opens automatically.


## 🧠 Theoretical Background

The **Traveling Salesperson Problem (TSP)** asks:  
> "Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?"

It is an NP-hard problem, meaning there is no known efficient algorithm to find the perfect solution for large numbers of cities. Therefore, we use heuristics to find "good enough" solutions in a reasonable time.

### 1. Hill Climbing (Local Search)

Hill Climbing is the simplest iterative algorithm. It works on the principle of continuous improvement.

- **Concept**: Imagine trying to climb a mountain in thick fog. You can only see one step ahead. You take a step; if it leads up, you stay there. If it leads down, you step back.

- **Implementation**:
  1. Start with a random tour.
  2. Create a "neighbor" solution by swapping two random cities (2-opt swap).
  3. Calculate the length of the new tour.
  4. Acceptance: If the new tour is shorter, keep it. If longer, discard it.
  5. Repeat until a set number of iterations or stagnation.

- **Pros/Cons**: Very fast and simple, but easily gets stuck in "local optima" (small hills) and misses the global optimum.

### 2. Genetic Algorithm (Evolutionary)

Inspired by Charles Darwin's theory of natural selection. It evolves a population of solutions over generations.

- **Concept**: "Survival of the fittest." Better solutions have a higher chance of passing their "genes" (city sub-sequences) to the next generation.

- **Implementation**:
  - **Population**: Create *N* random tours.
  - **Selection**: Use Tournament Selection to pick parents (compare random subsets, pick the best).
  - **Crossover**: Create children using Ordered Crossover (OX) to combine parts of two parents without repeating cities.
  - **Mutation**: With low probability, swap two cities in a child to introduce diversity.
  - **Elitism**: The absolute best tour is always preserved for the next generation.

- **Pros/Cons**: Good at exploring the global search space, less likely to get stuck than Hill Climbing, but computationally more expensive.

### 3. Ant Colony Optimization (Swarm Intelligence)

Inspired by how real ants find the shortest path between their nest and food using pheromone trails.

- **Concept**: Individual ants are simple, but the colony exhibits complex intelligence. Ants deposit pheromones on the ground. Shorter paths allow ants to return faster, leading to more frequent pheromone deposits, which attracts more ants.

- **Implementation**:
  - **Construction**: *N* ants build tours probabilistically. The probability of moving from City *i* to City *j* depends on:
    - Pheromone (τ): How popular is this edge?
    - Heuristic (η): How close is the city? (1 / distance)
    - Formula:  
      $$P_{ij} \propto (\tau_{ij})^\alpha \cdot (\eta_{ij})^\beta$$
  - **Evaporation**: After each iteration, pheromones on all edges decrease (evaporate) to prevent getting stuck in early bad decisions.
  - **Update**: Ants deposit new pheromones. Shorter tours deposit significantly more pheromones.

- **Pros/Cons**: Extremely effective for graph problems like TSP. It "learns" the structure of the map over time.

## 📂 Project Structure

- `main.py`: Entry point. Handles user input and initializes the visualization.
- `hill_climbing.py`: Implementation of the random-swap Hill Climbing logic.
- `genetic_algorithm.py`: Implementation of GA with tournament selection and ordered crossover.
- `ant_colony.py`: Implementation of ACO with pheromone evaporation logic.
- `visualizer.py`: Handles the matplotlib animations and drawing logic.
- `tsp_reader.py`: Utility to parse .tsp files (TSPLIB format) and calculate distance matrices.
- `berlin52.tsp`: A classic dataset containing coordinates for 52 locations in Berlin.
