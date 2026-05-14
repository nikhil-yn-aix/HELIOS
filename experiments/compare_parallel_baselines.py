import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cupy as cp
import numpy as np
import time
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Tuple, List, Optional
from tqdm import tqdm


try:
    from helios_mp import HeliosMP, HeliosMPConfig
except ImportError:
    print("FATAL: helios_mp.py not found. Please place it in the same directory.")
    exit()


@dataclass
class ParallelPSOConfig:

    population_size: int
    max_iterations: int
    inertia_w: float = 0.729
    cognitive_c1: float = 1.49445
    social_c2: float = 1.49445


class ParallelPSO:
    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        config: ParallelPSOConfig,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            cp.random.seed(seed)

        self.objective_func = objective_func
        self.config = config

        self.bounds = cp.asarray(bounds, dtype=cp.float32)
        self.pop_size = self.config.population_size
        self.dimensions = self.bounds.shape[0]

        self.positions = cp.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.pop_size, self.dimensions),
            dtype=cp.float32,
        )
        self.velocities = cp.zeros_like(self.positions)
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = cp.full(self.pop_size, cp.inf, dtype=cp.float32)

        self.gbest_position = cp.zeros(self.dimensions, dtype=cp.float32)
        self.gbest_fitness = cp.inf

        self.evaluation_count = 0
        self.fitness_history: List[float] = []

    def optimize(self) -> Tuple[np.ndarray, float]:
        progress_bar = tqdm(
            range(self.config.max_iterations),
            desc="  Parallel PSO ",
            leave=False,
            ncols=100,
        )

        for _ in progress_bar:
            fitness = self.objective_func(self.positions)
            self.evaluation_count += self.pop_size

            update_mask = fitness < self.pbest_fitness
            self.pbest_positions = cp.where(
                update_mask[:, None], self.positions, self.pbest_positions
            )
            self.pbest_fitness = cp.where(update_mask, fitness, self.pbest_fitness)

            current_best_idx = cp.argmin(self.pbest_fitness)
            self.gbest_fitness = self.pbest_fitness[current_best_idx]
            self.gbest_position = self.pbest_positions[current_best_idx]

            self.fitness_history.append(float(self.gbest_fitness))

            r1, r2 = cp.random.rand(2, self.pop_size, self.dimensions).astype(
                cp.float32
            )
            cognitive_comp = (
                self.config.cognitive_c1 * r1 * (self.pbest_positions - self.positions)
            )
            social_comp = (
                self.config.social_c2 * r2 * (self.gbest_position - self.positions)
            )
            self.velocities = (
                self.config.inertia_w * self.velocities + cognitive_comp + social_comp
            )
            self.positions += self.velocities
            self.positions = cp.clip(
                self.positions, self.bounds[:, 0], self.bounds[:, 1]
            )

            progress_bar.set_postfix(
                {"Best Fitness": f"{float(self.gbest_fitness):.4e}"}
            )

        return cp.asnumpy(self.gbest_position), float(self.gbest_fitness)


@dataclass
class ParallelDEConfig:

    population_size: int
    max_iterations: int
    mutation_factor: float = 0.5
    crossover_rate: float = 0.9


class ParallelDE:
    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        config: ParallelDEConfig,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            cp.random.seed(seed)

        self.objective_func = objective_func
        self.config = config

        self.bounds = cp.asarray(bounds, dtype=cp.float32)
        self.pop_size = self.config.population_size
        self.dimensions = self.bounds.shape[0]

        self.population = cp.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.pop_size, self.dimensions),
            dtype=cp.float32,
        )
        self.fitness = cp.full(self.pop_size, cp.inf, dtype=cp.float32)

        self.evaluation_count = 0
        self.fitness_history: List[float] = []

    def optimize(self) -> Tuple[np.ndarray, float]:
        self.fitness = self.objective_func(self.population)
        self.evaluation_count += self.pop_size

        progress_bar = tqdm(
            range(self.config.max_iterations),
            desc="  Parallel DE  ",
            leave=False,
            ncols=100,
        )

        for _ in progress_bar:

            indices = cp.arange(self.pop_size)
            p1 = cp.random.permutation(self.pop_size)
            p2 = cp.random.permutation(self.pop_size)
            p3 = cp.random.permutation(self.pop_size)

            p1 = cp.where(p1 == indices, (p1 + 1) % self.pop_size, p1)
            p2 = cp.where(p2 == indices, (p2 + 1) % self.pop_size, p2)
            p3 = cp.where(p3 == indices, (p3 + 1) % self.pop_size, p3)

            mutant_pop = self.population[p1] + self.config.mutation_factor * (
                self.population[p2] - self.population[p3]
            )
            mutant_pop = cp.clip(mutant_pop, self.bounds[:, 0], self.bounds[:, 1])

            crossover_mask = (
                cp.random.rand(self.pop_size, self.dimensions)
                < self.config.crossover_rate
            )
            j_rand = cp.random.randint(0, self.dimensions, size=self.pop_size)
            crossover_mask[cp.arange(self.pop_size), j_rand] = True

            trial_pop = cp.where(crossover_mask, mutant_pop, self.population)

            trial_fitness = self.objective_func(trial_pop)
            self.evaluation_count += self.pop_size

            improvement_mask = trial_fitness < self.fitness
            self.population = cp.where(
                improvement_mask[:, None], trial_pop, self.population
            )
            self.fitness = cp.where(improvement_mask, trial_fitness, self.fitness)

            best_fitness = cp.min(self.fitness)
            self.fitness_history.append(float(best_fitness))
            progress_bar.set_postfix({"Best Fitness": f"{float(best_fitness):.4e}"})

        best_idx = cp.argmin(self.fitness)
        return cp.asnumpy(self.population[best_idx]), float(self.fitness[best_idx])


def sphere_function(x):
    return cp.sum(x**2, axis=1)


def rastrigin_function(x):
    return 10 * x.shape[1] + cp.sum(x**2 - 10 * cp.cos(2 * cp.pi * x), axis=1)


def rosenbrock_function(x):
    return cp.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2, axis=1
    )


def ackley_function(x):
    n = x.shape[1]
    sum_sq = cp.sum(x**2, axis=1)
    sum_cos = cp.sum(cp.cos(2.0 * cp.pi * x), axis=1)
    return (
        -20.0 * cp.exp(-0.2 * cp.sqrt(sum_sq / n)) - cp.exp(sum_cos / n) + 20.0 + cp.e
    )


def griewank_function(x):
    sum_sq = cp.sum(x**2, axis=1) / 4000.0
    prod_cos = cp.prod(cp.cos(x / cp.sqrt(cp.arange(1, x.shape[1] + 1))), axis=1)
    return sum_sq - prod_cos + 1


def levy_function(x):
    w = 1 + (x - 1) / 4
    term1 = cp.sin(cp.pi * w[:, 0]) ** 2
    term3 = (w[:, -1] - 1) ** 2 * (1 + cp.sin(2 * cp.pi * w[:, -1]) ** 2)
    term2 = cp.sum(
        (w[:, :-1] - 1) ** 2 * (1 + 10 * cp.sin(cp.pi * w[:, :-1] + 1) ** 2), axis=1
    )
    return term1 + term2 + term3


def run_parallel_benchmark():
    print("=" * 80)
    print("HELIOS-MP vs. PARALLEL BASELINES COMPARISON")
    print("=" * 80)

    DIMENSIONS = 30
    NUM_RUNS = 5
    MASTER_SEED = 2024
    POPULATION_SIZE = 50000
    MAX_ITERATIONS = 200

    benchmarks = {
        "Sphere": {"func": sphere_function, "bounds": [-100.0, 100.0]},
        "Rastrigin": {"func": rastrigin_function, "bounds": [-5.12, 5.12]},
        "Rosenbrock": {"func": rosenbrock_function, "bounds": [-30.0, 30.0]},
        "Ackley": {"func": ackley_function, "bounds": [-32.768, 32.768]},
        "Griewank": {"func": griewank_function, "bounds": [-600.0, 600.0]},
        "Levy": {"func": levy_function, "bounds": [-10.0, 10.0]},
    }

    algorithms = {
        "Helios-MP": (
            HeliosMP,
            HeliosMPConfig(
                population_size=POPULATION_SIZE, max_iterations=MAX_ITERATIONS
            ),
        ),
        "Parallel-PSO": (
            ParallelPSO,
            ParallelPSOConfig(
                population_size=POPULATION_SIZE, max_iterations=MAX_ITERATIONS
            ),
        ),
        "Parallel-DE": (
            ParallelDE,
            ParallelDEConfig(
                population_size=POPULATION_SIZE, max_iterations=MAX_ITERATIONS
            ),
        ),
    }

    results_list = []

    for func_name, details in benchmarks.items():
        print(f"\n===== BENCHMARKING ON: {func_name.upper()} =====")
        bounds_D = np.array([details["bounds"]] * DIMENSIONS)

        for algo_name, (algo_class, config) in algorithms.items():
            for run in range(NUM_RUNS):
                print(f"\n--- Running {algo_name}, Run {run+1}/{NUM_RUNS} ---")
                run_seed = MASTER_SEED + run

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

                optimizer = algo_class(details["func"], bounds_D, config, seed=run_seed)

                start_time = time.time()
                position, fitness = optimizer.optimize()
                final_fitness = float(
                    fitness.item() if hasattr(fitness, "item") else fitness
                )
                end_time = time.time()

                total_time = end_time - start_time

                results_list.append(
                    {
                        "Function": func_name,
                        "Algorithm": algo_name,
                        "Run": run + 1,
                        "FinalFitness": final_fitness,
                        "ExecutionTime": total_time,
                        "Evaluations": optimizer.evaluation_count,
                        "ConvergenceHistory": optimizer.fitness_history,
                    }
                )

                print(
                    f"--- Finished {algo_name} R{run+1}: Fitness={final_fitness:.4e}, Time={total_time:.2f}s ---"
                )

    df_results = pd.DataFrame(results_list)
    output_filename = "parallel_baselines_comparison_results.csv"
    df_results.to_csv(output_filename, index=False)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. ALL RESULTS SAVED TO: {output_filename}")
    print("=" * 80)

    summary = (
        df_results.groupby(["Function", "Algorithm"])
        .agg(
            Median_Fitness=("FinalFitness", "median"),
            Best_Fitness=("FinalFitness", "min"),
            Mean_Time=("ExecutionTime", "mean"),
        )
        .round(4)
    )

    print("\n--- RESULTS SUMMARY (Median Fitness over 5 runs) ---")
    print(summary.to_string())


if __name__ == "__main__":
    run_parallel_benchmark()
