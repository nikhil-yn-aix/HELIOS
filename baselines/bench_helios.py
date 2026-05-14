import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor

try:
    from helios_as import HeliosAS, HeliosASConfig

    HELIOS_AVAILABLE = True
    print("Successfully imported HeliosAS from helios_as.py.")
except ImportError:
    print("CRITICAL ERROR: HeliosAS or HeliosASConfig not found in helios_as.py.")
    print("Please ensure helios_as.py is in the same directory or Python path.")
    HELIOS_AVAILABLE = False
    exit(1)


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=-1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] < 2:
        return np.inf if x.ndim > 1 else np.array([np.inf])
    return np.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (x[..., :-1] - 1) ** 2, axis=-1
    )


def ackley(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    if n == 0:
        return 0.0
    sum_sq = np.sum(x**2, axis=-1)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x), axis=-1)
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e
    )


def griewank(x: np.ndarray) -> np.ndarray:
    sum_sq = np.sum(x**2, axis=-1) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[-1] + 1))), axis=-1)
    return 1.0 + sum_sq - prod_cos


FUNCTIONS = {
    "Sphere": {"func": sphere, "bounds": [-100, 100], "tolerance": 1e-6},
    "Rastrigin": {"func": rastrigin, "bounds": [-5.12, 5.12], "tolerance": 1e-2},
    "Rosenbrock": {"func": rosenbrock, "bounds": [-30, 30], "tolerance": 1e-2},
    "Ackley": {"func": ackley, "bounds": [-32.768, 32.768], "tolerance": 1e-2},
    "Griewank": {"func": griewank, "bounds": [-600, 600], "tolerance": 1e-2},
}

DIMENSIONS = [10, 30, 50, 100]
NUM_RUNS = 10
MAX_EVALUATIONS_BUDGET = 1000000

HELIOS_CONFIG_PARAMS = {
    "num_explorers": 80,
    "num_refiners": 80,
    "max_iterations": 5000,
    "min_population_ratio": 0.25,
    "population_decay_exp": 2.0,
    "surrogate_k": 5,
    "surrogate_history_size": 2000,
    "neighborhood_radius": 0.6322,
    "explorer_memory_size": 40,
    "stagnation_threshold": 2000,
    "levy_beta": 1.5379,
    "de_factor_f": 0.3791,
    "p_levy": 0.3114,
    "p_de": 0.6860,
    "de_cr": 0.5012,
    "multimodal_cv_threshold": 0.7,
    "diversity_restart_threshold": 0.04,
}


class EvaluationObserver:
    def __init__(self, func):
        self.func = func
        self.best_fitness = np.inf
        self.convergence_curve = []
        self.eval_count = 0

    def __call__(self, x: np.ndarray):
        if x.ndim == 1:
            num_evals = 1
        else:
            num_evals = x.shape[0]

        self.eval_count += num_evals
        fitness = self.func(x)

        current_best = np.min(fitness)
        if current_best < self.best_fitness:
            self.best_fitness = current_best

        if self.eval_count % 100 == 0:
            self.convergence_curve.append((self.eval_count, self.best_fitness))

        return fitness


def run_helios_experiment(func_name, func_config, dimension, run_id, seed):
    func = func_config["func"]
    bounds = func_config["bounds"]
    tolerance = func_config["tolerance"]

    bounds_array = np.array([bounds] * dimension, dtype=np.float64)
    observer = EvaluationObserver(func)

    start_time = time.time()
    best_fitness = np.inf
    total_evaluations = 0
    success = False

    config = HeliosASConfig(**HELIOS_CONFIG_PARAMS)

    optimizer = HeliosAS(
        objective_func=observer, bounds=bounds_array, config=config, seed=seed
    )

    best_position, best_fitness = optimizer.optimize()
    total_evaluations = optimizer.evaluation_count
    success = best_fitness <= tolerance if np.isfinite(best_fitness) else False

    execution_time = time.time() - start_time

    convergence_iteration = MAX_EVALUATIONS_BUDGET
    for evals, fitness in observer.convergence_curve:
        if fitness <= tolerance:
            convergence_iteration = evals
            break

    return {
        "algorithm": "HeliosAS",
        "function": func_name,
        "dimension": dimension,
        "run_id": run_id,
        "seed": seed,
        "best_fitness": best_fitness,
        "total_evaluations": total_evaluations,
        "execution_time": execution_time,
        "success": success,
        "convergence_iteration": convergence_iteration,
        "convergence_curve": observer.convergence_curve,
    }


def run_helios_experiment_wrapper(args):
    func_name, func_config, dimension, run_id, seed = args
    result = run_helios_experiment(func_name, func_config, dimension, run_id, seed)

    status = "Success" if result["success"] else "Fail"
    print(
        f"Completed: {func_name:>10} {dimension}D Run {run_id+1:2d} | "
        f"Fitness: {result['best_fitness']:.4e} | "
        f"Evals: {result['total_evaluations']:6d} | "
        f"Time: {result['execution_time']:.2f}s | Status: {status}"
    )

    return result


def main():
    if not HELIOS_AVAILABLE:
        return

    print("\nParallelized HeliosAS Benchmark")
    print("=" * 60)
    total_pop = (
        HELIOS_CONFIG_PARAMS["num_explorers"] + HELIOS_CONFIG_PARAMS["num_refiners"]
    )
    print(
        f"Population: {HELIOS_CONFIG_PARAMS['num_explorers']} explorers + {HELIOS_CONFIG_PARAMS['num_refiners']} refiners = {total_pop} total"
    )
    print(f"Max Iterations: {HELIOS_CONFIG_PARAMS['max_iterations']:,}")
    print(
        f"Stagnation Threshold: {HELIOS_CONFIG_PARAMS['stagnation_threshold']} iterations"
    )
    print(f"Conceptual Evaluation Budget: {MAX_EVALUATIONS_BUDGET:,}")

    max_workers = min(8, os.cpu_count() or 1)
    print(f"Parallel workers: {max_workers}")

    start_time = time.time()

    experiments = []
    for func_name, func_config in FUNCTIONS.items():
        for dimension in DIMENSIONS:
            for run_id in range(NUM_RUNS):
                seed = (dimension * 100) + run_id + 42
                experiments.append((func_name, func_config, dimension, run_id, seed))

    print(f"\nRunning {len(experiments)} experiments in parallel...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_helios_experiment_wrapper, experiments))

    total_time = time.time() - start_time
    print(f"\nCompleted {len(results)} experiments in {total_time:.1f} seconds.")

    os.makedirs("benchmark_results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results/helios_as_raw_results.csv", index=False)

    convergence_data = []
    for result in results:
        for evals, fitness in result["convergence_curve"]:
            convergence_data.append(
                {
                    "algorithm": result["algorithm"],
                    "function": result["function"],
                    "dimension": result["dimension"],
                    "run_id": result["run_id"],
                    "evaluation": evals,
                    "fitness": fitness,
                }
            )

    if convergence_data:
        conv_df = pd.DataFrame(convergence_data)
        conv_df.to_csv(
            "benchmark_results/helios_as_convergence_curves.csv", index=False
        )

    summary_stats = (
        df.groupby(["function", "dimension"])
        .agg(
            {
                "best_fitness": ["mean", "std", "median", "min", "max"],
                "execution_time": ["mean", "std"],
                "total_evaluations": ["mean", "std"],
                "success": ["sum", "count"],
            }
        )
        .round(6)
    )

    summary_stats.columns = ["_".join(col).strip() for col in summary_stats.columns]
    summary_stats["success_rate"] = (
        summary_stats["success_sum"] / summary_stats["success_count"]
    )
    summary_stats.to_csv("benchmark_results/helios_as_summary_statistics.csv")

    print("\nHeliosAS Results Summary:")
    print("-" * 60)
    for func_name in FUNCTIONS.keys():
        for dim in DIMENSIONS:
            subset = df[(df["function"] == func_name) & (df["dimension"] == dim)]
            if not subset.empty:
                mean_fitness = subset["best_fitness"].mean()
                success_rate = subset["success"].mean()
                mean_evals = subset["total_evaluations"].mean()
                early_stops = (
                    subset["total_evaluations"]
                    < HELIOS_CONFIG_PARAMS["max_iterations"] * total_pop * 0.9
                ).sum()
                print(
                    f"{func_name:>12} {dim}D: Mean Fitness: {mean_fitness:.2e} | "
                    f"Success: {success_rate:.1%} | Avg Evals: {mean_evals:,.0f} | "
                    f"Early Stops: {early_stops}/{NUM_RUNS}"
                )

    print("\nResults saved to benchmark_results/")
    print("- helios_as_raw_results.csv")
    print("- helios_as_convergence_curves.csv")
    print("- helios_as_summary_statistics.csv")
    print(f"\nTotal execution time: {total_time:.1f} seconds")


if __name__ == "__main__":
    main()
