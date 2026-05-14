import numpy as np
import pandas as pd
import time
from scipy.optimize import differential_evolution
import os
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


def sphere(x):
    return np.sum(x**2)


def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    if len(x) < 2:
        return np.inf
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def ackley(x):
    n = len(x)
    if n == 0:
        return 0.0
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e
    )


def griewank(x):
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_sq - prod_cos


FUNCTIONS = {
    "Sphere": {"func": sphere, "bounds": [-100, 100], "tolerance": 1e-6},
    "Rastrigin": {"func": rastrigin, "bounds": [-5.12, 5.12], "tolerance": 1e-2},
    "Rosenbrock": {"func": rosenbrock, "bounds": [-30, 30], "tolerance": 1e-2},
    "Ackley": {"func": ackley, "bounds": [-32.768, 32.768], "tolerance": 1e-2},
    "Griewank": {"func": griewank, "bounds": [-600, 600], "tolerance": 1e-2},
}

DIMENSIONS = [10, 30, 50, 100]
NUM_RUNS = 10
MAX_EVALUATIONS = 1000000


class EvaluationTracker:
    def __init__(self, func, max_evals):
        self.func = func
        self.max_evals = max_evals
        self.eval_count = 0
        self.best_fitness = np.inf
        self.convergence_curve = []

    def __call__(self, x):
        if self.eval_count >= self.max_evals:
            return np.inf

        self.eval_count += 1
        fitness = self.func(x)

        if fitness < self.best_fitness:
            self.best_fitness = fitness

        if self.eval_count % 1000 == 0:
            self.convergence_curve.append(self.best_fitness)

        return fitness


def run_de_experiment(func_name, func_config, dimension, run_id, seed):

    np.random.seed(seed)

    func = func_config["func"]
    bounds = func_config["bounds"]
    tolerance = func_config["tolerance"]

    bounds_list = [bounds] * dimension

    tracker = EvaluationTracker(func, MAX_EVALUATIONS)

    start_time = time.time()

    try:

        result = differential_evolution(
            tracker,
            bounds_list,
            seed=seed,
            maxiter=MAX_EVALUATIONS // (15 * dimension),
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            atol=1e-12,
            disp=False,
        )

        best_fitness = result.fun
        best_position = result.x
        success = best_fitness <= tolerance

    except Exception as e:
        print(f"Error in DE {func_name}D{dimension} run {run_id}: {e}")
        best_fitness = np.inf
        best_position = None
        success = False

    execution_time = time.time() - start_time

    convergence_iteration = len(tracker.convergence_curve) * 1000
    for i, fitness in enumerate(tracker.convergence_curve):
        if fitness <= tolerance:
            convergence_iteration = (i + 1) * 1000
            break

    return {
        "algorithm": "DifferentialEvolution",
        "function": func_name,
        "dimension": dimension,
        "run_id": run_id,
        "seed": seed,
        "best_fitness": best_fitness,
        "total_evaluations": tracker.eval_count,
        "execution_time": execution_time,
        "success": success,
        "convergence_iteration": convergence_iteration,
        "convergence_curve": tracker.convergence_curve,
    }


def run_de_experiment_wrapper(args):

    func_name, func_config, dimension, run_id, seed = args

    result = run_de_experiment(func_name, func_config, dimension, run_id, seed)

    print(
        f"Completed: {func_name} {dimension}D run {run_id+1}: {result['best_fitness']:.2e} "
        f"({result['total_evaluations']} evals, {result['execution_time']:.1f}s)"
    )

    return result


def main():
    print("Parallelized Differential Evolution Benchmark")
    print("=" * 50)
    print(f"Max evaluations per run: {MAX_EVALUATIONS:,}")
    print("Population size: 15 × dimension")
    print("Parallel workers: 4 cores")

    start_time = time.time()

    experiments = []
    for func_name, func_config in FUNCTIONS.items():
        for dimension in DIMENSIONS:
            for run_id in range(NUM_RUNS):
                seed = run_id + 42
                experiments.append((func_name, func_config, dimension, run_id, seed))

    print(f"\nRunning {len(experiments)} experiments in parallel...")

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_de_experiment_wrapper, experiments))

    total_time = time.time() - start_time
    print(f"\nCompleted {len(results)} experiments in {total_time:.1f} seconds")

    os.makedirs("benchmark_results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results/de_raw_results.csv", index=False)

    convergence_data = []
    for result in results:
        for i, fitness in enumerate(result["convergence_curve"]):
            convergence_data.append(
                {
                    "algorithm": result["algorithm"],
                    "function": result["function"],
                    "dimension": result["dimension"],
                    "run_id": result["run_id"],
                    "evaluation": (i + 1) * 1000,
                    "fitness": fitness,
                }
            )

    if convergence_data:
        conv_df = pd.DataFrame(convergence_data)
        conv_df.to_csv("benchmark_results/de_convergence_curves.csv", index=False)

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
    summary_stats.to_csv("benchmark_results/de_summary_statistics.csv")

    print("\nDifferential Evolution Results Summary:")
    print("-" * 50)
    for func_name in FUNCTIONS.keys():
        for dim in DIMENSIONS:
            subset = df[(df["function"] == func_name) & (df["dimension"] == dim)]
            mean_fitness = subset["best_fitness"].mean()
            success_rate = subset["success"].mean()
            mean_evals = subset["total_evaluations"].mean()
            print(
                f"{func_name:>12} {dim}D: {mean_fitness:.2e} "
                f"(success: {success_rate:.1%}, evals: {mean_evals:.0f})"
            )

    print("\nResults saved to benchmark_results/")
    print("- de_raw_results.csv")
    print("- de_convergence_curves.csv")
    print("- de_summary_statistics.csv")
    print(f"\nTotal execution time: {total_time:.1f} seconds")


if __name__ == "__main__":
    main()
