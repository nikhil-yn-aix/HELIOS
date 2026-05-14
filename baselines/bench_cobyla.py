import numpy as np
import pandas as pd
import time
import os
import warnings
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal


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
MAX_TIME_SECONDS = 300


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Experiment timed out")


class EvaluationTracker:
    def __init__(self, func, max_evals, max_time):
        self.func = func
        self.max_evals = max_evals
        self.max_time = max_time
        self.eval_count = 0
        self.best_fitness = np.inf
        self.convergence_curve = []
        self.start_time = time.time()
        self.stopped = False

    def __call__(self, x):

        if self.stopped:
            return np.inf

        if self.eval_count >= self.max_evals:
            self.stopped = True
            return np.inf

        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_time:
            self.stopped = True
            return np.inf

        self.eval_count += 1

        try:
            fitness = self.func(x)
        except:
            fitness = np.inf

        if fitness < self.best_fitness:
            self.best_fitness = fitness

        if self.eval_count % 1000 == 0:
            self.convergence_curve.append(self.best_fitness)

        return fitness


def run_cobyla_experiment(func_name, func_config, dimension, run_id, seed):

    np.random.seed(seed)

    func = func_config["func"]
    bounds = func_config["bounds"]
    tolerance = func_config["tolerance"]

    tracker = EvaluationTracker(func, MAX_EVALUATIONS, MAX_TIME_SECONDS)

    lower, upper = bounds
    x0 = np.random.uniform(lower, upper, dimension)

    constraints = []
    for i in range(dimension):
        constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i] - lower})
        constraints.append({"type": "ineq", "fun": lambda x, i=i: upper - x[i]})

    start_time = time.time()

    try:

        max_iter = 1000000

        result = minimize(
            tracker,
            x0,
            method="COBYLA",
            constraints=constraints,
            options={
                "maxiter": max_iter,
                "rhobeg": min(1.0, (upper - lower) / 10),
                "rhoend": max(1e-8, tolerance / 100),
                "disp": False,
                "catol": tolerance * 10,
            },
        )

        best_fitness = result.fun if hasattr(result, "fun") else tracker.best_fitness
        best_position = result.x if hasattr(result, "x") else None
        success = best_fitness <= tolerance

        if tracker.best_fitness < best_fitness:
            best_fitness = tracker.best_fitness

    except TimeoutException:
        print(f"TIMEOUT: {func_name} {dimension}D run {run_id+1}")
        best_fitness = tracker.best_fitness if tracker.best_fitness < np.inf else np.inf
        best_position = None
        success = best_fitness <= tolerance

    except Exception as e:
        print(f"ERROR: {func_name} {dimension}D run {run_id+1}: {str(e)[:100]}")
        best_fitness = tracker.best_fitness if tracker.best_fitness < np.inf else np.inf
        best_position = None
        success = best_fitness <= tolerance

    execution_time = time.time() - start_time

    convergence_iteration = len(tracker.convergence_curve) * 1000
    for i, fitness in enumerate(tracker.convergence_curve):
        if fitness <= tolerance:
            convergence_iteration = (i + 1) * 1000
            break

    return {
        "algorithm": "COBYLA",
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
        "stopped_early": tracker.stopped,
    }


def run_cobyla_experiment_wrapper(args):

    func_name, func_config, dimension, run_id, seed = args

    try:

        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(MAX_TIME_SECONDS + 30)

        result = run_cobyla_experiment(func_name, func_config, dimension, run_id, seed)

        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

        status = "TIMEOUT" if result.get("stopped_early", False) else "OK"
        print(
            f"Completed [{status}]: {func_name} {dimension}D run {run_id+1}: {result['best_fitness']:.2e} "
            f"({result['total_evaluations']} evals, {result['execution_time']:.1f}s)"
        )

        return result

    except TimeoutException:
        return {
            "algorithm": "COBYLA",
            "function": func_name,
            "dimension": dimension,
            "run_id": run_id,
            "seed": seed,
            "best_fitness": np.inf,
            "total_evaluations": 0,
            "execution_time": MAX_TIME_SECONDS,
            "success": False,
            "convergence_iteration": 0,
            "convergence_curve": [],
            "stopped_early": True,
        }
    except Exception as e:
        print(f"FATAL ERROR: {func_name} {dimension}D run {run_id+1}: {e}")
        return {
            "algorithm": "COBYLA",
            "function": func_name,
            "dimension": dimension,
            "run_id": run_id,
            "seed": seed,
            "best_fitness": np.inf,
            "total_evaluations": 0,
            "execution_time": 0,
            "success": False,
            "convergence_iteration": 0,
            "convergence_curve": [],
            "stopped_early": True,
        }


def main():
    print("Fixed Parallelized COBYLA Benchmark")
    print("=" * 40)
    print("Method: Constrained Optimization BY Linear Approximations")
    print(f"Max evaluations per run: {MAX_EVALUATIONS:,}")
    print(f"Max time per run: {MAX_TIME_SECONDS} seconds")
    print("Parallel workers: 8 cores")

    start_time = time.time()

    experiments = []
    for func_name, func_config in FUNCTIONS.items():
        for dimension in DIMENSIONS:
            for run_id in range(NUM_RUNS):
                seed = run_id + 42
                experiments.append((func_name, func_config, dimension, run_id, seed))

    print(f"\nRunning {len(experiments)} experiments in parallel...")

    results = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=8) as executor:

        future_to_exp = {
            executor.submit(run_cobyla_experiment_wrapper, exp): exp
            for exp in experiments
        }

        for future in as_completed(future_to_exp):
            try:
                result = future.result(timeout=30)
                results.append(result)
                completed_count += 1

                if completed_count % 10 == 0:
                    print(f"Progress: {completed_count}/{len(experiments)} completed")

            except Exception as e:
                exp = future_to_exp[future]
                func_name, _, dimension, run_id, seed = exp
                print(f"FAILED: {func_name} {dimension}D run {run_id+1}: {e}")

                results.append(
                    {
                        "algorithm": "COBYLA",
                        "function": func_name,
                        "dimension": dimension,
                        "run_id": run_id,
                        "seed": seed,
                        "best_fitness": np.inf,
                        "total_evaluations": 0,
                        "execution_time": 0,
                        "success": False,
                        "convergence_iteration": 0,
                        "convergence_curve": [],
                        "stopped_early": True,
                    }
                )

    total_time = time.time() - start_time
    print(f"\nCompleted {len(results)} experiments in {total_time:.1f} seconds")

    os.makedirs("benchmark_results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results/cobyla_fixed_raw_results.csv", index=False)

    convergence_data = []
    for result in results:
        for i, fitness in enumerate(result.get("convergence_curve", [])):
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
        conv_df.to_csv(
            "benchmark_results/cobyla_fixed_convergence_curves.csv", index=False
        )

    summary_stats = (
        df.groupby(["function", "dimension"])
        .agg(
            {
                "best_fitness": ["mean", "std", "median", "min", "max"],
                "execution_time": ["mean", "std"],
                "total_evaluations": ["mean", "std"],
                "success": ["sum", "count"],
                "stopped_early": ["sum"],
            }
        )
        .round(6)
    )

    summary_stats.columns = ["_".join(col).strip() for col in summary_stats.columns]
    summary_stats["success_rate"] = (
        summary_stats["success_sum"] / summary_stats["success_count"]
    )
    summary_stats["timeout_rate"] = (
        summary_stats["stopped_early_sum"] / summary_stats["success_count"]
    )
    summary_stats.to_csv("benchmark_results/cobyla_fixed_summary_statistics.csv")

    print("\nFixed COBYLA Results Summary:")
    print("-" * 40)
    for func_name in FUNCTIONS.keys():
        for dim in DIMENSIONS:
            subset = df[(df["function"] == func_name) & (df["dimension"] == dim)]
            mean_fitness = (
                subset["best_fitness"].replace([np.inf, -np.inf], np.nan).mean()
            )
            success_rate = subset["success"].mean()
            timeout_rate = subset["stopped_early"].mean()
            mean_evals = subset["total_evaluations"].mean()
            print(
                f"{func_name:>12} {dim}D: {mean_fitness:.2e} "
                f"(success: {success_rate:.1%}, timeout: {timeout_rate:.1%}, evals: {mean_evals:.0f})"
            )

    print("\nResults saved to benchmark_results/")
    print("- cobyla_fixed_raw_results.csv")
    print("- cobyla_fixed_convergence_curves.csv")
    print("- cobyla_fixed_summary_statistics.csv")
    print(f"\nTotal execution time: {total_time:.1f} seconds")


if __name__ == "__main__":
    main()
