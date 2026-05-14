import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from helios_as import HeliosAS, HeliosASConfig

    print("Successfully imported HeliosAS from helios_as.py.")
except ImportError:
    print("CRITICAL ERROR: HeliosAS or HeliosASConfig not found in helios_as.py.")
    print("Please ensure helios_as.py is in the same directory or Python path.")
    exit(1)


def lennard_jones_potential_vectorized(X: np.ndarray) -> np.ndarray:

    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_solutions, n_dims = X.shape
    num_atoms = n_dims // 3

    if n_dims % 3 != 0:
        return np.full(n_solutions, 1e12)

    coords = X.reshape((n_solutions, num_atoms, 3))

    diffs = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]
    dist_sq = np.sum(diffs**2, axis=-1)

    indices = np.triu_indices(num_atoms, k=1)
    r_sq = dist_sq[:, indices[0], indices[1]]

    r_sq[r_sq < 1e-9] = 1e-9

    r6 = r_sq**3
    r12 = r6**2

    potential = np.sum((1.0 / r12) - 2.0 * (1.0 / r6), axis=-1)

    return 4.0 * potential


def solve_problem(
    problem_name: str, num_atoms: int, max_evals: int, expected_min: float
):
    dimension = num_atoms * 3
    bounds = np.array([[-8.0, 8.0]] * dimension, dtype=np.float64)
    seed = 2024

    helios_params = {
        "num_explorers": 200,
        "num_refiners": 200,
        "max_iterations": int(max_evals / 50),
        "min_population_ratio": 0.1,
        "population_decay_exp": 2.0,
        "surrogate_k": 5,
        "surrogate_history_size": 4000,
        "neighborhood_radius": 0.6322,
        "explorer_memory_size": 500,
        "stagnation_threshold": 1800,
        "levy_beta": 1.9379,
        "de_factor_f": 0.3791,
        "p_levy": 0.3114,
        "p_de": 0.6860,
        "de_cr": 0.6012,
        "multimodal_cv_threshold": 1.8,
        "diversity_restart_threshold": 0.2,
    }

    population_size = helios_params["num_explorers"] + helios_params["num_refiners"]
    max_iterations = helios_params["max_iterations"]

    print("----------------------------------------")
    print(f"PROBLEM: {problem_name}")
    print(f"Number of atoms: {num_atoms}")
    print(f"Dimensions: {dimension}")
    print(f"Population size: {population_size}")
    print(f"Max iterations: {max_iterations:,}")
    print(f"Evaluation budget: {max_evals:,} evaluations")
    print(f"Expected minimum: {expected_min}")
    print("----------------------------------------")

    start_time = time.time()
    try:
        config = HeliosASConfig(**helios_params)

        optimizer = HeliosAS(
            objective_func=lennard_jones_potential_vectorized,
            bounds=bounds,
            config=config,
            seed=seed,
        )

        _, best_fitness = optimizer.optimize()
        evals_used = optimizer.evaluation_count

    except Exception as e:
        print(f"FATAL ERROR during optimization: {e}")
        best_fitness, evals_used = np.inf, -1

    end_time = time.time()
    runtime = end_time - start_time
    gap = best_fitness - expected_min

    print("\nRESULTS:")
    print(f"  Best fitness found:     {best_fitness:.6f}")
    print(f"  Function evaluations:   {evals_used:,}")
    print(f"  Runtime:                {runtime:.2f} seconds")
    print(f"  Evaluations per second: {evals_used/runtime:.0f}")
    print(f"  Gap from known minimum: {gap:.6f}")

    if abs(gap) < 0.01:
        print("  ✅ Excellent result, known minimum achieved.")
    elif abs(gap) < 1.0:
        print("  ✅ Good result, very close to the known minimum.")
    elif abs(gap) < 5.0:
        print("  ⚠️  Reasonable result, within acceptable range.")
    else:
        print("  ❌ Result did not converge to the known minimum.")
    print()

    return {
        "problem": problem_name,
        "atoms": num_atoms,
        "dimensions": dimension,
        "best_fitness": best_fitness,
        "evaluations": evals_used,
        "runtime": runtime,
        "gap": gap,
        "expected_min": expected_min,
    }


def run_comprehensive_benchmark():

    problems = [
        ("LJ7", 7, 30000, -16.505384),
        ("LJ10", 10, 40000, -28.422532),
        ("LJ13", 13, 50000, -44.326801),
        ("LJ15", 15, 60000, -52.322627),
        ("LJ19", 19, 100000, -72.659782),
    ]

    results = []
    total_start_time = time.time()

    print("=====================================================================")
    print("HELIOS-AS OPTIMIZATION FOR LENNARD-JONES (LJ) CLUSTER PROBLEMS")
    print("=====================================================================")
    print()

    for problem_name, num_atoms, max_evals, expected_min in problems:
        result = solve_problem(problem_name, num_atoms, max_evals, expected_min)
        results.append(result)

    total_runtime = time.time() - total_start_time

    print("=====================================================================")
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=====================================================================")

    successful_results = [r for r in results if abs(r["gap"]) < 5.0]
    excellent_results = [r for r in results if abs(r["gap"]) < 0.01]
    good_results = [r for r in results if abs(r["gap"]) < 1.0]

    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Problems solved: {len(results)}")
    print(f"Excellent results (gap < 0.01): {len(excellent_results)}/{len(results)}")
    print(f"Good results (gap < 1.0): {len(good_results)}/{len(results)}")
    print(f"Acceptable results (gap < 5.0): {len(successful_results)}/{len(results)}")
    print()

    print("Detailed Results:")
    print("-" * 80)
    print(
        f"{'Problem':<8} {'Atoms':<6} {'Dims':<6} {'Best Fitness':<12} {'Gap':<10} {'Evals':<8} {'Time':<6}"
    )
    print("-" * 80)

    for r in results:
        status = "✅" if abs(r["gap"]) < 1.0 else "⚠️" if abs(r["gap"]) < 5.0 else "❌"
        print(
            f"{r['problem']:<8} {r['atoms']:<6} {r['dimensions']:<6} {r['best_fitness']:<12.6f} "
            f"{r['gap']:<10.6f} {r['evaluations']:<8,} {r['runtime']:<6.1f} {status}"
        )

    print("-" * 80)

    if results:
        avg_gap = np.mean([abs(r["gap"]) for r in results])
        avg_evals = np.mean([r["evaluations"] for r in results])
        avg_time = np.mean([r["runtime"] for r in results])

        print(f"Average absolute gap: {avg_gap:.6f}")
        print(f"Average evaluations: {avg_evals:,.0f}")
        print(f"Average runtime: {avg_time:.2f} seconds")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "comprehensive":
        run_comprehensive_benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("=====================================================================")
        print("HELIOS-AS QUICK TEST FOR LENNARD-JONES CLUSTER PROBLEMS")
        print("=====================================================================")
        print()
        solve_problem("LJ7", 7, 20000, -16.505384)
    else:
        print("=====================================================================")
        print("HELIOS-AS OPTIMIZATION FOR LENNARD-JONES (LJ) CLUSTER PROBLEMS")
        print("=====================================================================")
        print()

        solve_problem("LJ13", 13, 50000, -44.326801)
        solve_problem("LJ19", 19, 100000, -72.659782)

        print(
            "\nFor comprehensive benchmark, run: python atoms_helios_as.py comprehensive"
        )
        print("For quick test, run: python atoms_helios_as.py quick")
