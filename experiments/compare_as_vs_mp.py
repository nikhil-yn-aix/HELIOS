import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

try:
    from helios_mp import HeliosMP, HeliosMPConfig
    from helios_mp import (
        sphere_function,
        rastrigin_function,
        ackley_function,
        rosenbrock_function,
        griewank_function,
        levy_function,
    )

    gpu_available = True
except ImportError as e:
    print(f"Warning: Could not import Helios-MP: {e}")
    gpu_available = False

try:
    from helios_as import HeliosAS, HeliosASConfig

    cpu_available = True
except ImportError as e:
    print(f"Warning: Could not import Helios-AS: {e}")
    cpu_available = False


def run_definitive_benchmark():
    print("=" * 80)
    print("HELIOS DEFINITIVE BENCHMARK: Helios-MP vs Helios-AS")
    print("=" * 80)

    if not gpu_available:
        print("Helios-MP (GPU) not available - cannot run comparison")
        return None

    if not cpu_available:
        print("Helios-AS (CPU) not available - cannot run comparison")
        return None

    DIMENSIONS = 30
    NUM_RUNS = 5
    MASTER_SEED = 2024

    benchmarks = {
        "Rastrigin": {
            "func": rastrigin_function,
            "bounds": [-5.12, 5.12],
            "target": 1e-6,
        },
        "Ackley": {
            "func": ackley_function,
            "bounds": [-32.768, 32.768],
            "target": 1e-10,
        },
        "Rosenbrock": {
            "func": rosenbrock_function,
            "bounds": [-30.0, 30.0],
            "target": 1e-6,
        },
        "Griewank": {
            "func": griewank_function,
            "bounds": [-600.0, 600.0],
            "target": 1e-6,
        },
        "Levy": {"func": levy_function, "bounds": [-10.0, 10.0], "target": 1e-6},
        "Sphere": {"func": sphere_function, "bounds": [-100.0, 100.0], "target": 1e-10},
    }

    mp_config = HeliosMPConfig(population_size=50000, max_iterations=200)
    as_config = HeliosASConfig(num_explorers=50, num_refiners=50, max_iterations=2500)

    all_run_data = []
    convergence_curves = {}

    print("Configuration:")
    print(f"  Problem Dimension: {DIMENSIONS}")
    print(f"  Runs per Function: {NUM_RUNS}")
    print(f"  Helios-MP Population: {mp_config.population_size:,}")
    print(f"  Helios-AS Population: {as_config.num_explorers + as_config.num_refiners}")
    print(f"  Functions: {list(benchmarks.keys())}")
    print()

    total_experiments = len(benchmarks) * NUM_RUNS * 2

    with tqdm(total=total_experiments, desc="Running Experiments", unit="run") as pbar:

        for func_name, details in benchmarks.items():
            print(f"\nBENCHMARKING: {func_name.upper()}")
            print("-" * 60)

            bounds_D = np.array([details["bounds"]] * DIMENSIONS)
            convergence_curves[func_name] = {"MP": [], "AS": []}

            mp_results = []
            as_results = []
            mp_times = []
            as_times = []
            mp_evaluations = []
            as_evaluations = []

            for run in range(NUM_RUNS):
                run_seed = MASTER_SEED + run

                pbar.set_description(f"{func_name} R{run+1} - Helios-MP")

                try:
                    import cupy as cp

                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except:
                    pass

                mp_optimizer = HeliosMP(
                    details["func"], bounds_D, mp_config, seed=run_seed
                )
                start_mp = time.time()
                _, mp_fitness = mp_optimizer.optimize()
                time_mp = time.time() - start_mp

                mp_results.append(mp_fitness)
                mp_times.append(time_mp)
                mp_evaluations.append(mp_optimizer.evaluation_count)
                convergence_curves[func_name]["MP"].append(mp_optimizer.fitness_history)

                mp_success = "SUCCESS" if mp_fitness <= details["target"] else "FAIL"
                print(
                    f"   Run {run+1} - Helios-MP: {mp_fitness:.2e} ({mp_success}) in {time_mp:.1f}s"
                )

                pbar.update(1)

                pbar.set_description(f"{func_name} R{run+1} - Helios-AS")

                as_optimizer = HeliosAS(
                    details["func"], bounds_D, as_config, seed=run_seed
                )
                start_as = time.time()
                _, as_fitness = as_optimizer.optimize()
                time_as = time.time() - start_as

                as_results.append(as_fitness)
                as_times.append(time_as)
                as_evaluations.append(as_optimizer.evaluation_count)

                if (
                    hasattr(as_optimizer, "recent_fitness_history")
                    and as_optimizer.recent_fitness_history
                ):
                    convergence_curves[func_name]["AS"].append(
                        as_optimizer.recent_fitness_history
                    )
                else:
                    convergence_curves[func_name]["AS"].append(
                        [as_fitness] * len(mp_optimizer.fitness_history)
                    )

                as_success = "SUCCESS" if as_fitness <= details["target"] else "FAIL"
                print(
                    f"   Run {run+1} - Helios-AS: {as_fitness:.2e} ({as_success}) in {time_as:.1f}s"
                )

                speedup = time_as / time_mp if time_mp > 0 else 1.0
                print(f"   Run {run+1} - Speedup: {speedup:.1f}x")

                all_run_data.append(
                    {
                        "Function": func_name,
                        "Run": run + 1,
                        "MP_Fitness": mp_fitness,
                        "AS_Fitness": as_fitness,
                        "MP_Time": time_mp,
                        "AS_Time": time_as,
                        "MP_Evals": mp_optimizer.evaluation_count,
                        "AS_Evals": as_optimizer.evaluation_count,
                        "Speedup": speedup,
                        "MP_Success": mp_fitness <= details["target"],
                        "AS_Success": as_fitness <= details["target"],
                    }
                )

                pbar.update(1)

            mp_mean = np.mean(mp_results)
            mp_std = np.std(mp_results)
            mp_best = np.min(mp_results)
            mp_success_rate = (
                np.sum([r <= details["target"] for r in mp_results]) / NUM_RUNS
            )
            mp_avg_time = np.mean(mp_times)

            as_mean = np.mean(as_results)
            as_std = np.std(as_results)
            as_best = np.min(as_results)
            as_success_rate = (
                np.sum([r <= details["target"] for r in as_results]) / NUM_RUNS
            )
            as_avg_time = np.mean(as_times)

            avg_speedup = np.mean(
                [t_as / t_mp for t_as, t_mp in zip(as_times, mp_times)]
            )

            print(f"\n   SUMMARY for {func_name}:")
            print(
                f"      Helios-MP: Best={mp_best:.2e}, Success={mp_success_rate:.0%}, Time={mp_avg_time:.1f}s"
            )
            print(
                f"      Helios-AS: Best={as_best:.2e}, Success={as_success_rate:.0%}, Time={as_avg_time:.1f}s"
            )
            print(f"      Average Speedup: {avg_speedup:.1f}x")

    df_runs = pd.DataFrame(all_run_data)
    df_runs["Quality_Ratio"] = df_runs["AS_Fitness"] / df_runs["MP_Fitness"]

    summary = (
        df_runs.groupby("Function")
        .agg(
            MP_Best_Fitness=("MP_Fitness", "min"),
            AS_Best_Fitness=("AS_Fitness", "min"),
            MP_Mean_Fitness=("MP_Fitness", "mean"),
            AS_Mean_Fitness=("AS_Fitness", "mean"),
            Mean_Speedup=("Speedup", "mean"),
            Mean_Quality_Ratio=("Quality_Ratio", "mean"),
            MP_Success_Rate=("MP_Success", "mean"),
            AS_Success_Rate=("AS_Success", "mean"),
        )
        .round(4)
    )

    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print(summary.to_string())

    summary.to_csv("helios_final_summary.csv")
    df_runs.to_csv("helios_final_detailed_runs.csv", index=False)
    print(
        "\nResults saved to helios_final_summary.csv and helios_final_detailed_runs.csv"
    )

    overall_speedups = df_runs.groupby("Function")["Speedup"].mean()
    overall_speedup = overall_speedups.mean()

    mp_success_rates = df_runs.groupby("Function")["MP_Success"].mean()
    as_success_rates = df_runs.groupby("Function")["AS_Success"].mean()

    overall_mp_success = mp_success_rates.mean()
    overall_as_success = as_success_rates.mean()

    print("\nOVERALL PERFORMANCE SUMMARY:")
    print(f"   Average Speedup: {overall_speedup:.1f}x")
    print(f"   Helios-MP Success Rate: {overall_mp_success:.1%}")
    print(f"   Helios-AS Success Rate: {overall_as_success:.1%}")

    mp_wins = 0
    as_wins = 0

    for func_name in benchmarks.keys():
        func_data = df_runs[df_runs["Function"] == func_name]
        mp_best = func_data["MP_Fitness"].min()
        as_best = func_data["AS_Fitness"].min()

        if mp_best < as_best:
            mp_wins += 1
        else:
            as_wins += 1

    print("\nSOLUTION QUALITY COMPARISON:")
    print(
        f"   Functions where Helios-MP found better solution: {mp_wins}/{len(benchmarks)}"
    )
    print(
        f"   Functions where Helios-AS found better solution: {as_wins}/{len(benchmarks)}"
    )

    print("\nGenerating convergence plots...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    plt.style.use("default")
    sns.set_palette("husl")

    for i, (func_name, data) in enumerate(convergence_curves.items()):
        if i >= len(axes):
            break

        ax = axes[i]

        mp_curves = data["MP"]
        as_curves = data["AS"]

        max_mp_len = max(len(curve) for curve in mp_curves) if mp_curves else 0
        max_as_len = max(len(curve) for curve in as_curves) if as_curves else 0
        max_len = max(max_mp_len, max_as_len)

        if max_len == 0:
            continue

        mp_padded = []
        for curve in mp_curves:
            if len(curve) < max_len:
                padded = curve + [curve[-1]] * (max_len - len(curve))
            else:
                padded = curve[:max_len]
            mp_padded.append(padded)

        as_padded = []
        for curve in as_curves:
            if len(curve) < max_len:
                padded = curve + [curve[-1]] * (max_len - len(curve))
            else:
                padded = curve[:max_len]
            as_padded.append(padded)

        if mp_padded:
            for run_data in mp_padded:
                ax.plot(run_data, color="red", alpha=0.3, linewidth=0.8)

        if as_padded:
            for run_data in as_padded:
                ax.plot(run_data, color="blue", alpha=0.3, linewidth=0.8)

        if mp_padded:
            mean_mp = np.mean(mp_padded, axis=0)
            ax.plot(
                mean_mp, color="red", linewidth=3, label="Helios-MP (Mean)", zorder=5
            )

        if as_padded:
            mean_as = np.mean(as_padded, axis=0)
            ax.plot(
                mean_as, color="blue", linewidth=3, label="Helios-AS (Mean)", zorder=5
            )

        ax.set_title(f"Convergence on {func_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Iterations", fontsize=12)
        ax.set_ylabel("Best Fitness (log scale)", fontsize=12)
        ax.set_yscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.set_facecolor("#f8f9fa")

    plt.tight_layout()
    plt.savefig("helios_final_convergence_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Convergence plots saved to helios_final_convergence_plots.png")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    speedup_data = (
        df_runs.groupby("Function")["Speedup"].mean().sort_values(ascending=True)
    )
    bars1 = ax1.barh(
        range(len(speedup_data)),
        speedup_data.values,
        color="skyblue",
        edgecolor="navy",
        alpha=0.8,
    )
    ax1.set_yticks(range(len(speedup_data)))
    ax1.set_yticklabels(speedup_data.index)
    ax1.set_xlabel("Speedup Factor", fontsize=12)
    ax1.set_title("Average Speedup by Function", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}x",
            ha="left",
            va="center",
            fontweight="bold",
        )

    mp_success = df_runs.groupby("Function")["MP_Success"].mean() * 100
    as_success = df_runs.groupby("Function")["AS_Success"].mean() * 100

    x = np.arange(len(mp_success))
    width = 0.35

    bars2 = ax2.bar(
        x - width / 2,
        mp_success.values,
        width,
        label="Helios-MP",
        color="red",
        alpha=0.7,
    )
    bars3 = ax2.bar(
        x + width / 2,
        as_success.values,
        width,
        label="Helios-AS",
        color="blue",
        alpha=0.7,
    )

    ax2.set_xlabel("Functions", fontsize=12)
    ax2.set_ylabel("Success Rate (%)", fontsize=12)
    ax2.set_title("Success Rate Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mp_success.index, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    quality_ratios = df_runs.groupby("Function")["Quality_Ratio"].mean()
    colors = ["green" if ratio < 1 else "orange" for ratio in quality_ratios.values]
    bars4 = ax3.bar(
        quality_ratios.index,
        quality_ratios.values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    ax3.axhline(y=1, color="red", linestyle="--", linewidth=2, label="Equal Quality")
    ax3.set_xlabel("Functions", fontsize=12)
    ax3.set_ylabel("Quality Ratio (AS/MP)", fontsize=12)
    ax3.set_title("Solution Quality Ratio", fontsize=14, fontweight="bold")
    ax3.set_xticklabels(quality_ratios.index, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    eval_ratio_data = []
    for func_name in benchmarks.keys():
        func_data = df_runs[df_runs["Function"] == func_name]
        mp_evals = func_data["MP_Evals"].mean()
        as_evals = func_data["AS_Evals"].mean()
        eval_ratio_data.append(mp_evals / as_evals)

    bars5 = ax4.bar(
        list(benchmarks.keys()),
        eval_ratio_data,
        color="purple",
        alpha=0.7,
        edgecolor="black",
    )
    ax4.set_xlabel("Functions", fontsize=12)
    ax4.set_ylabel("Evaluation Ratio (MP/AS)", fontsize=12)
    ax4.set_title("Evaluation Efficiency", fontsize=14, fontweight="bold")
    ax4.set_xticklabels(list(benchmarks.keys()), rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("helios_final_performance_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Performance analysis plots saved to helios_final_performance_analysis.png")

    print(f"\n{'='*80}")
    print("DEFINITIVE BENCHMARK COMPLETE")
    print(f"{'='*80}")

    return summary, df_runs


if __name__ == "__main__":
    try:
        import cupy as cp

        print(f"GPU Device: {cp.cuda.Device()}")
        print(f"GPU Memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB\n")

        results_summary, results_detailed = run_definitive_benchmark()

        if results_summary is not None:
            print(
                "\nComparison complete! Check the CSV files and PNG plots for detailed results."
            )
            print("Use these results for your HiPC paper experimental section.")
        else:
            print("\nComparison failed - check algorithm imports and dependencies.")

    except ImportError:
        print("CuPy not available - GPU benchmark cannot run")
        print("Install CuPy: pip install cupy-cuda11x")
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure GPU has sufficient memory for large populations")
