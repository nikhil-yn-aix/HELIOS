import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings("ignore")


def set_professional_style():

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "figure.titlesize": 22,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 1.2,
        }
    )


class ProfessionalBenchmarkAnalyzer:
    def __init__(self, results_dir="results/benchmark/data"):
        self.results_dir = results_dir

        self.algorithms = [
            "HeliosAS",
            "CMAES",
            "DifferentialEvolution",
            "PSO",
            "COBYLA",
        ]
        self.algorithm_display_names = {
            "HeliosAS": "Helios-AS",
            "CMAES": "CMA-ES",
            "DifferentialEvolution": "Differential Evolution",
            "PSO": "Particle Swarm Optimization",
            "COBYLA": "COBYLA",
        }
        self.functions = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank"]
        self.dimensions = [10, 30, 50, 100]

        self.algo_colors = {
            "HeliosAS": "#E31A1C",
            "CMAES": "#1F78B4",
            "DifferentialEvolution": "#33A02C",
            "PSO": "#FF7F00",
            "COBYLA": "#6A3D9A",
        }

        self.raw_data = self._load_data()
        self.convergence_data = self._load_convergence_data()

        os.makedirs("results/benchmark/figures", exist_ok=True)

    def _load_data(self):

        file_mapping = {
            "HeliosAS": "helios_raw_results.csv",
            "CMAES": "cmaes_raw_results.csv",
            "DifferentialEvolution": "de_raw_results.csv",
            "PSO": "pso_raw_results.csv",
            "COBYLA": "cobyla_raw_results.csv",
        }

        all_data = []
        for algo, filename in file_mapping.items():
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df["algorithm"] = algo
                df["display_name"] = self.algorithm_display_names[algo]
                all_data.append(df)
                print(f"✓ Loaded {len(df)} results for {algo}")

        if not all_data:
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)

        combined_df["log_fitness"] = np.log10(combined_df["best_fitness"] + 1e-16)
        combined_df["accuracy"] = -combined_df["log_fitness"]

        if "total_evaluations" in combined_df.columns:
            combined_df["efficiency"] = (
                combined_df["accuracy"] / combined_df["total_evaluations"] * 1e6
            )

        print(f"Total experiments loaded: {len(combined_df)}")
        return combined_df

    def _load_convergence_data(self):

        file_mapping = {
            "HeliosAS": "helios_convergence_curves.csv",
            "CMAES": "cmaes_convergence_curves.csv",
            "DifferentialEvolution": "de_convergence_curves.csv",
            "PSO": "pso_convergence_curves.csv",
            "COBYLA": "cobyla_convergence_curves.csv",
        }

        all_conv_data = []
        for algo, filename in file_mapping.items():
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df["algorithm"] = algo
                df["display_name"] = self.algorithm_display_names[algo]
                all_conv_data.append(df)

        return (
            pd.concat(all_conv_data, ignore_index=True)
            if all_conv_data
            else pd.DataFrame()
        )

    def _add_significance_stars(self, ax, data, x_col, y_col, pairs, height_offset=0.1):

        max_y = data[y_col].max()

        for i, (alg1, alg2) in enumerate(pairs):
            group1 = data[data[x_col] == alg1][y_col].dropna()
            group2 = data[data[x_col] == alg2][y_col].dropna()

            if len(group1) > 0 and len(group2) > 0:
                try:
                    statistic, p_value = mannwhitneyu(
                        group1, group2, alternative="two-sided"
                    )

                    if p_value < 0.001:
                        stars = "***"
                    elif p_value < 0.01:
                        stars = "**"
                    elif p_value < 0.05:
                        stars = "*"
                    else:
                        stars = "ns"

                    x1_pos = list(data[x_col].unique()).index(alg1)
                    x2_pos = list(data[x_col].unique()).index(alg2)
                    y_pos = max_y * (1 + height_offset + i * 0.1)

                    ax.plot([x1_pos, x2_pos], [y_pos, y_pos], "k-", linewidth=1)
                    ax.plot(
                        [x1_pos, x1_pos],
                        [y_pos - max_y * 0.02, y_pos],
                        "k-",
                        linewidth=1,
                    )
                    ax.plot(
                        [x2_pos, x2_pos],
                        [y_pos - max_y * 0.02, y_pos],
                        "k-",
                        linewidth=1,
                    )
                    ax.text(
                        (x1_pos + x2_pos) / 2,
                        y_pos + max_y * 0.02,
                        stars,
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=12,
                    )
                except:
                    continue

    def create_figure_1_overall_performance(self):

        if self.raw_data.empty:
            return

        print("Creating Figure 1: Overall Performance Comparison...")

        fig, ax = plt.subplots(figsize=(14, 10))

        algo_order = (
            self.raw_data.groupby("display_name")["log_fitness"]
            .median()
            .sort_values()
            .index
        )

        parts = ax.violinplot(
            [
                self.raw_data[self.raw_data["display_name"] == algo][
                    "log_fitness"
                ].values
                for algo in algo_order
            ],
            positions=range(len(algo_order)),
            widths=0.7,
            showmeans=False,
            showmedians=False,
        )

        for i, (pc, algo) in enumerate(zip(parts["bodies"], algo_order)):
            original_algo = [
                k for k, v in self.algorithm_display_names.items() if v == algo
            ][0]
            pc.set_facecolor(self.algo_colors[original_algo])
            pc.set_alpha(0.6)
            pc.set_edgecolor("white")
            pc.set_linewidth(2)

        bp = ax.boxplot(
            [
                self.raw_data[self.raw_data["display_name"] == algo][
                    "log_fitness"
                ].values
                for algo in algo_order
            ],
            positions=range(len(algo_order)),
            widths=0.3,
            patch_artist=True,
            boxprops=dict(alpha=0.8, linewidth=2),
            medianprops=dict(color="darkred", linewidth=3),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
        )

        for patch, algo in zip(bp["boxes"], algo_order):
            original_algo = [
                k for k, v in self.algorithm_display_names.items() if v == algo
            ][0]
            patch.set_facecolor(self.algo_colors[original_algo])
            patch.set_alpha(0.9)
            patch.set_edgecolor("white")

        helios_pairs = [
            ("Helios-AS", algo) for algo in algo_order if algo != "Helios-AS"
        ]
        self._add_significance_stars(
            ax, self.raw_data, "display_name", "log_fitness", helios_pairs
        )

        ax.set_title(
            "Overall Algorithm Performance Comparison\n(All Problems & Dimensions)",
            fontweight="bold",
            pad=30,
        )
        ax.set_xlabel("Algorithm", fontweight="bold", labelpad=15)
        ax.set_ylabel("Best Fitness (log10 scale)", fontweight="bold", labelpad=15)
        ax.set_xticks(range(len(algo_order)))
        ax.set_xticklabels(algo_order, rotation=15, ha="right")

        for i, algo in enumerate(algo_order):
            n = len(self.raw_data[self.raw_data["display_name"] == algo])
            ax.text(
                i,
                ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                f"n={n}",
                ha="center",
                va="top",
                fontsize=10,
                alpha=0.7,
            )

        plt.tight_layout()
        plt.savefig(
            "results/benchmark/figures/figure_1_overall_performance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Figure 1 saved to results/benchmark/figures/")

    def create_figure_2_scalability_analysis(self):
        if self.raw_data.empty:
            return

        print("Creating Figure 2: Scalability Analysis...")

        fig, ax = plt.subplots(figsize=(14, 10))

        agg_data = (
            self.raw_data.groupby(["algorithm", "dimension"])["best_fitness"]
            .agg(["mean", "median"])
            .reset_index()
        )

        agg_data["log_mean_fitness"] = np.log10(agg_data["mean"] + 1e-16)
        agg_data["log_median_fitness"] = np.log10(agg_data["median"] + 1e-16)

        for algo in self.algorithms:
            algo_data = agg_data[agg_data["algorithm"] == algo]
            if algo_data.empty:
                continue

            color = self.algo_colors.get(algo, "gray")

            ax.plot(
                algo_data["dimension"],
                algo_data["log_mean_fitness"],
                "o-",
                color=color,
                linewidth=3,
                markersize=10,
                label=f"{algo} (Mean)",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2.5,
            )

            ax.plot(
                algo_data["dimension"],
                algo_data["log_median_fitness"],
                "x--",
                color=color,
                linewidth=2,
                markersize=8,
                label=f"{algo} (Median)",
                alpha=0.7,
            )

        ax.set_title(
            "Algorithm Scalability with Problem Dimension", fontweight="bold", pad=20
        )
        ax.set_xlabel("Problem Dimension", fontweight="bold", labelpad=15)
        ax.set_ylabel("Best Fitness (log10 scale)", fontweight="bold", labelpad=15)
        ax.set_xticks(self.dimensions)

        handles, labels = ax.get_legend_handles_labels()

        from collections import OrderedDict

        new_handles = OrderedDict(zip(labels, handles))
        ax.legend(
            new_handles.values(),
            new_handles.keys(),
            title="Algorithm",
            frameon=True,
            fancybox=True,
            shadow=True,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(
            "results/benchmark/figures/figure_2_scalability_analysis.png", dpi=300
        )
        plt.close()
        print("✓ Figure 2 saved to results/benchmark/figures/")

    def create_figure_3_performance_by_function(self):

        if self.raw_data.empty:
            return

        print("Creating Figure 3: Performance by Function...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, func in enumerate(self.functions):
            ax = axes[i]
            func_data = self.raw_data[self.raw_data["function"] == func]

            if func_data.empty:
                ax.set_visible(False)
                continue

            algo_order = (
                func_data.groupby("display_name")["log_fitness"]
                .median()
                .sort_values()
                .index
            )

            bp = ax.boxplot(
                [
                    func_data[func_data["display_name"] == algo]["log_fitness"].values
                    for algo in algo_order
                ],
                labels=algo_order,
                patch_artist=True,
                boxprops=dict(alpha=0.8, linewidth=1.5),
                medianprops=dict(color="darkred", linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
            )

            for patch, algo in zip(bp["boxes"], algo_order):
                original_algo = [
                    k for k, v in self.algorithm_display_names.items() if v == algo
                ][0]
                patch.set_facecolor(self.algo_colors[original_algo])
                patch.set_alpha(0.7)

            ax.set_title(f"{func} Function", fontweight="bold", fontsize=16)
            ax.set_ylabel("Best Fitness (log10)", fontweight="bold")
            ax.tick_params(axis="x", rotation=45, labelsize=10)
            ax.grid(True, alpha=0.3)

        axes[-1].set_visible(False)

        fig.suptitle(
            "Algorithm Performance by Benchmark Function",
            fontweight="bold",
            fontsize=20,
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(
            "results/benchmark/figures/figure_3_performance_by_function.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Figure 3 saved to results/benchmark/figures/")

    def create_figure_4_convergence_analysis(self):

        if self.convergence_data.empty:
            return

        print("Creating Figure 4: Convergence Analysis...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, func in enumerate(self.functions):
            ax = axes[i]
            func_conv_data = self.convergence_data[
                self.convergence_data["function"] == func
            ]

            if func_conv_data.empty:
                ax.set_visible(False)
                continue

            for algo in self.algorithms:
                algo_func_data = func_conv_data[func_conv_data["algorithm"] == algo]
                if algo_func_data.empty:
                    continue

                convergence_stats = (
                    algo_func_data.groupby("evaluation")["fitness"]
                    .agg(["median", "std"])
                    .reset_index()
                )

                if len(convergence_stats) < 5:
                    continue

                display_name = self.algorithm_display_names[algo]
                color = self.algo_colors[algo]

                if len(convergence_stats) > 10:
                    smoothed_median = gaussian_filter1d(
                        convergence_stats["median"], sigma=1.0
                    )
                    ax.plot(
                        convergence_stats["evaluation"],
                        smoothed_median,
                        label=display_name,
                        color=color,
                        linewidth=2.5,
                    )
                else:
                    ax.plot(
                        convergence_stats["evaluation"],
                        convergence_stats["median"],
                        label=display_name,
                        color=color,
                        linewidth=2.5,
                    )

            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"{func} Function", fontweight="bold")
            ax.set_xlabel("Function Evaluations")
            ax.set_ylabel("Best Fitness (log scale)")
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.legend(fontsize=10)

        axes[-1].set_visible(False)

        fig.suptitle(
            "Convergence Behavior Analysis", fontweight="bold", fontsize=20, y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(
            "results/benchmark/figures/figure_4_convergence_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Figure 4 saved to results/benchmark/figures/")

    def create_summary_statistics_table(self):

        if self.raw_data.empty:
            return

        print("Creating Summary Statistics Table...")

        stats_summary = []

        for algo in self.algorithms:
            algo_data = self.raw_data[self.raw_data["algorithm"] == algo]
            if algo_data.empty:
                continue

            display_name = self.algorithm_display_names[algo]

            stats = {
                "Algorithm": display_name,
                "Total Runs": len(algo_data),
                "Mean Fitness": f"{algo_data['best_fitness'].mean():.2e}",
                "Median Fitness": f"{algo_data['best_fitness'].median():.2e}",
                "Std Fitness": f"{algo_data['best_fitness'].std():.2e}",
                "Success Rate (< 1e-10)": f"{(algo_data['best_fitness'] < 1e-10).mean():.1%}",
                "Best Result": f"{algo_data['best_fitness'].min():.2e}",
                "Worst Result": f"{algo_data['best_fitness'].max():.2e}",
            }

            if "total_evaluations" in algo_data.columns:
                stats["Mean Evaluations"] = (
                    f"{algo_data['total_evaluations'].mean():.0f}"
                )
                stats["Efficiency Score"] = (
                    f"{algo_data['efficiency'].mean():.2f}"
                    if "efficiency" in algo_data.columns
                    else "N/A"
                )

            stats_summary.append(stats)

        stats_df = pd.DataFrame(stats_summary)
        stats_df.to_csv("results/benchmark/figures/summary_statistics.csv", index=False)
        print(
            "✓ Summary statistics saved to results/benchmark/figures/summary_statistics.csv"
        )

        print("KEY INSIGHTS:")
        best_overall = (
            self.raw_data.groupby("algorithm")["best_fitness"].mean().idxmin()
        )
        print(f"Best Overall Algorithm: {self.algorithm_display_names[best_overall]}")

        helios_results = self.raw_data[self.raw_data["algorithm"] == "HeliosAS"][
            "best_fitness"
        ]
        for algo in ["CMAES", "DifferentialEvolution", "PSO", "COBYLA"]:
            other_results = self.raw_data[self.raw_data["algorithm"] == algo][
                "best_fitness"
            ]
            if len(other_results) > 0:
                try:
                    statistic, p_value = mannwhitneyu(
                        helios_results, other_results, alternative="two-sided"
                    )
                    significance = (
                        "***"
                        if p_value < 0.001
                        else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    )
                    print(
                        f"Helios-AS vs {self.algorithm_display_names[algo]}: p={p_value:.4f} {significance}"
                    )
                except:
                    continue

    def run_professional_analysis(self):
        print("=" * 70)

        if self.raw_data.empty:
            print(
                "❌ No data found in benchmark_results/. Please run benchmarks first."
            )
            return

        print(f"Analyzing {len(self.raw_data)} experimental results")
        print(
            f"Algorithms: {', '.join([self.algorithm_display_names[a] for a in self.algorithms])}"
        )
        print(f"Functions: {', '.join(self.functions)}")
        print(f"Dimensions: {self.dimensions}")
        print()

        self.create_figure_1_overall_performance()
        self.create_figure_2_scalability_analysis()
        self.create_figure_3_performance_by_function()
        self.create_figure_4_convergence_analysis()
        self.create_summary_statistics_table()

        print("Professional Analysis Complete!")
        print("All figures saved in 'results/benchmark/figures/' directory")
        print("Generated 4 publication-ready figures + statistical summary")
        print("Ready for submission to conferences like HiPC!")


def main():

    set_professional_style()

    try:
        analyzer = ProfessionalBenchmarkAnalyzer()
        analyzer.run_professional_analysis()
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
