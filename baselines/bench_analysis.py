import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")


plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.titlesize": 20,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linewidth": 0.8,
        "lines.linewidth": 3,
        "lines.markersize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.2,
    }
)


class ElegantSingleGraphAnalyzer:
    def __init__(self, results_dir="benchmark_results"):
        self.results_dir = results_dir
        self.algorithms = ["RWO", "DifferentialEvolution", "PSO", "COBYLA", "CMAES"]
        self.functions = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank"]
        self.dimensions = [10, 30, 50, 100]

        self.algo_colors = {
            "RWO": "#FF6B6B",
            "DifferentialEvolution": "#4ECDC4",
            "PSO": "#45B7D1",
            "COBYLA": "#FFA07A",
            "CMAES": "#98D8C8",
        }

        self.function_colors = {
            "Sphere": "#FF9999",
            "Rastrigin": "#66B2FF",
            "Rosenbrock": "#99FF99",
            "Ackley": "#FFCC99",
            "Griewank": "#FF99CC",
        }

        self.dimension_colors = {
            10: "#E8F4FD",
            30: "#B8E0D2",
            50: "#D6EAF8",
            100: "#FADBD8",
        }

        self.raw_data = self.load_data()
        self.convergence_data = self.load_convergence_data()

        os.makedirs("elegant_visualizations", exist_ok=True)

    def load_data(self):
        all_data = []
        file_mapping = {
            "RWO": "rwo_raw_results.csv",
            "DifferentialEvolution": "de_raw_results.csv",
            "PSO": "pso_raw_results.csv",
            "COBYLA": "cobyla_raw_results.csv",
            "CMAES": "cmaes_raw_results.csv",
        }

        for algo, filename in file_mapping.items():
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df["algorithm"] = algo
                all_data.append(df)
                print(f"✓ Loaded {len(df)} results for {algo}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            if (
                "total_evaluations" in combined_df.columns
                and "best_fitness" in combined_df.columns
            ):
                combined_df["efficiency"] = (
                    -np.log10(combined_df["best_fitness"] + 1e-15)
                    / combined_df["total_evaluations"]
                    * 1e6
                )
                combined_df["log_fitness"] = np.log10(
                    combined_df["best_fitness"] + 1e-15
                )
                combined_df["accuracy"] = -combined_df["log_fitness"]
            print(f"Total results: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()

    def load_convergence_data(self):
        all_conv_data = []
        file_mapping = {
            "RWO": "rwo_convergence_curves.csv",
            "DifferentialEvolution": "de_convergence_curves.csv",
            "PSO": "pso_convergence_curves.csv",
            "COBYLA": "cobyla_convergence_curves.csv",
            "CMAES": "cmaes_convergence_curves.csv",
        }

        for algo, filename in file_mapping.items():
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df["algorithm"] = algo
                all_conv_data.append(df)

        return (
            pd.concat(all_conv_data, ignore_index=True)
            if all_conv_data
            else pd.DataFrame()
        )

    def plot_individual_algorithm_performance(self):

        if self.raw_data.empty:
            return

        for algo in self.algorithms:
            algo_data = self.raw_data[self.raw_data["algorithm"] == algo]
            if algo_data.empty:
                continue

            plt.figure(figsize=(12, 8))

            parts = plt.violinplot(
                [algo_data["best_fitness"].values],
                positions=[0],
                widths=0.6,
                showmeans=True,
                showmedians=True,
            )

            for pc in parts["bodies"]:
                pc.set_facecolor(self.algo_colors[algo])
                pc.set_alpha(0.7)
                pc.set_edgecolor("white")
                pc.set_linewidth(2)

            parts["cmeans"].set_color("darkred")
            parts["cmeans"].set_linewidth(3)
            parts["cmedians"].set_color("navy")
            parts["cmedians"].set_linewidth(3)

            plt.yscale("log")
            plt.ylabel("Best Fitness (log scale)", fontweight="bold", fontsize=16)
            plt.title(
                f"{algo} Performance Distribution",
                fontweight="bold",
                fontsize=18,
                pad=20,
            )
            plt.xticks([])

            mean_val = algo_data["best_fitness"].mean()
            median_val = algo_data["best_fitness"].median()
            std_val = algo_data["best_fitness"].std()

            stats_text = f"Mean: {mean_val:.2e}\nMedian: {median_val:.2e}\nStd: {std_val:.2e}\nRuns: {len(algo_data)}"
            plt.text(
                0.02,
                0.98,
                stats_text,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                f"elegant_visualizations/performance_{algo.lower()}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"✓ Created performance_{algo.lower()}.png")

    def plot_individual_function_difficulty(self):

        if self.raw_data.empty:
            return

        for func in self.functions:
            func_data = self.raw_data[self.raw_data["function"] == func]
            if func_data.empty:
                continue

            plt.figure(figsize=(12, 8))

            algo_means = (
                func_data.groupby("algorithm")["best_fitness"].mean().sort_values()
            )
            algo_stds = func_data.groupby("algorithm")["best_fitness"].std()

            bars = plt.bar(
                range(len(algo_means)),
                algo_means.values,
                color=[self.algo_colors[algo] for algo in algo_means.index],
                alpha=0.8,
                edgecolor="white",
                linewidth=2,
            )

            plt.errorbar(
                range(len(algo_means)),
                algo_means.values,
                yerr=[algo_stds[algo] for algo in algo_means.index],
                fmt="none",
                ecolor="black",
                capsize=8,
                capthick=2,
                alpha=0.7,
            )

            plt.yscale("log")
            plt.ylabel("Mean Best Fitness (log scale)", fontweight="bold", fontsize=16)
            plt.xlabel("Algorithm", fontweight="bold", fontsize=16)
            plt.title(
                f"{func} Function - Algorithm Comparison",
                fontweight="bold",
                fontsize=18,
                pad=20,
            )
            plt.xticks(
                range(len(algo_means)), algo_means.index, rotation=45, ha="right"
            )

            for i, (bar, val) in enumerate(zip(bars, algo_means.values)):
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val * 1.5,
                    f"{val:.2e}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

            plt.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(
                f"elegant_visualizations/function_difficulty_{func.lower()}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"✓ Created function_difficulty_{func.lower()}.png")

    def plot_individual_dimension_scalability(self):

        if self.raw_data.empty:
            return

        for algo in self.algorithms:
            algo_data = self.raw_data[self.raw_data["algorithm"] == algo]
            if algo_data.empty:
                continue

            plt.figure(figsize=(12, 8))

            dim_performance = algo_data.groupby("dimension")["best_fitness"].agg(
                ["mean", "std"]
            )

            plt.plot(
                dim_performance.index,
                dim_performance["mean"],
                "o-",
                color=self.algo_colors[algo],
                linewidth=4,
                markersize=12,
                markerfacecolor="white",
                markeredgecolor=self.algo_colors[algo],
                markeredgewidth=3,
            )

            plt.fill_between(
                dim_performance.index,
                dim_performance["mean"] - dim_performance["std"],
                dim_performance["mean"] + dim_performance["std"],
                alpha=0.3,
                color=self.algo_colors[algo],
            )

            plt.yscale("log")
            plt.xlabel("Dimension", fontweight="bold", fontsize=16)
            plt.ylabel("Mean Best Fitness (log scale)", fontweight="bold", fontsize=16)
            plt.title(
                f"{algo} Scalability Analysis", fontweight="bold", fontsize=18, pad=20
            )

            for dim, perf in zip(dim_performance.index, dim_performance["mean"]):
                plt.annotate(
                    f"{perf:.2e}",
                    (dim, perf),
                    textcoords="offset points",
                    xytext=(0, 15),
                    ha="center",
                    fontweight="bold",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                f"elegant_visualizations/scalability_{algo.lower()}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"✓ Created scalability_{algo.lower()}.png")

    def plot_individual_efficiency_analysis(self):

        if self.raw_data.empty or "efficiency" not in self.raw_data.columns:
            return

        for algo in self.algorithms:
            algo_data = self.raw_data[self.raw_data["algorithm"] == algo]
            if algo_data.empty:
                continue

            plt.figure(figsize=(12, 8))

            scatter = plt.scatter(
                algo_data["total_evaluations"],
                algo_data["accuracy"],
                c=algo_data["efficiency"],
                cmap="viridis",
                s=100,
                alpha=0.7,
                edgecolors="white",
                linewidth=1,
            )

            plt.colorbar(scatter, label="Efficiency Score", shrink=0.8)

            plt.xlabel("Total Function Evaluations", fontweight="bold", fontsize=16)
            plt.ylabel("Accuracy (-log₁₀(fitness))", fontweight="bold", fontsize=16)
            plt.title(
                f"{algo} Efficiency Analysis", fontweight="bold", fontsize=18, pad=20
            )

            if len(algo_data) > 1:
                z = np.polyfit(algo_data["total_evaluations"], algo_data["accuracy"], 1)
                p = np.poly1d(z)
                plt.plot(
                    algo_data["total_evaluations"],
                    p(algo_data["total_evaluations"]),
                    "--",
                    color="red",
                    linewidth=2,
                    alpha=0.8,
                    label="Trend",
                )
                plt.legend()

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                f"elegant_visualizations/efficiency_{algo.lower()}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"✓ Created efficiency_{algo.lower()}.png")

    def plot_individual_convergence_curves(self):

        if self.convergence_data.empty:
            return

        for algo in self.algorithms:
            algo_conv_data = self.convergence_data[
                self.convergence_data["algorithm"] == algo
            ]
            if algo_conv_data.empty:
                continue

            plt.figure(figsize=(12, 8))

            for func in self.functions:
                func_data = algo_conv_data[algo_conv_data["function"] == func]
                if func_data.empty:
                    continue

                convergence_median = (
                    func_data.groupby("evaluation")["fitness"].median().reset_index()
                )

                if len(convergence_median) > 10:

                    smoothed_fitness = gaussian_filter1d(
                        convergence_median["fitness"], sigma=1.5
                    )
                    plt.plot(
                        convergence_median["evaluation"],
                        smoothed_fitness,
                        label=func,
                        linewidth=3,
                        color=self.function_colors[func],
                        alpha=0.9,
                    )
                else:
                    plt.plot(
                        convergence_median["evaluation"],
                        convergence_median["fitness"],
                        label=func,
                        linewidth=3,
                        color=self.function_colors[func],
                        alpha=0.9,
                    )

            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Function Evaluations", fontweight="bold", fontsize=16)
            plt.ylabel("Best Fitness (log scale)", fontweight="bold", fontsize=16)
            plt.title(
                f"{algo} Convergence Analysis", fontweight="bold", fontsize=18, pad=20
            )
            plt.legend(
                frameon=True, fancybox=True, shadow=True, loc="best", fontsize=12
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                f"elegant_visualizations/convergence_{algo.lower()}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"✓ Created convergence_{algo.lower()}.png")

    def plot_individual_dimension_comparison(self):

        if self.raw_data.empty:
            return

        for dim in self.dimensions:
            dim_data = self.raw_data[self.raw_data["dimension"] == dim]
            if dim_data.empty:
                continue

            plt.figure(figsize=(12, 8))

            algo_data_list = []
            algo_labels = []

            for algo in self.algorithms:
                algo_dim_data = dim_data[dim_data["algorithm"] == algo]
                if len(algo_dim_data) > 0:
                    algo_data_list.append(algo_dim_data["best_fitness"].values)
                    algo_labels.append(algo)

            if algo_data_list:
                bp = plt.boxplot(
                    algo_data_list,
                    labels=algo_labels,
                    patch_artist=True,
                    boxprops=dict(alpha=0.8, linewidth=2),
                    medianprops=dict(color="darkred", linewidth=3),
                    whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2),
                )

                for patch, label in zip(bp["boxes"], algo_labels):
                    patch.set_facecolor(self.algo_colors[label])
                    patch.set_edgecolor("white")

            plt.yscale("log")
            plt.ylabel("Best Fitness (log scale)", fontweight="bold", fontsize=16)
            plt.xlabel("Algorithm", fontweight="bold", fontsize=16)
            plt.title(
                f"Algorithm Comparison - {dim}D Problems",
                fontweight="bold",
                fontsize=18,
                pad=20,
            )
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(
                f"elegant_visualizations/dimension_comparison_{dim}d.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"✓ Created dimension_comparison_{dim}d.png")

    def plot_algorithm_ranking_summary(self):

        if self.raw_data.empty:
            return

        plt.figure(figsize=(12, 8))

        overall_performance = (
            self.raw_data.groupby("algorithm")["best_fitness"].mean().sort_values()
        )

        bars = plt.barh(
            range(len(overall_performance)),
            overall_performance.values,
            color=[self.algo_colors[algo] for algo in overall_performance.index],
            alpha=0.8,
            edgecolor="white",
            linewidth=2,
        )

        plt.yticks(range(len(overall_performance)), overall_performance.index)
        plt.xscale("log")
        plt.xlabel("Mean Best Fitness (log scale)", fontweight="bold", fontsize=16)
        plt.title("Overall Algorithm Ranking", fontweight="bold", fontsize=18, pad=20)

        for i, (bar, val) in enumerate(zip(bars, overall_performance.values)):
            plt.text(
                val * 1.5,
                bar.get_y() + bar.get_height() / 2.0,
                f"#{i+1}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=14,
                bbox=dict(boxstyle="circle", facecolor="white", alpha=0.9),
            )

            plt.text(
                val * 0.1,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.2e}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=12,
                color="white",
            )

        plt.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(
            "elegant_visualizations/algorithm_ranking_summary.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Created algorithm_ranking_summary.png")

    def run_elegant_analysis(self):

        print("Starting Elegant Individual Graph Analysis")
        print("=" * 60)

        if self.raw_data.empty:
            print("No data available for analysis")
            return

        print(f"Analyzing {len(self.raw_data)} experiments")
        print(f"Algorithms: {', '.join(self.algorithms)}")
        print(f"Functions: {', '.join(self.functions)}")
        print(f"Dimensions: {self.dimensions}")
        print()

        print("Creating individual performance graphs...")
        self.plot_individual_algorithm_performance()

        print("Creating function difficulty graphs...")
        self.plot_individual_function_difficulty()

        print("Creating scalability graphs...")
        self.plot_individual_dimension_scalability()

        print("Creating efficiency graphs...")
        self.plot_individual_efficiency_analysis()

        print("Creating convergence graphs...")
        self.plot_individual_convergence_curves()

        print("Creating dimension comparison graphs...")
        self.plot_individual_dimension_comparison()

        print("Creating summary ranking...")
        self.plot_algorithm_ranking_summary()

        print("\nElegant Analysis Complete!")
        print("All individual graphs saved in 'elegant_visualizations/' directory")
        print(
            f"Generated {len(self.algorithms) * 4 + len(self.functions) + len(self.dimensions) + 1} individual graphs"
        )


def main():

    try:
        analyzer = ElegantSingleGraphAnalyzer()
        analyzer.run_elegant_analysis()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
