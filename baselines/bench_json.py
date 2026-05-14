import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class BenchmarkAnalyzer:
    def __init__(self, results_dir="benchmark_results"):
        self.results_dir = results_dir
        self.algorithms = ["RWO", "DifferentialEvolution", "PSO", "COBYLA", "CMAES"]
        self.functions = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank"]
        self.dimensions = [10, 30, 50, 100]

        self.raw_data = self.load_all_data()
        self.convergence_data = self.load_convergence_data()

        os.makedirs("plots", exist_ok=True)

    def load_all_data(self):

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
                print(f"Loaded {len(df)} results for {algo}")
            else:
                print(f"Warning: {filepath} not found")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Total combined results: {len(combined_df)}")
            return combined_df
        else:
            raise FileNotFoundError("No benchmark result files found")

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

        if all_conv_data:
            return pd.concat(all_conv_data, ignore_index=True)
        else:
            print("Warning: No convergence data found")
            return pd.DataFrame()

    def create_performance_rankings(self):

        rankings = {}

        overall_ranking = (
            self.raw_data.groupby("algorithm")["best_fitness"]
            .agg(["mean", "median", "std", "min", "max"])
            .round(6)
        )
        overall_ranking["rank_by_mean"] = overall_ranking["mean"].rank()

        function_rankings = {}
        for func in self.functions:
            func_data = self.raw_data[self.raw_data["function"] == func]
            func_ranking = (
                func_data.groupby("algorithm")["best_fitness"]
                .agg(["mean", "median", "std"])
                .round(6)
            )
            func_ranking["rank"] = func_ranking["mean"].rank()
            function_rankings[func] = func_ranking.to_dict("index")

        dimension_rankings = {}
        for dim in self.dimensions:
            dim_data = self.raw_data[self.raw_data["dimension"] == dim]
            dim_ranking = (
                dim_data.groupby("algorithm")["best_fitness"]
                .agg(["mean", "median", "std"])
                .round(6)
            )
            dim_ranking["rank"] = dim_ranking["mean"].rank()
            dimension_rankings[f"{dim}D"] = dim_ranking.to_dict("index")

        success_rates = (
            self.raw_data.groupby("algorithm")["success"]
            .agg(["mean", "sum", "count"])
            .round(3)
        )
        success_rates["success_percentage"] = success_rates["mean"] * 100

        efficiency = (
            self.raw_data.groupby("algorithm")["total_evaluations"]
            .agg(["mean", "std", "min", "max"])
            .round(0)
        )

        timing = (
            self.raw_data.groupby("algorithm")["execution_time"]
            .agg(["mean", "std", "min", "max"])
            .round(2)
        )

        rankings = {
            "overall_performance": overall_ranking.to_dict("index"),
            "function_wise": function_rankings,
            "dimension_wise": dimension_rankings,
            "success_rates": success_rates.to_dict("index"),
            "efficiency": efficiency.to_dict("index"),
            "execution_time": timing.to_dict("index"),
            "metadata": {
                "total_experiments": len(self.raw_data),
                "algorithms_tested": list(self.raw_data["algorithm"].unique()),
                "functions_tested": list(self.raw_data["function"].unique()),
                "dimensions_tested": list(self.raw_data["dimension"].unique()),
                "analysis_date": datetime.now().isoformat(),
            },
        }

        return rankings

    def plot_performance_comparison(self):

        plt.figure(figsize=(12, 8))

        plot_data = self.raw_data[np.isfinite(self.raw_data["best_fitness"])]

        plt.subplot(2, 2, 1)
        sns.boxplot(data=plot_data, x="algorithm", y="best_fitness")
        plt.yscale("log")
        plt.title("Overall Performance Distribution (Log Scale)")
        plt.xticks(rotation=45)
        plt.ylabel("Best Fitness (log scale)")

        plt.subplot(2, 2, 2)
        success_by_algo = self.raw_data.groupby("algorithm")["success"].mean() * 100
        bars = plt.bar(success_by_algo.index, success_by_algo.values)
        plt.title("Success Rate by Algorithm")
        plt.ylabel("Success Rate (%)")
        plt.xticks(rotation=45)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.subplot(2, 2, 3)
        heatmap_data = (
            self.raw_data.groupby(["algorithm", "function"])["best_fitness"]
            .mean()
            .unstack()
        )

        heatmap_data_log = np.log10(heatmap_data + 1e-15)

        sns.heatmap(heatmap_data_log, annot=True, fmt=".1f", cmap="viridis_r")
        plt.title("Performance Heatmap (log10 scale)")
        plt.ylabel("Algorithm")

        plt.subplot(2, 2, 4)
        efficiency_data = self.raw_data.groupby("algorithm")["total_evaluations"].mean()
        bars = plt.bar(efficiency_data.index, efficiency_data.values)
        plt.title("Average Evaluations Used")
        plt.ylabel("Evaluations")
        plt.xticks(rotation=45)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5000,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig("plots/performance_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_convergence_curves(self):

        if self.convergence_data.empty:
            print("No convergence data available for plotting")
            return

        for func in self.functions:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            for i, dim in enumerate(self.dimensions):
                ax = axes[i]

                func_dim_data = self.convergence_data[
                    (self.convergence_data["function"] == func)
                    & (self.convergence_data["dimension"] == dim)
                ]

                if func_dim_data.empty:
                    continue

                for algo in self.algorithms:
                    algo_data = func_dim_data[func_dim_data["algorithm"] == algo]

                    if algo_data.empty:
                        continue

                    convergence_stats = (
                        algo_data.groupby("evaluation")["fitness"]
                        .agg(["median", "mean", "std"])
                        .reset_index()
                    )

                    ax.plot(
                        convergence_stats["evaluation"],
                        convergence_stats["median"],
                        label=algo,
                        linewidth=2,
                        alpha=0.8,
                    )

                    upper = convergence_stats["median"] + convergence_stats["std"]
                    lower = convergence_stats["median"] - convergence_stats["std"]
                    ax.fill_between(
                        convergence_stats["evaluation"], lower, upper, alpha=0.2
                    )

                ax.set_yscale("log")
                ax.set_xlabel("Function Evaluations")
                ax.set_ylabel("Best Fitness (log scale)")
                ax.set_title(f"{func} {dim}D")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.suptitle(f"Convergence Curves: {func}", fontsize=16)
            plt.tight_layout()
            plt.savefig(
                f"plots/convergence_{func.lower()}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def plot_statistical_analysis(self):

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        for dim in self.dimensions:
            dim_data = self.raw_data[self.raw_data["dimension"] == dim]
            means = dim_data.groupby("algorithm")["best_fitness"].mean()
            plt.plot(means.index, means.values, "o-", label=f"{dim}D", markersize=8)

        plt.yscale("log")
        plt.title("Performance vs Dimension")
        plt.ylabel("Mean Best Fitness (log)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 2)
        time_data = self.raw_data.groupby("algorithm")["execution_time"].mean()
        bars = plt.bar(time_data.index, time_data.values)
        plt.title("Average Execution Time")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
            )

        plt.subplot(2, 3, 3)
        success_by_func = (
            self.raw_data.groupby(["algorithm", "function"])["success"].mean().unstack()
        )
        success_by_func.plot(kind="bar", ax=plt.gca())
        plt.title("Success Rate by Function")
        plt.ylabel("Success Rate")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.subplot(2, 3, 4)
        consistency = self.raw_data.groupby("algorithm")["best_fitness"].agg(
            ["mean", "std"]
        )
        consistency["cv"] = consistency["std"] / consistency["mean"]

        bars = plt.bar(consistency.index, consistency["cv"])
        plt.title("Performance Consistency (CV)")
        plt.ylabel("Coefficient of Variation")
        plt.xticks(rotation=45)

        plt.subplot(2, 3, 5)

        ranking_data = []
        for func in self.functions:
            for dim in self.dimensions:
                subset = self.raw_data[
                    (self.raw_data["function"] == func)
                    & (self.raw_data["dimension"] == dim)
                ]
                if len(subset) > 0:
                    means = subset.groupby("algorithm")["best_fitness"].mean()
                    ranks = means.rank()
                    for algo, rank in ranks.items():
                        ranking_data.append({"algorithm": algo, "rank": rank})

        if ranking_data:
            rank_df = pd.DataFrame(ranking_data)
            rank_summary = rank_df.groupby("algorithm")["rank"].mean().sort_values()

            bars = plt.bar(rank_summary.index, rank_summary.values)
            plt.title("Average Ranking (Lower is Better)")
            plt.ylabel("Average Rank")
            plt.xticks(rotation=45)

            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )

        plt.subplot(2, 3, 6)

        efficiency_data = self.raw_data.groupby("algorithm").agg(
            {"success": "mean", "total_evaluations": "mean"}
        )

        plt.scatter(
            efficiency_data["total_evaluations"],
            efficiency_data["success"] * 100,
            s=100,
        )

        for algo, row in efficiency_data.iterrows():
            plt.annotate(
                algo,
                (row["total_evaluations"], row["success"] * 100),
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.xlabel("Average Evaluations Used")
        plt.ylabel("Success Rate (%)")
        plt.title("Efficiency: Success vs Evaluations")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/statistical_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def create_summary_report(self):

        rankings = self.create_performance_rankings()

        overall_scores = (
            self.raw_data.groupby("algorithm")["best_fitness"].mean().sort_values()
        )
        winner = overall_scores.index[0]

        win_counts = {}
        total_comparisons = 0

        for func in self.functions:
            for dim in self.dimensions:
                subset = self.raw_data[
                    (self.raw_data["function"] == func)
                    & (self.raw_data["dimension"] == dim)
                ]
                if len(subset) > 0:
                    means = subset.groupby("algorithm")["best_fitness"].mean()
                    if len(means) > 0:
                        best_algo = means.idxmin()
                        win_counts[best_algo] = win_counts.get(best_algo, 0) + 1
                        total_comparisons += 1

        win_percentages = {
            algo: (count / total_comparisons) * 100
            for algo, count in win_counts.items()
        }

        summary = {
            "benchmark_summary": {
                "overall_winner": winner,
                "total_experiments": len(self.raw_data),
                "algorithms_tested": len(self.algorithms),
                "functions_tested": len(self.functions),
                "dimensions_tested": len(self.dimensions),
            },
            "performance_summary": {
                "win_percentages": win_percentages,
                "overall_mean_fitness": overall_scores.to_dict(),
                "success_rates": {
                    algo: float(
                        self.raw_data[self.raw_data["algorithm"] == algo][
                            "success"
                        ].mean()
                    )
                    for algo in self.algorithms
                },
            },
            "efficiency_summary": {
                "avg_evaluations_used": {
                    algo: float(
                        self.raw_data[self.raw_data["algorithm"] == algo][
                            "total_evaluations"
                        ].mean()
                    )
                    for algo in self.algorithms
                },
                "avg_execution_time": {
                    algo: float(
                        self.raw_data[self.raw_data["algorithm"] == algo][
                            "execution_time"
                        ].mean()
                    )
                    for algo in self.algorithms
                },
            },
            "detailed_rankings": rankings,
        }

        return summary

    def generate_all_plots_and_analysis(self):

        print("Generating summary report...")
        summary = self.create_summary_report()

        with open("benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.create_text_summary(summary)

        print("\nAnalysis complete! Generated files:")
        print("- plots/performance_comparison.png")
        print("- plots/convergence_*.png (one per function)")
        print("- plots/statistical_analysis.png")
        print("- benchmark_summary.json")
        print("- benchmark_summary.txt")

    def create_text_summary(self, summary):

        with open("benchmark_summary.txt", "w") as f:
            f.write("METAHEURISTIC BENCHMARK ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Total Experiments: {summary['benchmark_summary']['total_experiments']}\n"
            )
            f.write(
                f"Algorithms Tested: {summary['benchmark_summary']['algorithms_tested']}\n"
            )
            f.write(
                f"Functions Tested: {summary['benchmark_summary']['functions_tested']}\n"
            )
            f.write(
                f"Dimensions Tested: {summary['benchmark_summary']['dimensions_tested']}\n\n"
            )

            f.write("OVERALL WINNER\n")
            f.write("-" * 20 + "\n")
            f.write(f"WINNER: {summary['benchmark_summary']['overall_winner']}\n\n")

            f.write("WIN PERCENTAGES (Best Performance per Function-Dimension)\n")
            f.write("-" * 55 + "\n")
            for algo, percentage in sorted(
                summary["performance_summary"]["win_percentages"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                f.write(f"{algo:>20}: {percentage:>6.1f}%\n")
            f.write("\n")

            f.write("SUCCESS RATES\n")
            f.write("-" * 15 + "\n")
            for algo, rate in sorted(
                summary["performance_summary"]["success_rates"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                f.write(f"{algo:>20}: {rate:>6.1%}\n")
            f.write("\n")

            f.write("AVERAGE EVALUATIONS USED\n")
            f.write("-" * 25 + "\n")
            for algo, evals in sorted(
                summary["efficiency_summary"]["avg_evaluations_used"].items(),
                key=lambda x: x[1],
            ):
                f.write(f"{algo:>20}: {evals:>8.0f}\n")
            f.write("\n")

            f.write("AVERAGE EXECUTION TIME\n")
            f.write("-" * 23 + "\n")
            for algo, time in sorted(
                summary["efficiency_summary"]["avg_execution_time"].items(),
                key=lambda x: x[1],
            ):
                f.write(f"{algo:>20}: {time:>6.1f}s\n")


def main():
    print("Metaheuristic Benchmark Analysis")
    print("=" * 40)

    try:
        analyzer = BenchmarkAnalyzer()
        analyzer.generate_all_plots_and_analysis()

        print("\n🎉 Analysis complete!")
        print("📊 Check the plots/ directory for visualizations")
        print("📋 Check benchmark_summary.json for detailed results")
        print("📄 Check benchmark_summary.txt for readable summary")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
