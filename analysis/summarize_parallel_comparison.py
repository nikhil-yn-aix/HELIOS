import pandas as pd
import os


def analyze_parallel_comparison(
    filename="results/parallel_comparison/parallel_baselines_comparison_results.csv",
):

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found in the current directory.")
        print("Please make sure you have run the comparison script first.")
        return

    summary = df.groupby(["Function", "Algorithm"]).agg(
        Median_Fitness=("FinalFitness", "median"),
        Best_Fitness=("FinalFitness", "min"),
        Mean_Time_s=("ExecutionTime", "mean"),
    )

    print("=" * 80)
    print("Analysis of Parallel Baseline Comparison Results")
    print(f"Source: {filename}")
    print("=" * 80)

    styled_summary = summary.style.format(
        {"Median_Fitness": "{:.4e}", "Best_Fitness": "{:.4e}", "Mean_Time_s": "{:.2f}s"}
    ).set_properties(**{"text-align": "left"})

    def highlight_min(s):
        is_min = s == s.min()
        return ["font-weight: bold" if v else "" for v in is_min]

    print(styled_summary.to_string())

    print("\n" + "=" * 80)


if __name__ == "__main__":

    if os.path.exists("parallel_baselines_comparison_results.csv"):
        analyze_parallel_comparison()
    else:
        print("Could not find 'parallel_baselines_comparison_results.csv'.")
        print("Please run 'run_parallel_comparison.py' first to generate the results.")
