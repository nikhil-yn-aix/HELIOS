import cupy as cp
import numpy as np
import math
import time
from typing import Callable, Optional
from dataclasses import dataclass
from tqdm import tqdm
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import psutil
import gc
import pandas as pd
import json


class OptimizationRegime(Enum):
    EXPLORATION = 1
    EXPLOITATION = 2
    CONVERGENCE = 3


@dataclass
class HeliosMPConfig:
    population_size: int = 40000
    explorer_ratio: float = 0.5
    max_iterations: int = 200
    regime_op_dist: dict = None
    local_search_freq: int = 15
    local_search_elite_fraction: float = 0.025
    bb_ls_max_iter: int = 10
    stagnation_window: int = 25
    stagnation_threshold: float = 1e-5
    shockwave_fraction: float = 0.2
    shockwave_scale: float = 0.1
    de_factor_f: float = 0.5
    de_crossover_rate: float = 0.7
    levy_beta: float = 1.5

    def __post_init__(self):
        if self.regime_op_dist is None:
            self.regime_op_dist = {
                OptimizationRegime.EXPLORATION: (0.6, 0.2, 0.1, 0.1),
                OptimizationRegime.EXPLOITATION: (0.2, 0.5, 0.2, 0.1),
                OptimizationRegime.CONVERGENCE: (0.1, 0.3, 0.4, 0.2),
            }


class GPUOperations:
    @staticmethod
    def parallel_bb_local_search(
        positions, fitness, objective_func, bounds, max_iter=10
    ):
        lower_bounds, upper_bounds = bounds
        x_k = cp.copy(positions)
        f_k = fitness

        grad_k = GPUOperations.parallel_gradient_approx(x_k, objective_func, bounds)

        for _ in range(max_iter):
            s_k = -grad_k
            alpha_k = 0.01
            x_k_plus_1 = cp.clip(x_k + alpha_k * s_k, lower_bounds, upper_bounds)
            grad_k_plus_1 = GPUOperations.parallel_gradient_approx(
                x_k_plus_1, objective_func, bounds
            )

            s_k = x_k_plus_1 - x_k
            y_k = grad_k_plus_1 - grad_k

            x_k = x_k_plus_1
            grad_k = grad_k_plus_1

        final_fitness = objective_func(x_k)
        return x_k, final_fitness

    @staticmethod
    def parallel_gradient_approx(x, objective_func, bounds, epsilon=1e-5):
        lower_bounds, upper_bounds = bounds
        grad = cp.zeros_like(x)
        for d in range(x.shape[1]):
            h = cp.zeros_like(x)
            h[:, d] = epsilon
            f_plus_h = objective_func(cp.clip(x + h, lower_bounds, upper_bounds))
            f_minus_h = objective_func(cp.clip(x - h, lower_bounds, upper_bounds))
            grad[:, d] = (f_plus_h - f_minus_h) / (2 * epsilon)
        return grad

    @staticmethod
    def generate_levy_flights(n_samples, dimensions, beta=1.5, scale=0.1):
        num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (num / den) ** (1 / beta)
        u = cp.random.normal(0, sigma, (int(n_samples), int(dimensions)))
        v = cp.random.normal(0, 1, (int(n_samples), int(dimensions)))
        levy_steps = u / (cp.abs(v) ** (1 / beta))
        return scale * levy_steps


class HeliosMP:
    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        config: HeliosMPConfig,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            cp.random.seed(seed)

        self.config = config
        self.objective_func = objective_func
        self.bounds = cp.asarray(bounds)
        self.dimensions = int(self.bounds.shape[0])
        self.lower_bounds, self.upper_bounds = self.bounds[:, 0], self.bounds[:, 1]
        self.range_width = self.upper_bounds - self.lower_bounds

        self.num_explorers = int(config.population_size * config.explorer_ratio)
        self.num_refiners = config.population_size - self.num_explorers

        self.explorers = cp.random.uniform(
            self.lower_bounds, self.upper_bounds, (self.num_explorers, self.dimensions)
        )
        self.refiners = cp.random.uniform(
            self.lower_bounds, self.upper_bounds, (self.num_refiners, self.dimensions)
        )

        self.explorer_fitness = cp.full(self.num_explorers, cp.inf)
        self.refiner_fitness = cp.full(self.num_refiners, cp.inf)

        self.global_best_fitness = cp.inf
        self.global_best_position = None
        self.evaluation_count = 0
        self.current_iteration = 0

        self.regime = OptimizationRegime.EXPLORATION
        self.fitness_history = []
        self.last_shockwave_iter = 0

        self._initial_evaluation()

    def _initial_evaluation(self):
        self.explorer_fitness = self.objective_func(self.explorers)
        self.refiner_fitness = self.objective_func(self.refiners)
        self.evaluation_count += self.config.population_size
        self._update_global_best()

    def _update_global_best(self):
        min_explorer_fitness = cp.min(self.explorer_fitness)
        min_refiner_fitness = cp.min(self.refiner_fitness)

        if min_explorer_fitness < min_refiner_fitness:
            current_best_fitness = min_explorer_fitness
            best_agent_position = self.explorers[cp.argmin(self.explorer_fitness)]
        else:
            current_best_fitness = min_refiner_fitness
            best_agent_position = self.refiners[cp.argmin(self.refiner_fitness)]

        if current_best_fitness < self.global_best_fitness:
            self.global_best_fitness = current_best_fitness
            self.global_best_position = best_agent_position.copy()

        self.fitness_history.append(float(self.global_best_fitness))

    def _update_regime(self):
        progress_ratio = self.current_iteration / self.config.max_iterations
        if progress_ratio < 0.4:
            self.regime = OptimizationRegime.EXPLORATION
        elif progress_ratio < 0.8:
            self.regime = OptimizationRegime.EXPLOITATION
        else:
            self.regime = OptimizationRegime.CONVERGENCE

    def _run_parallel_operators(self):
        all_pop = cp.vstack([self.explorers, self.refiners])
        candidates = cp.copy(all_pop)

        p_levy, p_de, p_best, p_rand = self.config.regime_op_dist[self.regime]
        choices = cp.random.rand(self.config.population_size)

        mask = choices < p_levy
        if cp.any(mask):
            n = int(cp.sum(mask))
            levy_steps = GPUOperations.generate_levy_flights(
                n, self.dimensions, self.config.levy_beta
            )
            candidates[mask] += levy_steps * self.range_width

        mask = (choices >= p_levy) & (choices < p_levy + p_de)
        if cp.any(mask):
            n = int(cp.sum(mask))
            p1, p2, p3 = cp.random.randint(0, self.config.population_size, (3, n))
            mutant = all_pop[p1] + self.config.de_factor_f * (all_pop[p2] - all_pop[p3])
            crossover_mask = (
                cp.random.rand(n, self.dimensions) < self.config.de_crossover_rate
            )
            candidates[mask] = cp.where(crossover_mask, mutant, all_pop[mask])

        mask = (choices >= p_levy + p_de) & (choices < p_levy + p_de + p_best)
        if cp.any(mask) and self.global_best_position is not None:
            n = int(cp.sum(mask))
            direction = self.global_best_position - all_pop[mask]
            step_sizes = cp.random.uniform(0.1, 1.0, (n, 1))
            candidates[mask] += step_sizes * direction

        mask = choices >= p_levy + p_de + p_best
        if cp.any(mask):
            n = int(cp.sum(mask))
            candidates[mask] = cp.random.uniform(
                self.lower_bounds, self.upper_bounds, (n, self.dimensions)
            )

        return cp.clip(candidates, self.lower_bounds, self.upper_bounds)

    def _evaluate_and_select(self, candidates):
        candidate_fitness = self.objective_func(candidates)
        self.evaluation_count += self.config.population_size

        current_fitness = cp.hstack([self.explorer_fitness, self.refiner_fitness])
        current_pop = cp.vstack([self.explorers, self.refiners])

        improvement_mask = candidate_fitness < current_fitness

        new_pop = cp.where(improvement_mask[:, cp.newaxis], candidates, current_pop)
        new_fitness = cp.where(improvement_mask, candidate_fitness, current_fitness)

        self.explorers = new_pop[: self.num_explorers]
        self.refiners = new_pop[self.num_explorers :]
        self.explorer_fitness = new_fitness[: self.num_explorers]
        self.refiner_fitness = new_fitness[self.num_refiners :]

    def _run_local_search_phase(self):
        num_elite = int(
            self.config.population_size * self.config.local_search_elite_fraction
        )

        all_fitness = cp.hstack([self.explorer_fitness, self.refiner_fitness])
        elite_indices = cp.argpartition(all_fitness, num_elite)[:num_elite]

        all_pop = cp.vstack([self.explorers, self.refiners])
        elite_positions = all_pop[elite_indices]
        elite_fitness = all_fitness[elite_indices]

        improved_positions, improved_fitness = GPUOperations.parallel_bb_local_search(
            elite_positions,
            elite_fitness,
            self.objective_func,
            (self.lower_bounds, self.upper_bounds),
            max_iter=self.config.bb_ls_max_iter,
        )

        all_pop[elite_indices] = improved_positions
        all_fitness[elite_indices] = improved_fitness

        self.explorers = all_pop[: self.num_explorers]
        self.refiners = all_pop[self.num_explorers :]
        self.explorer_fitness = all_fitness[: self.num_explorers]
        self.refiner_fitness = all_fitness[self.num_refiners :]

        self.evaluation_count += num_elite * (
            self.config.bb_ls_max_iter * (2 * self.dimensions)
        )

    def _run_stagnation_shockwave(self):
        if (
            self.current_iteration - self.last_shockwave_iter
            < self.config.stagnation_window
        ):
            return

        if len(self.fitness_history) < self.config.stagnation_window:
            return

        past_best = self.fitness_history[-self.config.stagnation_window]
        current_best = self.fitness_history[-1]
        improvement = (past_best - current_best) / (abs(past_best) + 1e-9)

        if improvement < self.config.stagnation_threshold:
            num_to_shock = int(
                self.config.population_size * self.config.shockwave_fraction
            )
            indices_to_shock = cp.random.choice(
                self.config.population_size, num_to_shock, replace=False
            )

            perturbations = (
                cp.random.normal(
                    0, self.config.shockwave_scale, (num_to_shock, self.dimensions)
                )
                * self.range_width
            )

            all_pop = cp.vstack([self.explorers, self.refiners])
            all_pop[indices_to_shock] = cp.clip(
                all_pop[indices_to_shock] + perturbations,
                self.lower_bounds,
                self.upper_bounds,
            )

            self.explorers = all_pop[: self.num_explorers]
            self.refiners = all_pop[self.num_refiners :]

            all_fitness = cp.hstack([self.explorer_fitness, self.refiner_fitness])
            all_fitness[indices_to_shock] = cp.inf
            self.explorer_fitness = all_fitness[: self.num_explorers]
            self.refiner_fitness = all_fitness[self.num_refiners :]

            self.last_shockwave_iter = self.current_iteration

    def optimize(self):
        pbar = tqdm(range(self.config.max_iterations), desc="Helios-MP", unit="iter")

        for iteration in pbar:
            self.current_iteration = iteration

            self._update_regime()

            candidates = self._run_parallel_operators()

            self._evaluate_and_select(candidates)

            if iteration % self.config.local_search_freq == 0 and iteration > 0:
                self._run_local_search_phase()

            self._run_stagnation_shockwave()

            self._update_global_best()

            pbar.set_postfix(
                {
                    "Best": f"{float(self.global_best_fitness):.4e}",
                    "Regime": self.regime.name[0],
                    "Evals": f"{self.evaluation_count/1e6:.2f}M",
                }
            )

        pbar.close()
        return cp.asnumpy(self.global_best_position), float(self.global_best_fitness)


def sphere_function(x):
    if hasattr(x, "get"):
        return cp.sum(x**2, axis=-1)
    else:
        return np.sum(x**2, axis=-1)


def rastrigin_function(x):
    A = 10
    if hasattr(x, "get"):
        n = x.shape[-1]
        return A * n + cp.sum(x**2 - A * cp.cos(2 * cp.pi * x), axis=-1)
    else:
        n = x.shape[-1]
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=-1)


def rosenbrock_function(x):
    if hasattr(x, "get"):
        return cp.sum(
            100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (x[..., :-1] - 1) ** 2,
            axis=-1,
        )
    else:
        return np.sum(
            100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (x[..., :-1] - 1) ** 2,
            axis=-1,
        )


def ackley_function(x):
    if hasattr(x, "get"):
        n = x.shape[-1]
        sum_sq = cp.sum(x**2, axis=-1) / n
        sum_cos = cp.sum(cp.cos(2.0 * cp.pi * x), axis=-1) / n
        return -20.0 * cp.exp(-0.2 * cp.sqrt(sum_sq)) - cp.exp(sum_cos) + 20.0 + cp.e
    else:
        n = x.shape[-1]
        sum_sq = np.sum(x**2, axis=-1) / n
        sum_cos = np.sum(np.cos(2.0 * np.pi * x), axis=-1) / n
        return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20.0 + np.e


def griewank_function(x):
    if hasattr(x, "get"):
        sum_sq = cp.sum(x**2, axis=-1) / 4000.0
        indices = cp.arange(1, x.shape[-1] + 1)
        prod_cos = cp.prod(cp.cos(x / cp.sqrt(indices)), axis=-1)
        return sum_sq - prod_cos + 1
    else:
        sum_sq = np.sum(x**2, axis=-1) / 4000.0
        indices = np.arange(1, x.shape[-1] + 1)
        prod_cos = np.prod(np.cos(x / np.sqrt(indices)), axis=-1)
        return sum_sq - prod_cos + 1


def levy_function(x):
    if hasattr(x, "get"):
        w = 1 + (x - 1) / 4
        term1 = cp.sin(cp.pi * w[..., 0]) ** 2

        if x.shape[-1] == 1:
            return term1

        term2 = cp.sum(
            (w[..., :-1] - 1) ** 2 * (1 + 10 * cp.sin(cp.pi * w[..., :-1] + 1) ** 2),
            axis=-1,
        )
        term3 = (w[..., -1] - 1) ** 2 * (1 + cp.sin(2 * cp.pi * w[..., -1]) ** 2)
        return term1 + term2 + term3
    else:
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[..., 0]) ** 2

        if x.shape[-1] == 1:
            return term1

        term2 = np.sum(
            (w[..., :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[..., :-1] + 1) ** 2),
            axis=-1,
        )
        term3 = (w[..., -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[..., -1]) ** 2)
        return term1 + term2 + term3


def comprehensive_scaling_analysis(objective_func, func_name, bounds_template):
    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE HELIOS-MP SCALING ANALYSIS - {func_name.upper()}")
    print("=" * 80)

    dimensions_list = [10, 20, 30, 50, 100, 200]
    population_sizes = [5000, 10000, 20000, 40000, 80000, 160000]

    results = defaultdict(list)
    convergence_data = {}

    total_tests = 0
    for dim in dimensions_list:
        for pop_size in population_sizes:
            estimated_memory_gb = (pop_size * dim * 8 * 10) / (1024**3)
            if estimated_memory_gb <= 16:
                total_tests += 1

    print(f"Function: {func_name}")
    print(f"Bounds template: {bounds_template}")
    print(f"Total tests to run: {total_tests}")
    print(f"Estimated duration: {total_tests * 20:.0f}-{total_tests * 90:.0f} seconds")
    print()

    with tqdm(
        total=total_tests,
        desc=f"{func_name} Analysis",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        position=0,
        leave=True,
    ) as pbar_main:

        for dim_idx, dim in enumerate(dimensions_list):
            bounds = np.array([bounds_template] * dim)

            valid_pop_sizes = []
            for pop_size in population_sizes:
                estimated_memory_gb = (pop_size * dim * 8 * 10) / (1024**3)
                if estimated_memory_gb <= 16:
                    valid_pop_sizes.append(pop_size)

            if not valid_pop_sizes:
                continue

            dim_desc = f"D={dim} ({len(valid_pop_sizes)} populations)"
            with tqdm(
                valid_pop_sizes,
                desc=dim_desc,
                bar_format="{desc}: {n_fmt}/{total_fmt}|{bar}|",
                position=1,
                leave=False,
            ) as pbar_dim:

                for pop_idx, pop_size in enumerate(pbar_dim):
                    test_desc = f"D={dim}, P={pop_size//1000}K"
                    pbar_main.set_description(test_desc)

                    if "cp" in globals():
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    gc.collect()

                    initial_gpu_memory = (
                        cp.cuda.Device().mem_info[0] if "cp" in globals() else 0
                    )

                    initial_cpu_memory = psutil.Process().memory_info().rss

                    config = HeliosMPConfig(
                        population_size=pop_size,
                        max_iterations=200,
                        local_search_freq=20,
                        stagnation_window=15,
                    )

                    timing_breakdown = {}
                    total_start = time.time()

                    init_desc = "Initializing"
                    with tqdm(
                        total=1,
                        desc=init_desc,
                        position=2,
                        leave=False,
                        bar_format="{desc}: {percentage:3.0f}%",
                    ) as pbar_init:
                        init_start = time.time()
                        optimizer = HeliosMP(objective_func, bounds, config, seed=42)
                        init_time = time.time() - init_start
                        timing_breakdown["initialization"] = init_time
                        pbar_init.update(1)

                    post_init_gpu_memory = (
                        cp.cuda.Device().mem_info[0] if "cp" in globals() else 0
                    )

                    post_init_cpu_memory = psutil.Process().memory_info().rss

                    convergence_history = []
                    evaluation_history = []

                    optimization_start = time.time()
                    opt_desc = f"Optimizing ({config.max_iterations} iters)"

                    with tqdm(
                        total=config.max_iterations,
                        desc=opt_desc,
                        position=2,
                        leave=False,
                        unit="iter",
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
                    ) as pbar_opt:

                        for iteration in range(config.max_iterations):
                            iter_start = time.time()

                            optimizer.current_iteration = iteration
                            optimizer._update_regime()

                            candidates = optimizer._run_parallel_operators()
                            optimizer._evaluate_and_select(candidates)

                            if (
                                iteration % config.local_search_freq == 0
                                and iteration > 0
                            ):
                                optimizer._run_local_search_phase()

                            optimizer._run_stagnation_shockwave()
                            optimizer._update_global_best()

                            iter_time = time.time() - iter_start

                            convergence_history.append(
                                float(optimizer.global_best_fitness)
                            )
                            evaluation_history.append(optimizer.evaluation_count)

                            if len(convergence_history) > 20 and all(
                                abs(convergence_history[-1] - convergence_history[-i])
                                < 1e-12
                                for i in range(1, min(11, len(convergence_history)))
                            ):
                                pbar_opt.set_description("✅ Converged early")
                                break

                            if iteration % max(1, config.max_iterations // 20) == 0:
                                pbar_opt.set_postfix(
                                    {
                                        "Best": f"{optimizer.global_best_fitness:.2e}",
                                        "Regime": optimizer.regime.name[0],
                                        "Evals": f"{optimizer.evaluation_count//1000}K",
                                    }
                                )

                            pbar_opt.update(1)

                    optimization_time = time.time() - optimization_start
                    timing_breakdown["optimization"] = optimization_time
                    total_time = time.time() - total_start

                    final_gpu_memory = (
                        cp.cuda.Device().mem_info[0] if "cp" in globals() else 0
                    )

                    final_cpu_memory = psutil.Process().memory_info().rss

                    if (
                        initial_gpu_memory > 0
                        and final_gpu_memory <= initial_gpu_memory
                    ):
                        peak_gpu_memory = initial_gpu_memory - final_gpu_memory
                    else:
                        peak_gpu_memory = max(1024**2, pop_size * dim * 8)

                    peak_gpu_memory = max(peak_gpu_memory, 1024**2)

                    evals_per_second = (
                        optimizer.evaluation_count / total_time if total_time > 0 else 0
                    )
                    memory_per_eval = (
                        peak_gpu_memory / optimizer.evaluation_count
                        if optimizer.evaluation_count > 0
                        else 0
                    )

                    if len(convergence_history) > 0:
                        final_fitness = convergence_history[-1]
                        if final_fitness > 0:
                            convergence_rate = -np.log10(final_fitness + 1e-15) / len(
                                convergence_history
                            )
                        else:
                            convergence_rate = len(convergence_history)
                    else:
                        final_fitness = float(optimizer.global_best_fitness)
                        convergence_rate = 0

                    gpu_utilization = min(
                        100.0, (peak_gpu_memory / (16 * 1024**3)) * 100
                    )

                    result_entry = {
                        "dimension": dim,
                        "population": pop_size,
                        "total_time": total_time,
                        "init_time": init_time,
                        "opt_time": optimization_time,
                        "evaluations": optimizer.evaluation_count,
                        "evals_per_second": evals_per_second,
                        "final_fitness": final_fitness,
                        "convergence_rate": max(0, convergence_rate),
                        "peak_gpu_memory_mb": peak_gpu_memory / (1024**2),
                        "memory_per_eval_kb": memory_per_eval / 1024,
                        "gpu_utilization": gpu_utilization,
                        "regime_counts": {
                            "exploration": sum(
                                1 for i in range(int(0.4 * len(convergence_history)))
                            ),
                            "exploitation": sum(
                                1
                                for i in range(
                                    int(0.4 * len(convergence_history)),
                                    int(0.8 * len(convergence_history)),
                                )
                            ),
                            "convergence": sum(
                                1
                                for i in range(
                                    int(0.8 * len(convergence_history)),
                                    len(convergence_history),
                                )
                            ),
                        },
                    }

                    results["all"].append(result_entry)

                    key = f"D{dim}_P{pop_size}"
                    convergence_data[key] = {
                        "fitness": convergence_history,
                        "evaluations": evaluation_history,
                        "dimension": dim,
                        "population": pop_size,
                    }

                    pbar_main.set_postfix(
                        {
                            "Time": f"{total_time:.1f}s",
                            "Evals/s": f"{evals_per_second:.0f}",
                            "Memory": f"{peak_gpu_memory/(1024**2):.0f}MB",
                            "Fitness": f"{final_fitness:.1e}",
                        }
                    )

                    pbar_main.update(1)

    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS FOR {func_name.upper()}")
    print(f"{'='*70}")

    visualization_steps = [
        "3D Performance Surface",
        "Memory Usage Heatmap",
        "Efficiency Scaling",
        "Convergence Comparison",
        "Memory Efficiency Analysis",
        "Regime Distribution",
    ]

    with tqdm(
        visualization_steps,
        desc="Creating plots",
        bar_format="{desc}: {n_fmt}/{total_fmt}|{bar}|",
    ) as pbar_viz:

        valid_results = [r for r in results["all"] if r["evals_per_second"] > 0]

        if not valid_results:
            print("❌ No valid results for visualization")
            return results, convergence_data

        fig1 = plt.figure(figsize=(14, 10))
        ax1 = fig1.add_subplot(111, projection="3d")

        dims = [r["dimension"] for r in valid_results]
        pops = [r["population"] for r in valid_results]
        evals_per_sec = [r["evals_per_second"] for r in valid_results]

        scatter = ax1.scatter(
            dims, pops, evals_per_sec, c=evals_per_sec, cmap="viridis", s=80, alpha=0.8
        )
        ax1.set_xlabel("Dimensions", fontsize=12)
        ax1.set_ylabel("Population Size", fontsize=12)
        ax1.set_zlabel("Evaluations/Second", fontsize=12)
        ax1.set_title(
            f"Helios-MP Performance Scaling Surface - {func_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.colorbar(scatter, ax=ax1, shrink=0.6, label="Evals/Second")
        fig1.savefig(
            f"1_helios_mp_{func_name.lower()}_performance_surface.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig1)
        pbar_viz.update(1)

        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111)

        unique_dims = sorted(set(dims))
        unique_pops = sorted(set(pops))
        memory_matrix = np.full((len(unique_dims), len(unique_pops)), np.nan)

        for r in valid_results:
            if r["dimension"] in unique_dims and r["population"] in unique_pops:
                dim_idx = unique_dims.index(r["dimension"])
                pop_idx = unique_pops.index(r["population"])
                memory_matrix[dim_idx, pop_idx] = r["peak_gpu_memory_mb"]

        mask = ~np.isnan(memory_matrix)
        if np.any(mask):
            sns.heatmap(
                memory_matrix,
                xticklabels=[f"{p//1000}K" for p in unique_pops],
                yticklabels=unique_dims,
                annot=True,
                fmt=".0f",
                cmap="Reds",
                ax=ax2,
                mask=~mask,
            )
        ax2.set_title(
            f"Helios-MP Peak GPU Memory Usage (MB) - {func_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("Population Size", fontsize=12)
        ax2.set_ylabel("Dimensions", fontsize=12)
        fig2.savefig(
            f"2_helios_mp_{func_name.lower()}_memory_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig2)
        pbar_viz.update(1)

        fig3 = plt.figure(figsize=(12, 8))
        ax3 = fig3.add_subplot(111)

        for dim in sorted(set(dims))[:5]:
            dim_data = [r for r in valid_results if r["dimension"] == dim]
            if dim_data:
                pops_dim = [r["population"] for r in dim_data]
                efficiency = [r["evals_per_second"] / r["population"] for r in dim_data]
                ax3.plot(
                    pops_dim,
                    efficiency,
                    marker="o",
                    label=f"D={dim}",
                    linewidth=2.5,
                    markersize=6,
                )

        ax3.set_xlabel("Population Size", fontsize=12)
        ax3.set_ylabel("Efficiency (Evals/sec/Agent)", fontsize=12)
        ax3.set_title(
            f"Helios-MP Computational Efficiency Scaling - {func_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax3.legend(fontsize=10)
        ax3.set_xscale("log")
        ax3.grid(True, alpha=0.4)
        fig3.savefig(
            f"3_helios_mp_{func_name.lower()}_efficiency_scaling.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig3)
        pbar_viz.update(1)

        fig4 = plt.figure(figsize=(14, 8))
        ax4 = fig4.add_subplot(111)

        sample_keys = list(convergence_data.keys())[:8]
        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_keys)))

        for i, key in enumerate(sample_keys):
            data = convergence_data[key]
            if data["fitness"]:
                iterations = range(len(data["fitness"]))
                ax4.semilogy(
                    iterations,
                    data["fitness"],
                    label=f"D{data['dimension']}_P{data['population']//1000}K",
                    alpha=0.8,
                    linewidth=2,
                    color=colors[i],
                )

        ax4.set_xlabel("Iterations", fontsize=12)
        ax4.set_ylabel("Best Fitness (log scale)", fontsize=12)
        ax4.set_title(
            f"Helios-MP Convergence Behavior - {func_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax4.grid(True, alpha=0.4)
        fig4.savefig(
            f"4_helios_mp_{func_name.lower()}_convergence.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig4)
        pbar_viz.update(1)

        fig5 = plt.figure(figsize=(12, 8))
        ax5 = fig5.add_subplot(111)

        memory_per_eval = [r["memory_per_eval_kb"] for r in valid_results]
        total_params = [r["dimension"] * r["population"] for r in valid_results]

        scatter2 = ax5.scatter(
            total_params,
            memory_per_eval,
            c=[r["dimension"] for r in valid_results],
            cmap="plasma",
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )
        ax5.set_xlabel("Total Parameters (Dim × Population)", fontsize=12)
        ax5.set_ylabel("Memory per Evaluation (KB)", fontsize=12)
        ax5.set_title(
            f"Helios-MP Memory Efficiency Analysis - {func_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax5.set_xscale("log")
        ax5.set_yscale("log")
        plt.colorbar(scatter2, ax=ax5, label="Dimensions")
        ax5.grid(True, alpha=0.4)
        fig5.savefig(
            f"5_helios_mp_{func_name.lower()}_memory_efficiency.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig5)
        pbar_viz.update(1)

        fig6 = plt.figure(figsize=(12, 8))
        ax6 = fig6.add_subplot(111)

        regime_data = []
        for r in valid_results[:20]:
            total_iters = sum(r["regime_counts"].values())
            if total_iters > 0:
                exploration_pct = r["regime_counts"]["exploration"] / total_iters * 100
                exploitation_pct = (
                    r["regime_counts"]["exploitation"] / total_iters * 100
                )
                convergence_pct = r["regime_counts"]["convergence"] / total_iters * 100
                regime_data.append([exploration_pct, exploitation_pct, convergence_pct])

        if regime_data:
            regime_array = np.array(regime_data)
            x_pos = np.arange(len(regime_data))

            ax6.bar(
                x_pos,
                regime_array[:, 0],
                label="Exploration",
                color="lightblue",
                alpha=0.8,
            )
            ax6.bar(
                x_pos,
                regime_array[:, 1],
                bottom=regime_array[:, 0],
                label="Exploitation",
                color="orange",
                alpha=0.8,
            )
            ax6.bar(
                x_pos,
                regime_array[:, 2],
                bottom=regime_array[:, 0] + regime_array[:, 1],
                label="Convergence",
                color="lightgreen",
                alpha=0.8,
            )

        ax6.set_xlabel("Configuration Index", fontsize=12)
        ax6.set_ylabel("Regime Distribution (%)", fontsize=12)
        ax6.set_title(
            f"Helios-MP Optimization Regime Distribution - {func_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.4, axis="y")
        fig6.savefig(
            f"6_helios_mp_{func_name.lower()}_regime_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig6)
        pbar_viz.update(1)

    print(f"\n{'='*70}")
    print(f"STATISTICAL ANALYSIS FOR {func_name.upper()}")
    print(f"{'='*70}")

    if valid_results:
        stats_df = pd.DataFrame(valid_results)
        summary_stats = (
            stats_df.groupby("dimension")
            .agg(
                {
                    "evals_per_second": ["mean", "std", "max"],
                    "peak_gpu_memory_mb": ["mean", "std", "max"],
                    "convergence_rate": ["mean", "std"],
                    "final_fitness": ["mean", "std", "min"],
                }
            )
            .round(3)
        )

        print(f"\nSUMMARY BY DIMENSION for {func_name}:")
        print(summary_stats.to_string())

        populations = [r["population"] for r in valid_results]
        evals_per_sec = [r["evals_per_second"] for r in valid_results]
        memory_mb = [max(1, r["peak_gpu_memory_mb"]) for r in valid_results]

        if len(populations) > 1:
            log_populations = np.log10(populations)
            log_evals_per_sec = np.log10([max(1, eps) for eps in evals_per_sec])
            log_memory = np.log10(memory_mb)

            try:
                perf_coeff = np.polyfit(log_populations, log_evals_per_sec, 1)[0]
                memory_coeff = np.polyfit(log_populations, log_memory, 1)[0]
            except:
                perf_coeff = 0.0
                memory_coeff = 1.0

            print(f"\nSCALING ANALYSIS INSIGHTS for {func_name}:")
            print(f"   Performance Scaling Exponent: {perf_coeff:.3f}")
            print(f"   Memory Scaling Exponent: {memory_coeff:.3f}")
            print(
                f"   Scaling Quality: {'Good' if perf_coeff > 0.7 else 'Moderate' if perf_coeff > 0.4 else 'Poor'}"
            )

            efficiency_scores = []
            for r in valid_results:
                if r["peak_gpu_memory_mb"] > 0:
                    score = r["evals_per_second"] / (r["peak_gpu_memory_mb"] / 1000)
                    efficiency_scores.append((r["dimension"], r["population"], score))

            if efficiency_scores:
                best_config = max(efficiency_scores, key=lambda x: x[2])
                print(f"\n🎯 OPTIMAL CONFIGURATION for {func_name}:")
                print(f"   Dimensions: {best_config[0]}")
                print(f"   Population: {best_config[1]:,}")
                print(f"   Efficiency Score: {best_config[2]:.2f}")
        else:
            perf_coeff = memory_coeff = 0.0
            best_config = (0, 0, 0.0)

    print(f"\n{'='*50}")
    print("GENERATING REPORTS")
    print(f"{'='*50}")

    json_filename = f"helios_mp_{func_name.lower()}_scaling_report.json"
    json_report = {
        "metadata": {
            "function_name": func_name,
            "bounds_template": bounds_template,
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(results["all"]),
            "successful_tests": len(valid_results),
            "dimensions_tested": dimensions_list,
            "population_sizes_tested": population_sizes,
        },
        "results": results["all"],
        "convergence_data": {k: v for k, v in convergence_data.items()},
        "scaling_analysis": {
            "performance_exponent": (
                float(perf_coeff) if "perf_coeff" in locals() else 0.0
            ),
            "memory_exponent": (
                float(memory_coeff) if "memory_coeff" in locals() else 0.0
            ),
            "optimal_config": {
                "dimensions": int(best_config[0]) if "best_config" in locals() else 0,
                "population": int(best_config[1]) if "best_config" in locals() else 0,
                "efficiency_score": (
                    float(best_config[2]) if "best_config" in locals() else 0.0
                ),
            },
        },
    }

    with open(json_filename, "w") as json_file:
        json.dump(json_report, json_file, indent=2, default=str)

    txt_filename = f"helios_mp_{func_name.lower()}_scaling_report.txt"
    with open(txt_filename, "w", encoding="utf-8") as txt_file:
        txt_file.write("HELIOS-MP SCALING ANALYSIS REPORT\n")
        txt_file.write("=" * 80 + "\n")
        txt_file.write(f"Function: {func_name}\n")
        txt_file.write(f"Bounds: {bounds_template}\n")
        txt_file.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        txt_file.write(f"Total Tests: {len(results['all'])}\n")
        txt_file.write(f"Successful Tests: {len(valid_results)}\n\n")

        if valid_results:
            txt_file.write("PERFORMANCE SUMMARY\n")
            txt_file.write("-" * 50 + "\n")

            best_perf = max(valid_results, key=lambda x: x["evals_per_second"])
            txt_file.write(
                f"Best Performance: {best_perf['evals_per_second']:,.0f} evals/sec\n"
            )
            txt_file.write(
                f"  Configuration: D={best_perf['dimension']}, P={best_perf['population']:,}\n"
            )
            txt_file.write(
                f"  Memory Usage: {best_perf['peak_gpu_memory_mb']:.1f} MB\n\n"
            )

            txt_file.write("SCALING ANALYSIS\n")
            txt_file.write("-" * 50 + "\n")
            if "perf_coeff" in locals():
                txt_file.write(f"Performance Scaling Exponent: {perf_coeff:.3f}\n")
                txt_file.write(f"Memory Scaling Exponent: {memory_coeff:.3f}\n")
                txt_file.write(
                    f"Scaling Assessment: {'Good' if perf_coeff > 0.7 else 'Moderate' if perf_coeff > 0.4 else 'Poor'}\n\n"
                )

            if "best_config" in locals():
                txt_file.write("OPTIMAL CONFIGURATION\n")
                txt_file.write("-" * 50 + "\n")
                txt_file.write(f"Recommended Dimensions: {best_config[0]}\n")
                txt_file.write(f"Recommended Population: {best_config[1]:,}\n")
                txt_file.write(f"Efficiency Score: {best_config[2]:.2f}\n\n")

            txt_file.write("DETAILED RESULTS BY DIMENSION\n")
            txt_file.write("-" * 50 + "\n")
            for dim in sorted(set([r["dimension"] for r in valid_results])):
                dim_results = [r for r in valid_results if r["dimension"] == dim]
                if dim_results:
                    avg_perf = np.mean([r["evals_per_second"] for r in dim_results])
                    avg_memory = np.mean([r["peak_gpu_memory_mb"] for r in dim_results])
                    best_fitness = min([r["final_fitness"] for r in dim_results])

                    txt_file.write(f"D={dim:3d}: Avg Perf: {avg_perf:8.0f} evals/s, ")
                    txt_file.write(f"Avg Memory: {avg_memory:6.1f} MB, ")
                    txt_file.write(f"Best Fitness: {best_fitness:.2e}\n")

        txt_file.write("\nGenerated Files:\n")
        txt_file.write(f"- 1_helios_mp_{func_name.lower()}_performance_surface.png\n")
        txt_file.write(f"- 2_helios_mp_{func_name.lower()}_memory_heatmap.png\n")
        txt_file.write(f"- 3_helios_mp_{func_name.lower()}_efficiency_scaling.png\n")
        txt_file.write(f"- 4_helios_mp_{func_name.lower()}_convergence.png\n")
        txt_file.write(f"- 5_helios_mp_{func_name.lower()}_memory_efficiency.png\n")
        txt_file.write(f"- 6_helios_mp_{func_name.lower()}_regime_distribution.png\n")
        txt_file.write(f"- {json_filename}\n")
        txt_file.write(f"- {txt_filename}\n")

    print(f"✅ COMPREHENSIVE SCALING ANALYSIS for {func_name.upper()} COMPLETE!")
    print("Generated Files:")
    print("   6 visualization plots")
    print(f"   JSON report: {json_filename}")
    print(f"   TXT report: {txt_filename}")
    print(f"   Tests completed: {len(valid_results)}/{len(results['all'])}")

    return results, convergence_data


if __name__ == "__main__":
    import cupy as cp

    print("CuPy detected - GPU acceleration enabled")
    print(f"GPU Device: {cp.cuda.Device()}")
    print(f"GPU Memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB")

    available_memory_gb = cp.cuda.Device().mem_info[0] / 1024**3
    print(f"Available GPU Memory: {available_memory_gb:.1f} GB")

    if available_memory_gb < 4:
        print("Warning: Low GPU memory detected. Consider reducing population sizes.")

    print("\n" + "=" * 60)
    print("HELIOS-MP SCALING ANALYSIS SUITE")
    print("=" * 60)

    print("\nANALYSIS 1: SPHERE FUNCTION")
    print("   Characteristics: Smooth, unimodal, simple optimization")
    results_sphere, convergence_sphere = comprehensive_scaling_analysis(
        sphere_function, "Sphere", [-10, 10]
    )

    print("\nANALYSIS 2: RASTRIGIN FUNCTION")
    print("   Characteristics: Highly multimodal, many local minima, challenging")
    results_rastrigin, convergence_rastrigin = comprehensive_scaling_analysis(
        rastrigin_function, "Rastrigin", [-5.12, 5.12]
    )

    print("\n" + "=" * 60)
    print("HELIOS-MP SCALING ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Summary:")
    print(f"   ✅ Sphere function: {len(results_sphere['all'])} tests")
    print(f"   ✅ Rastrigin function: {len(results_rastrigin['all'])} tests")
    print("   Generated 12 plots total (6 per function)")
    print("   Generated 4 reports total (JSON + TXT per function)")
    print("\nUse these results for:")
    print("   - Performance optimization")
    print("   - Resource allocation planning")
    print("   - Algorithm parameter tuning")
    print("   - Publication figures and tables")
