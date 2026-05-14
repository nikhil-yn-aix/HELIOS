import time
import logging
import math
from typing import Callable, Tuple, Optional
import numpy as np
from numpy.random import RandomState
from scipy.spatial import cKDTree, distance
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import json
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HeliosASConfig:
    num_explorers: int = 40
    num_refiners: int = 40
    max_iterations: int = 2000
    min_population_ratio: float = 0.25
    population_decay_exp: float = 2.0
    surrogate_k: int = 5
    surrogate_history_size: int = 2000
    neighborhood_radius: float = 0.6322
    explorer_memory_size: int = 10
    stagnation_threshold: int = 800
    levy_beta: float = 1.5379
    de_factor_f: float = 0.3791
    p_levy: float = 0.3114
    p_de: float = 0.6860
    de_cr: float = 0.5012
    multimodal_cv_threshold: float = 0.5
    diversity_restart_threshold: float = 0.04


class IndependentMechanismStates:
    def __init__(self):
        self.basin_iterations_since_last = 0
        self.basin_cumulative_progress = 0.0
        self.basin_total_calls = 0
        self.plateau_improvement_slopes = []
        self.plateau_true_stagnation_count = 0
        self.plateau_total_calls = 0
        self.scaling_consecutive_failures = 0
        self.scaling_stagnation = 0
        self.ls_stagnation_counter = 0
        self.ls_total_calls = 0
        self.ls_total_improvement = 0.0
        self.last_significant_improvement_iter = 0

    def update_on_improvement(self, improvement_amount, current_iter):
        self.scaling_consecutive_failures = 0
        self.ls_stagnation_counter = 0
        if np.isfinite(improvement_amount):
            if not np.isfinite(self.basin_cumulative_progress):
                self.basin_cumulative_progress = 0.0
            self.basin_cumulative_progress += improvement_amount
            self.plateau_improvement_slopes.append(improvement_amount)
            if improvement_amount > 1e-8:
                self.plateau_true_stagnation_count = 0
                self.last_significant_improvement_iter = current_iter
            else:
                self.plateau_true_stagnation_count += 1
        if len(self.plateau_improvement_slopes) > 12:
            self.plateau_improvement_slopes.pop(0)

    def update_on_failure(self):
        self.scaling_consecutive_failures += 1
        self.scaling_stagnation += 1
        self.ls_stagnation_counter += 1
        self.plateau_true_stagnation_count += 1
        self.plateau_improvement_slopes.append(0.0)
        if len(self.plateau_improvement_slopes) > 12:
            self.plateau_improvement_slopes.pop(0)

    def update_time_counters(self):
        self.basin_iterations_since_last += 1

    def should_trigger_basin_search(self, problem_scale=None, max_iterations=2000):
        T1 = int(0.15 * max_iterations)
        eps1 = 0.5 * (problem_scale or 1000)
        sufficient_time = self.basin_iterations_since_last > T1
        minimal_progress = self.basin_cumulative_progress < eps1
        not_overused = self.basin_total_calls < 5
        return sufficient_time and minimal_progress and not_overused

    def should_trigger_plateau_escape(self):
        if len(self.plateau_improvement_slopes) < 6:
            return False
        recent_slope = np.mean(self.plateau_improvement_slopes[-6:])
        trend_direction = np.polyfit(range(6), self.plateau_improvement_slopes[-6:], 1)[
            0
        ]
        return (
            trend_direction <= 0
            and recent_slope < 1e-9
            and self.plateau_true_stagnation_count > 30
        )

    def should_trigger_local_search(
        self, current_iter, global_best_fitness, bounds=None
    ):
        basic_trigger = self.ls_stagnation_counter > 150
        if bounds is not None:
            fitness_scale = np.mean(np.abs(bounds.flatten())) * 10
        else:
            fitness_scale = 1000
        precision_trigger = (
            len(self.plateau_improvement_slopes) >= 5
            and np.mean(self.plateau_improvement_slopes[-5:]) < 1e-8
            and global_best_fitness < fitness_scale
        )
        valley_trigger = (
            current_iter - self.last_significant_improvement_iter > 200
            and global_best_fitness < fitness_scale * 10
        )
        return basic_trigger or precision_trigger or valley_trigger

    def get_scaling_multiplier(self):
        return min(20.0, 2.0 ** (self.scaling_consecutive_failures / 10))

    def on_basin_search_success(self):
        self.basin_iterations_since_last = 0
        self.basin_cumulative_progress = 0.0
        self.basin_total_calls += 1

    def on_plateau_escape_success(self):
        self.plateau_true_stagnation_count = 0
        self.plateau_total_calls += 1

    def on_local_search_success(self, improvement):
        self.ls_total_calls += 1
        self.ls_total_improvement += improvement


class HeliosAS:
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        config: HeliosASConfig,
        seed: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.config = config
        self._rng = RandomState(seed)
        self.bounds = np.array(bounds, dtype=np.float64)
        self.dimensions = self.bounds.shape[0]
        self.lower_bounds, self.upper_bounds = self.bounds[:, 0], self.bounds[:, 1]
        self.range_width = self.upper_bounds - self.lower_bounds
        self.problem_scale = np.linalg.norm(self.range_width)

        assert self.dimensions > 0, "Problem must have at least 1 dimension"
        assert np.all(self.range_width > 0), "All bounds must have positive width"

        self.num_explorers_initial, self.num_refiners_initial = (
            config.num_explorers,
            config.num_refiners,
        )
        self.num_explorers_min = max(
            2, int(config.num_explorers * config.min_population_ratio)
        )
        self.num_refiners_min = max(
            2, int(config.num_refiners * config.min_population_ratio)
        )
        self.current_num_explorers, self.current_num_refiners = (
            config.num_explorers,
            config.num_refiners,
        )
        self.max_iterations = config.max_iterations
        self.neighborhood_radius = config.neighborhood_radius * self.problem_scale

        self.explorers = self._rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            (config.num_explorers, self.dimensions),
        )
        self.explorer_fitness = np.full(config.num_explorers, np.inf)
        self.refiners = self._rng.uniform(
            self.lower_bounds, self.upper_bounds, (config.num_refiners, self.dimensions)
        )
        self.refiner_fitness = np.full(config.num_refiners, np.inf)

        if config.explorer_memory_size > 0:
            self.explorer_memory = np.zeros(
                (config.num_explorers, config.explorer_memory_size, self.dimensions)
            )
            self.memory_indices = np.zeros(config.num_explorers, dtype=int)

        self.history_positions = np.zeros(
            (config.surrogate_history_size, self.dimensions)
        )
        self.history_fitness = np.full(config.surrogate_history_size, np.inf)
        self.history_count = 0
        self.history_tree = None

        self.evaluation_count = 0
        self.current_iter = 0
        self.global_best_fitness = np.inf
        self.global_best_position = None
        self.initial_best_fitness = np.inf
        self.best_explorer_fitness = np.inf
        self.best_explorer_position = None
        self.recent_fitness_history = []
        self.initial_diversity = None
        self.last_restart_iteration = 0
        self.center_escape_attempts = 0
        self.exploration_scale = 0.01
        self.exploitation_scale = 0.01

        self.mechanism_states = IndependentMechanismStates()
        self._initial_evaluation()

    def _safe_batch_evaluate(self, positions: np.ndarray) -> np.ndarray:
        if positions.shape[0] == 0:
            return np.array([])
        self.evaluation_count += positions.shape[0]
        try:
            results = self.objective_func(positions)
            if np.isscalar(results):
                results = np.array([results])
            return np.nan_to_num(results, nan=np.inf, posinf=np.inf)
        except (TypeError, ValueError):
            results = []
            for pos in positions:
                try:
                    result = self.objective_func(pos.reshape(1, -1))
                    results.append(result[0] if hasattr(result, "__len__") else result)
                except:
                    results.append(np.inf)
            return np.nan_to_num(np.array(results), nan=np.inf, posinf=np.inf)

    def _update_history(self, positions: np.ndarray, fitnesses: np.ndarray):
        if positions.shape[0] == 0:
            return

        num_new = positions.shape[0]

        if num_new >= self.config.surrogate_history_size:
            recent_positions = positions[-self.config.surrogate_history_size :]
            recent_fitness = fitnesses[-self.config.surrogate_history_size :]
            self.history_positions[:] = recent_positions
            self.history_fitness[:] = recent_fitness
            self.history_count += num_new
            return

        start_idx = self.history_count % self.config.surrogate_history_size

        if start_idx + num_new <= self.config.surrogate_history_size:
            self.history_positions[start_idx : start_idx + num_new] = positions
            self.history_fitness[start_idx : start_idx + num_new] = fitnesses
        else:
            part1_size = self.config.surrogate_history_size - start_idx
            part2_size = num_new - part1_size

            self.history_positions[start_idx:] = positions[:part1_size]
            self.history_fitness[start_idx:] = fitnesses[:part1_size]

            self.history_positions[:part2_size] = positions[part1_size:]
            self.history_fitness[:part2_size] = fitnesses[part1_size:]

        self.history_count += num_new

        current_buffer_size = min(
            self.history_count, self.config.surrogate_history_size
        )
        if (
            current_buffer_size > self.config.surrogate_k
            and self.history_count % 50 == 0
        ):
            self.history_tree = cKDTree(self.history_positions[:current_buffer_size])

    def _detect_multimodality(self) -> bool:
        if len(self.recent_fitness_history) < 30:
            return False
        fitness_array = np.array(self.recent_fitness_history[-30:])
        fitness_std = np.std(fitness_array)
        fitness_mean = np.mean(np.abs(fitness_array))
        if fitness_mean < 1e-10:
            return False
        coefficient_of_variation = fitness_std / fitness_mean
        return coefficient_of_variation > self.config.multimodal_cv_threshold

    def _update_multi_scale_parameters(self):
        recent_improvements = (
            self.mechanism_states.plateau_improvement_slopes[-10:]
            if len(self.mechanism_states.plateau_improvement_slopes) >= 10
            else []
        )
        improvement_rate = np.mean(recent_improvements) if recent_improvements else 0.0

        landscape_roughness = 0.0
        if len(self.recent_fitness_history) >= 20:
            recent_fitness = np.array(self.recent_fitness_history[-20:])
            fitness_std = np.std(recent_fitness)
            fitness_mean = np.mean(np.abs(recent_fitness))
            landscape_roughness = fitness_std / (fitness_mean + 1e-12)

        max_exploration_scale = 0.2 * self.problem_scale / 0.01
        max_exploitation_scale = 0.05 * self.problem_scale / 0.01
        scaling_multiplier = self.mechanism_states.get_scaling_multiplier()

        if improvement_rate > 1e-8:
            exploration_multiplier = min(
                5.0, 1.0 + self.mechanism_states.ls_stagnation_counter / 100
            )
            exploitation_multiplier = max(0.1, 1.0 - improvement_rate * 1000)
        elif landscape_roughness > 0.5:
            exploration_multiplier = min(max_exploration_scale, scaling_multiplier)
            exploitation_multiplier = min(
                2.0, 1.0 + self.mechanism_states.scaling_consecutive_failures / 100
            )
        else:
            exploration_multiplier = min(
                max_exploration_scale / 2,
                1.0 + self.mechanism_states.scaling_consecutive_failures / 50,
            )
            exploitation_multiplier = max(
                0.5, 1.0 - self.mechanism_states.ls_stagnation_counter / 1000
            )

        self.exploration_scale = 0.01 * exploration_multiplier
        self.exploitation_scale = 0.01 * exploitation_multiplier

    def _unbiased_plateau_escape(self) -> bool:
        if (
            not self.mechanism_states.should_trigger_plateau_escape()
            or self.global_best_position is None
        ):
            return False

        step_sizes = (
            np.array([0.1, 0.3, 0.5])
            * self.problem_scale
            / np.linalg.norm(self.range_width)
        )
        all_candidates = []

        for _ in range(3):
            direction = self._rng.normal(0, 1, self.dimensions)
            direction /= np.linalg.norm(direction)
            for step_size in step_sizes:
                candidate = self.global_best_position + step_size * direction
                candidate = np.clip(candidate, self.lower_bounds, self.upper_bounds)
                all_candidates.append(candidate)

        for dim in range(min(self.dimensions, 4)):
            coord_direction = np.zeros(self.dimensions)
            coord_direction[dim] = 1.0
            for step_size in step_sizes:
                for sign in [-1, 1]:
                    candidate = (
                        self.global_best_position + sign * step_size * coord_direction
                    )
                    candidate = np.clip(candidate, self.lower_bounds, self.upper_bounds)
                    all_candidates.append(candidate)

        position_distance = np.linalg.norm(self.global_best_position)
        if (
            position_distance < 0.3 * self.problem_scale
            and self.center_escape_attempts < 3
        ):
            center_direction = -self.global_best_position / max(
                position_distance, 1e-10
            )
            for step_size in step_sizes:
                candidate = self.global_best_position + step_size * center_direction
                candidate = np.clip(candidate, self.lower_bounds, self.upper_bounds)
                all_candidates.append(candidate)
            self.center_escape_attempts += 1

        if not all_candidates:
            return False

        all_candidates = np.array(all_candidates)
        escape_fitness = self._safe_batch_evaluate(all_candidates)
        best_escape_idx = np.argmin(escape_fitness)

        if escape_fitness[best_escape_idx] < self.global_best_fitness:
            improvement = self.global_best_fitness - escape_fitness[best_escape_idx]
            self.global_best_fitness = escape_fitness[best_escape_idx]
            self.global_best_position = all_candidates[best_escape_idx].copy()
            self.mechanism_states.update_on_improvement(improvement, self.current_iter)
            self.mechanism_states.on_plateau_escape_success()
            return True

        return False

    def _basin_search_mode(self) -> bool:
        if (
            not self.mechanism_states.should_trigger_basin_search(
                self.problem_scale, self.max_iterations
            )
            or self.global_best_position is None
        ):
            return False

        large_jump_candidates = []
        scales = (
            np.array([0.2, 0.4, 0.6, 0.8, 1.0])
            * self.problem_scale
            / np.linalg.norm(self.range_width)
        )
        attempts_per_scale = 3

        for scale in scales:
            for _ in range(attempts_per_scale):
                direction = self._rng.normal(0, 1, self.dimensions)
                direction /= np.linalg.norm(direction)
                candidate = self.global_best_position + scale * direction
                candidate = np.clip(candidate, self.lower_bounds, self.upper_bounds)
                large_jump_candidates.append(candidate)

        if not large_jump_candidates:
            self.mechanism_states.basin_total_calls += 1
            return False

        large_jump_candidates = np.array(large_jump_candidates)
        jump_fitness = self._safe_batch_evaluate(large_jump_candidates)
        best_jump_idx = np.argmin(jump_fitness)
        improvement_threshold = self.global_best_fitness * 0.9

        if jump_fitness[best_jump_idx] < improvement_threshold:
            improvement = self.global_best_fitness - jump_fitness[best_jump_idx]
            self.global_best_fitness = jump_fitness[best_jump_idx]
            self.global_best_position = large_jump_candidates[best_jump_idx].copy()
            self.mechanism_states.update_on_improvement(improvement, self.current_iter)
            self.mechanism_states.on_basin_search_success()
            return True

        self.mechanism_states.basin_total_calls += 1
        return False

    def _apply_intelligent_local_search(self):
        if not self.mechanism_states.should_trigger_local_search(
            self.current_iter, self.global_best_fitness, self.bounds
        ):
            return

        if self.evaluation_count >= 0.8 * self.max_iterations * 80:
            return

        try:
            distance_from_origin = np.linalg.norm(self.global_best_position)
            is_near_origin = distance_from_origin < 0.1 * self.problem_scale

            if is_near_origin:
                max_iter = min(500, max(50, 5000 // self.dimensions))
                options = {"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-15}
            else:
                max_iter = min(300, max(30, 3000 // self.dimensions))
                options = {"maxiter": max_iter}

            result = minimize(
                lambda x: (
                    self.objective_func(x.reshape(1, -1))[0]
                    if x.ndim == 1
                    else self.objective_func(x)
                ),
                self.global_best_position,
                method="L-BFGS-B",
                bounds=[(lb, ub) for lb, ub in self.bounds],
                options=options,
            )

            if result.success and result.fun < self.global_best_fitness:
                improvement = self.global_best_fitness - result.fun
                self.global_best_position = result.x.copy()
                self.global_best_fitness = result.fun
                self.evaluation_count += result.nfev
                self.mechanism_states.update_on_improvement(
                    improvement, self.current_iter
                )
                self.mechanism_states.on_local_search_success(improvement)
            else:
                self.evaluation_count += getattr(result, "nfev", 50)
        except Exception:
            pass

    def _smart_surrogate_filter(
        self, candidates: np.ndarray, parent_fitness: np.ndarray
    ) -> np.ndarray:
        if self._detect_multimodality():
            return np.arange(len(candidates))

        if self.history_tree is None or len(candidates) == 0:
            return np.arange(len(candidates))

        try:
            distances, indices = self.history_tree.query(
                candidates, k=self.config.surrogate_k
            )
            distances = np.maximum(distances, 1e-9)
            weights = 1.0 / distances
            weights /= np.sum(weights, axis=1, keepdims=True)

            current_buffer_size = min(
                self.history_count, self.config.surrogate_history_size
            )
            valid_indices = np.clip(indices, 0, current_buffer_size - 1)
            neighbor_fitnesses = self.history_fitness[valid_indices]
            predicted_fitness = np.sum(weights * neighbor_fitnesses, axis=1)
            uncertainty_bonus = np.std(neighbor_fitnesses, axis=1) * 0.1
            improvement_threshold = parent_fitness * 0.95

            return np.where(
                predicted_fitness < (improvement_threshold + uncertainty_bonus)
            )[0]
        except Exception:
            return np.arange(len(candidates))

    def _update_population_dynamics(self):
        recent_improvements = (
            self.mechanism_states.plateau_improvement_slopes[-5:]
            if len(self.mechanism_states.plateau_improvement_slopes) >= 5
            else []
        )

        if recent_improvements:
            recent_progress = np.mean(recent_improvements)
            if recent_progress > 1e-8:
                ratio = self.current_iter / self.max_iterations
                decay = (1 - ratio) ** self.config.population_decay_exp
            else:
                ratio = self.current_iter / self.max_iterations
                decay = max(0.6, (1 - ratio) ** (self.config.population_decay_exp / 2))
        else:
            ratio = self.current_iter / self.max_iterations
            decay = max(0.7, (1 - ratio) ** (self.config.population_decay_exp / 3))

        new_explorers = self.num_explorers_min + int(
            (self.num_explorers_initial - self.num_explorers_min) * decay
        )
        new_refiners = self.num_refiners_min + int(
            (self.num_refiners_initial - self.num_refiners_min) * decay
        )

        self.current_num_explorers = new_explorers
        self.current_num_refiners = new_refiners

    def _check_and_restart_population(self):
        if (
            self.current_iter % 100 != 0
            or self.current_iter - self.last_restart_iteration < 150
        ):
            return

        active_population = np.vstack(
            [
                self.explorers[: self.current_num_explorers],
                self.refiners[: self.current_num_refiners],
            ]
        )

        if len(active_population) <= 1:
            return

        try:
            pairwise_distances = distance.pdist(active_population)
            current_diversity = np.mean(pairwise_distances)

            if self.initial_diversity is None:
                self.initial_diversity = current_diversity
                return

            diversity_ratio = current_diversity / (self.initial_diversity + 1e-10)
            should_restart = (
                diversity_ratio < 0.15
                and self.mechanism_states.ls_stagnation_counter > 50
            )

            if should_restart:
                restart_fraction = 0.3
                total_active = self.current_num_explorers + self.current_num_refiners
                num_restart = int(total_active * restart_fraction)

                restart_positions = self._rng.uniform(
                    self.lower_bounds, self.upper_bounds, (num_restart, self.dimensions)
                )

                explorer_restart = min(num_restart // 2, self.current_num_explorers)
                refiner_restart = num_restart - explorer_restart

                if explorer_restart > 0:
                    restart_indices = self._rng.choice(
                        self.current_num_explorers, explorer_restart, replace=False
                    )
                    self.explorers[restart_indices] = restart_positions[
                        :explorer_restart
                    ]
                    self.explorer_fitness[restart_indices] = np.inf

                if refiner_restart > 0:
                    restart_indices = self._rng.choice(
                        self.current_num_refiners, refiner_restart, replace=False
                    )
                    self.refiners[restart_indices] = restart_positions[
                        explorer_restart : explorer_restart + refiner_restart
                    ]
                    self.refiner_fitness[restart_indices] = np.inf

                self.mechanism_states.plateau_true_stagnation_count = 0
                self.last_restart_iteration = self.current_iter
        except Exception:
            pass

    def _initial_evaluation(self):
        if self.num_explorers_initial > 0:
            self.explorer_fitness[:] = self._safe_batch_evaluate(self.explorers)
        if self.num_refiners_initial > 0:
            self.refiner_fitness[:] = self._safe_batch_evaluate(self.refiners)

        all_positions = np.vstack((self.explorers, self.refiners))
        all_fitness = np.hstack((self.explorer_fitness, self.refiner_fitness))
        self._update_history(all_positions, all_fitness)
        self._update_bests()
        self.initial_best_fitness = self.global_best_fitness

    def _update_bests(self):
        combined_pop = np.vstack(
            (
                self.explorers[: self.current_num_explorers],
                self.refiners[: self.current_num_refiners],
            )
        )
        combined_fit = np.hstack(
            (
                self.explorer_fitness[: self.current_num_explorers],
                self.refiner_fitness[: self.current_num_refiners],
            )
        )

        if combined_fit.size == 0:
            return

        min_idx_global = np.argmin(combined_fit)
        current_best = combined_fit[min_idx_global]

        if current_best < self.global_best_fitness:
            improvement = self.global_best_fitness - current_best
            if improvement > 1e-12:
                self.mechanism_states.update_on_improvement(
                    improvement, self.current_iter
                )
            self.global_best_fitness = current_best
            self.global_best_position = combined_pop[min_idx_global].copy()
        else:
            self.mechanism_states.update_on_failure()

        self.mechanism_states.update_time_counters()
        self.recent_fitness_history.append(current_best)
        if len(self.recent_fitness_history) > 100:
            self.recent_fitness_history.pop(0)

        if self.current_num_explorers > 1:
            active_explorers_pop = self.explorers[: self.current_num_explorers]
            active_explorers_fit = self.explorer_fitness[: self.current_num_explorers]

            if len(active_explorers_pop) > 1:
                distances = np.linalg.norm(
                    active_explorers_pop[:, np.newaxis, :]
                    - active_explorers_pop[np.newaxis, :, :],
                    axis=2,
                )
                is_isolated = np.ones(self.current_num_explorers, dtype=bool)

                for i in range(self.current_num_explorers):
                    neighbors = (distances[i] < self.neighborhood_radius) & (
                        np.arange(self.current_num_explorers) != i
                    )
                    if np.any(neighbors) and np.any(
                        active_explorers_fit[neighbors] < active_explorers_fit[i]
                    ):
                        is_isolated[i] = False

                isolated_indices = np.where(is_isolated)[0]
                if isolated_indices.size > 0:
                    best_iso_idx = isolated_indices[
                        np.argmin(active_explorers_fit[isolated_indices])
                    ]
                    if active_explorers_fit[best_iso_idx] < self.best_explorer_fitness:
                        self.best_explorer_fitness = active_explorers_fit[best_iso_idx]
                        self.best_explorer_position = active_explorers_pop[
                            best_iso_idx
                        ].copy()

    def _clip_to_bounds(self, positions: np.ndarray) -> np.ndarray:
        return np.clip(positions, self.lower_bounds, self.upper_bounds)

    def _update_explorer_memory(self, indices, positions, success_mask):
        if self.config.explorer_memory_size <= 0:
            return

        for i, (pos, is_success) in enumerate(zip(positions, success_mask)):
            if is_success:
                global_idx = indices[i]
                mem_idx = self.memory_indices[global_idx]
                self.explorer_memory[global_idx, mem_idx] = pos
                self.memory_indices[global_idx] = (
                    mem_idx + 1
                ) % self.config.explorer_memory_size

    def _get_active_indices(self, fitness_array, current_num, initial_num):
        if current_num >= len(fitness_array):
            return np.arange(len(fitness_array))
        preliminary_indices = np.argpartition(fitness_array, current_num - 1)[
            :current_num
        ]
        return preliminary_indices[np.argsort(fitness_array[preliminary_indices])]

    def _run_explorer_phase(self):
        if self.current_num_explorers == 0:
            return

        active_indices = self._get_active_indices(
            self.explorer_fitness,
            self.current_num_explorers,
            self.num_explorers_initial,
        )
        active_explorers = self.explorers[active_indices]
        choices = self._rng.rand(self.current_num_explorers)
        candidates = np.copy(active_explorers)

        exploration_agents = self.current_num_explorers // 2
        exploitation_agents = self.current_num_explorers - exploration_agents

        p_mem_revisit, p_mem_explore, p_to_best = 0.3, 0.4, 0.05
        t1 = self.config.p_levy
        t2 = t1 + p_mem_revisit
        t3 = t2 + p_mem_explore
        t4 = t3 + p_to_best

        levy_mask = choices < t1
        if np.any(levy_mask):
            n = np.sum(levy_mask)
            beta = self.config.levy_beta
            num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
            sigma = (num / den) ** (1 / beta)

            u = self._rng.normal(0, sigma, (n, self.dimensions))
            v = self._rng.normal(0, 1, (n, self.dimensions))
            scales = np.where(
                np.arange(n) < exploration_agents,
                self.exploration_scale,
                self.exploitation_scale,
            )
            levy_steps = (
                scales[:, np.newaxis]
                * self.range_width
                * (u / (np.abs(v) ** (1 / beta)))
            )
            candidates[levy_mask] = active_explorers[levy_mask] + levy_steps

        if self.config.explorer_memory_size > 0:
            mem_mask = (choices >= t1) & (choices < t3)
            if np.any(mem_mask):
                mem_indices_local = np.where(mem_mask)[0]
                mem_indices_global = active_indices[mem_indices_local]
                mem_sample_indices = self._rng.randint(
                    0, self.config.explorer_memory_size, len(mem_indices_local)
                )
                mem_positions = self.explorer_memory[
                    mem_indices_global, mem_sample_indices
                ]

                revisit_mask = ((choices >= t1) & (choices < t2))[mem_mask]
                if np.any(revisit_mask):
                    noise_scale = 0.1 * self.exploitation_scale
                    candidates[mem_indices_local[revisit_mask]] = mem_positions[
                        revisit_mask
                    ] + self._rng.normal(
                        0,
                        noise_scale * self.range_width,
                        (np.sum(revisit_mask), self.dimensions),
                    )

                explore_mask = ((choices >= t2) & (choices < t3))[mem_mask]
                if np.any(explore_mask):
                    noise_scale = 0.3 * self.exploration_scale
                    candidates[mem_indices_local[explore_mask]] = mem_positions[
                        explore_mask
                    ] + self._rng.normal(
                        0,
                        noise_scale * self.range_width,
                        (np.sum(explore_mask), self.dimensions),
                    )

        global_move_mask = (choices >= t3) & (choices < t4)
        if np.any(global_move_mask) and self.global_best_position is not None:
            step_sizes = self._rng.uniform(0.1, 0.5, (np.sum(global_move_mask), 1))
            candidates[global_move_mask] = active_explorers[
                global_move_mask
            ] + step_sizes * (
                self.global_best_position - active_explorers[global_move_mask]
            )

        random_mask = choices >= t4
        if np.any(random_mask):
            candidates[random_mask] = self._rng.uniform(
                self.lower_bounds,
                self.upper_bounds,
                (np.sum(random_mask), self.dimensions),
            )

        candidates = self._clip_to_bounds(candidates)
        eval_indices_local = self._smart_surrogate_filter(
            candidates, self.explorer_fitness[active_indices]
        )

        if eval_indices_local.size > 0:
            new_fitness = self._safe_batch_evaluate(candidates[eval_indices_local])
            self._update_history(candidates[eval_indices_local], new_fitness)

            update_mask_local = (
                new_fitness < self.explorer_fitness[active_indices[eval_indices_local]]
            )
            if np.any(update_mask_local):
                update_indices_local = eval_indices_local[update_mask_local]
                update_indices_global = active_indices[update_indices_local]

                self.explorers[update_indices_global] = candidates[update_indices_local]
                self.explorer_fitness[update_indices_global] = new_fitness[
                    update_mask_local
                ]

                success_mask = np.zeros(len(eval_indices_local), dtype=bool)
                success_mask[update_mask_local] = True
                self._update_explorer_memory(
                    update_indices_global,
                    candidates[update_indices_local],
                    success_mask[update_mask_local],
                )

    def _run_refiner_phase(self):
        if self.current_num_refiners == 0:
            return

        active_indices = self._get_active_indices(
            self.refiner_fitness, self.current_num_refiners, self.num_refiners_initial
        )
        active_refiners = self.refiners[active_indices]
        choices = self._rng.rand(self.current_num_refiners)
        candidates = np.copy(active_refiners)

        de_mask = choices < self.config.p_de
        if np.any(de_mask):
            n = np.sum(de_mask)
            combined_pop = np.vstack(
                (
                    self.explorers[: self.current_num_explorers],
                    self.refiners[: self.current_num_refiners],
                )
            )

            if combined_pop.shape[0] >= 3:
                p1 = self._rng.randint(0, combined_pop.shape[0], n)
                p2 = self._rng.randint(0, combined_pop.shape[0], n)
                p3 = self._rng.randint(0, combined_pop.shape[0], n)

                for i in range(n):
                    while p2[i] == p1[i]:
                        p2[i] = self._rng.randint(0, combined_pop.shape[0])
                    while p3[i] == p1[i] or p3[i] == p2[i]:
                        p3[i] = self._rng.randint(0, combined_pop.shape[0])

                mutant = self._clip_to_bounds(
                    combined_pop[p1]
                    + self.config.de_factor_f * (combined_pop[p2] - combined_pop[p3])
                )

                crossover_mask = self._rng.rand(n, self.dimensions) < self.config.de_cr
                candidates[de_mask] = np.where(
                    crossover_mask, mutant, active_refiners[de_mask]
                )

        hunt_mask = choices >= self.config.p_de
        if np.any(hunt_mask) and self.global_best_position is not None:
            target = (
                self.best_explorer_position
                if self.best_explorer_position is not None
                else self.global_best_position
            )
            step_sizes = self._rng.uniform(0.8, 1.5, (np.sum(hunt_mask), 1))
            candidates[hunt_mask] = active_refiners[hunt_mask] + step_sizes * (
                target - active_refiners[hunt_mask]
            )

        candidates = self._clip_to_bounds(candidates)
        eval_indices_local = self._smart_surrogate_filter(
            candidates, self.refiner_fitness[active_indices]
        )

        if eval_indices_local.size > 0:
            new_fitness = self._safe_batch_evaluate(candidates[eval_indices_local])
            self._update_history(candidates[eval_indices_local], new_fitness)

            update_mask_local = (
                new_fitness < self.refiner_fitness[active_indices[eval_indices_local]]
            )
            if np.any(update_mask_local):
                update_indices_local = eval_indices_local[update_mask_local]
                update_indices_global = active_indices[update_indices_local]

                self.refiners[update_indices_global] = candidates[update_indices_local]
                self.refiner_fitness[update_indices_global] = new_fitness[
                    update_mask_local
                ]

    def get_performance_summary(self):
        improvement_ratio = self.initial_best_fitness / max(
            self.global_best_fitness, 1e-16
        )
        return {
            "final_fitness": self.global_best_fitness,
            "total_evaluations": self.evaluation_count,
            "improvement_ratio": improvement_ratio,
            "local_search_calls": self.mechanism_states.ls_total_calls,
            "plateau_escapes": self.mechanism_states.plateau_total_calls,
            "basin_searches": self.mechanism_states.basin_total_calls,
            "max_consecutive_failures": self.mechanism_states.scaling_consecutive_failures,
            "exploration_scale": self.exploration_scale,
            "exploitation_scale": self.exploitation_scale,
        }

    def optimize(self) -> Tuple[Optional[np.ndarray], float]:
        if self.explorers.size == 0 and self.refiners.size == 0:
            return None, np.inf

        start_time = time.time()

        for iteration in range(self.max_iterations):
            self.current_iter = iteration

            self._update_multi_scale_parameters()
            self._update_population_dynamics()
            self._run_explorer_phase()
            self._run_refiner_phase()
            self._update_bests()

            if iteration % 50 == 0:
                self._unbiased_plateau_escape()
            if iteration % 100 == 0:
                self._apply_intelligent_local_search()
            if iteration % 150 == 0 and iteration > 200:
                self._basin_search_mode()

            self._check_and_restart_population()

            if (
                self.mechanism_states.ls_stagnation_counter
                > self.config.stagnation_threshold
            ):
                break

        elapsed_time = time.time() - start_time
        return self.global_best_position, self.global_best_fitness


class AblationHeliosAS(HeliosAS):
    def __init__(
        self, objective_func, bounds, config, seed=None, ablation_type="baseline"
    ):
        self.ablation_type = ablation_type
        super().__init__(objective_func, bounds, config, seed)

    def _apply_intelligent_local_search(self):
        if self.ablation_type == "no_local_search":
            return
        super()._apply_intelligent_local_search()

    def _smart_surrogate_filter(self, candidates, parent_fitness):
        if self.ablation_type == "no_surrogate":
            return np.arange(len(candidates))
        return super()._smart_surrogate_filter(candidates, parent_fitness)

    def _unbiased_plateau_escape(self):
        if self.ablation_type == "no_escape_mechanisms":
            return False
        return super()._unbiased_plateau_escape()

    def _basin_search_mode(self):
        if self.ablation_type == "no_escape_mechanisms":
            return False
        return super()._basin_search_mode()

    def _update_population_dynamics(self):
        if self.ablation_type == "no_population_decay":
            self.current_num_explorers = self.num_explorers_initial
            self.current_num_refiners = self.num_refiners_initial
            return
        super()._update_population_dynamics()

    def _check_and_restart_population(self):
        if self.ablation_type == "no_diversity_restart":
            return
        super()._check_and_restart_population()

    def _run_refiner_phase(self):
        if self.current_num_refiners == 0:
            return

        active_indices = self._get_active_indices(
            self.refiner_fitness, self.current_num_refiners, self.num_refiners_initial
        )
        active_refiners = self.refiners[active_indices]
        choices = self._rng.rand(self.current_num_refiners)
        candidates = np.copy(active_refiners)

        de_mask = choices < self.config.p_de

        if self.ablation_type == "no_de_operator":
            if np.any(de_mask) and self.global_best_position is not None:
                target = self.global_best_position
                step_sizes = self._rng.uniform(0.3, 0.9, (np.sum(de_mask), 1))
                candidates[de_mask] = active_refiners[de_mask] + step_sizes * (
                    target - active_refiners[de_mask]
                )
        else:
            if np.any(de_mask):
                n = np.sum(de_mask)
                combined_pop = np.vstack(
                    (
                        self.explorers[: self.current_num_explorers],
                        self.refiners[: self.current_num_refiners],
                    )
                )

                if combined_pop.shape[0] >= 3:
                    p1 = self._rng.randint(0, combined_pop.shape[0], n)
                    p2 = self._rng.randint(0, combined_pop.shape[0], n)
                    p3 = self._rng.randint(0, combined_pop.shape[0], n)

                    for i in range(n):
                        while p2[i] == p1[i]:
                            p2[i] = self._rng.randint(0, combined_pop.shape[0])
                        while p3[i] == p1[i] or p3[i] == p2[i]:
                            p3[i] = self._rng.randint(0, combined_pop.shape[0])

                    mutant = self._clip_to_bounds(
                        combined_pop[p1]
                        + self.config.de_factor_f
                        * (combined_pop[p2] - combined_pop[p3])
                    )

                    crossover_mask = (
                        self._rng.rand(n, self.dimensions) < self.config.de_cr
                    )
                    candidates[de_mask] = np.where(
                        crossover_mask, mutant, active_refiners[de_mask]
                    )

        hunt_mask = choices >= self.config.p_de
        if np.any(hunt_mask) and self.global_best_position is not None:
            target = (
                self.best_explorer_position
                if self.best_explorer_position is not None
                else self.global_best_position
            )
            step_sizes = self._rng.uniform(0.8, 1.5, (np.sum(hunt_mask), 1))
            candidates[hunt_mask] = active_refiners[hunt_mask] + step_sizes * (
                target - active_refiners[hunt_mask]
            )

        candidates = self._clip_to_bounds(candidates)
        eval_indices_local = self._smart_surrogate_filter(
            candidates, self.refiner_fitness[active_indices]
        )

        if eval_indices_local.size > 0:
            new_fitness = self._safe_batch_evaluate(candidates[eval_indices_local])
            self._update_history(candidates[eval_indices_local], new_fitness)

            update_mask_local = (
                new_fitness < self.refiner_fitness[active_indices[eval_indices_local]]
            )
            if np.any(update_mask_local):
                update_indices_local = eval_indices_local[update_mask_local]
                update_indices_global = active_indices[update_indices_local]

                self.refiners[update_indices_global] = candidates[update_indices_local]
                self.refiner_fitness[update_indices_global] = new_fitness[
                    update_mask_local
                ]


def sphere_function(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=-1)


def rosenbrock_function(x: np.ndarray) -> np.ndarray:
    return np.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (x[..., :-1] - 1) ** 2, axis=-1
    )


def rastrigin_function(x: np.ndarray) -> np.ndarray:
    return 10 * x.shape[-1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def ackley_function(x: np.ndarray) -> np.ndarray:
    dim = x.shape[-1]
    sum_sq = np.sum(x**2, axis=-1) / dim
    sum_cos = np.sum(np.cos(2.0 * np.pi * x), axis=-1) / dim
    return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20.0 + np.e


def griewank_function(x: np.ndarray) -> np.ndarray:
    sum_sq = np.sum(x**2, axis=-1) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[-1] + 1))), axis=-1)
    return sum_sq - prod_cos + 1


def print_ablation_summary(func_name, dim, ablation_name, results):
    fitnesses = [r["final_fitness"] for r in results]
    evaluations = [r["total_evaluations"] for r in results]
    runtimes = [r["runtime_seconds"] for r in results]

    print(f"\nRESULTS: {func_name} {dim}D - {ablation_name}")
    print("-" * 60)
    print(f"Best Fitness:    {min(fitnesses):.6e}")
    print(f"Mean Fitness:    {np.mean(fitnesses):.6e} ± {np.std(fitnesses):.6e}")
    print(f"Worst Fitness:   {max(fitnesses):.6e}")
    print(f"Avg Evaluations: {np.mean(evaluations):.0f}")
    print(f"Avg Runtime:     {np.mean(runtimes):.2f}s")
    print("-" * 60)


def print_function_dimension_comparison(func_name, dim, results_dict):
    print(f"\nCOMPARISON: {func_name} {dim}D")
    print("=" * 80)

    comparison_data = []
    for ablation_name, results in results_dict.items():
        fitnesses = [r["final_fitness"] for r in results]
        comparison_data.append(
            {
                "Ablation": ablation_name,
                "Best": f"{min(fitnesses):.2e}",
                "Mean": f"{np.mean(fitnesses):.2e}",
                "Std": f"{np.std(fitnesses):.2e}",
                "Success Rate": f"{sum(1 for f in fitnesses if f < 1e-6)/len(fitnesses)*100:.0f}%",
            }
        )

    comparison_data.sort(key=lambda x: float(x["Mean"]))

    headers = ["Rank", "Ablation", "Best", "Mean", "Std", "Success Rate"]
    print(
        f"{headers[0]:<4} {headers[1]:<20} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12}"
    )
    print("-" * 80)

    for i, data in enumerate(comparison_data, 1):
        rank_symbol = (
            "1st" if i == 1 else "2nd" if i == 2 else "3rd" if i == 3 else f"{i:2d}"
        )
        print(
            f"{rank_symbol:<4} {data['Ablation']:<20} {data['Best']:<12} {data['Mean']:<12} {data['Std']:<12} {data['Success Rate']:<12}"
        )
    print("=" * 80)


def print_final_summary(results_df):
    print("\nFINAL SUMMARY STATISTICS")
    print("=" * 80)

    mean_fitness_by_ablation = (
        results_df.groupby("ablation_type")["final_fitness"]
        .agg(["mean", "std", "min"])
        .round(6)
    )
    mean_fitness_by_ablation = mean_fitness_by_ablation.sort_values("mean")

    print("\nOVERALL RANKING BY MEAN FITNESS:")
    print("-" * 50)
    for i, (ablation, stats) in enumerate(mean_fitness_by_ablation.iterrows(), 1):
        symbol = (
            "1st" if i == 1 else "2nd" if i == 2 else "3rd" if i == 3 else f"{i:2d}"
        )
        print(
            f"{symbol} {ablation:<20} Mean: {stats['mean']:.2e} (±{stats['std']:.2e})"
        )

    print("\nFUNCTION-SPECIFIC BEST ABLATIONS:")
    print("-" * 50)
    for func_name in results_df["function"].unique():
        func_data = results_df[results_df["function"] == func_name]
        best_ablation = (
            func_data.groupby("ablation_type")["final_fitness"].mean().idxmin()
        )
        best_fitness = func_data.groupby("ablation_type")["final_fitness"].mean().min()
        print(f"{func_name:<12}: {best_ablation:<20} ({best_fitness:.2e})")

    print(
        "\nResults saved to: helios_as_ablation_results.csv & helios_as_ablation_results.json"
    )
    print("=" * 80)


if __name__ == "__main__":
    dimensions_to_test = [10, 30]
    num_runs = 10
    ablation_types = [
        "baseline",
        "no_local_search",
        "no_surrogate",
        "no_escape_mechanisms",
        "no_population_decay",
        "no_diversity_restart",
        "no_de_operator",
    ]

    benchmarks = {
        "Sphere": {"func": sphere_function, "bounds": [-100.0, 100.0]},
        "Rosenbrock": {"func": rosenbrock_function, "bounds": [-30.0, 30.0]},
        "Rastrigin": {"func": rastrigin_function, "bounds": [-5.12, 5.12]},
        "Ackley": {"func": ackley_function, "bounds": [-32.768, 32.768]},
        "Griewank": {"func": griewank_function, "bounds": [-600.0, 600.0]},
    }

    all_results = []
    nested_results = {}

    config = HeliosASConfig(
        num_explorers=40, num_refiners=40, max_iterations=2000, stagnation_threshold=800
    )

    total_experiments = (
        len(benchmarks) * len(dimensions_to_test) * len(ablation_types) * num_runs
    )

    with tqdm(
        total=total_experiments,
        desc="Helios-AS Ablation Study Progress",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        colour="white",
    ) as main_pbar:

        for func_name, details in benchmarks.items():
            nested_results[func_name] = {}

            for dim in dimensions_to_test:
                nested_results[func_name][str(dim)] = {}
                bounds_D = np.array(
                    [[details["bounds"][0], details["bounds"][1]]] * dim
                )

                desc = f"{func_name} {dim}D"
                ablation_pbar = tqdm(
                    ablation_types, desc=desc, leave=False, colour="blue"
                )

                for ablation_name in ablation_pbar:
                    nested_results[func_name][str(dim)][ablation_name] = []
                    ablation_pbar.set_postfix_str(f"Testing: {ablation_name}")

                    run_results = []
                    run_pbar = tqdm(
                        range(num_runs),
                        desc=f"  {ablation_name}",
                        leave=False,
                        colour="yellow",
                    )

                    for i in run_pbar:
                        run_pbar.set_postfix_str(f"Run {i+1}")

                        seed = 2024 + i
                        start_time = time.time()

                        optimizer = AblationHeliosAS(
                            objective_func=details["func"],
                            bounds=bounds_D,
                            config=config,
                            seed=seed,
                            ablation_type=ablation_name,
                        )

                        best_pos, best_fit = optimizer.optimize()
                        performance = optimizer.get_performance_summary()
                        run_time = time.time() - start_time

                        result = {
                            "ablation_type": ablation_name,
                            "function": func_name,
                            "dimensions": dim,
                            "run_id": i + 1,
                            "final_fitness": best_fit,
                            "total_evaluations": performance["total_evaluations"],
                            "improvement_ratio": performance["improvement_ratio"],
                            "local_search_calls": performance["local_search_calls"],
                            "plateau_escapes": performance["plateau_escapes"],
                            "basin_searches": performance["basin_searches"],
                            "final_exploration_scale": performance["exploration_scale"],
                            "final_exploitation_scale": performance[
                                "exploitation_scale"
                            ],
                            "runtime_seconds": run_time,
                        }

                        all_results.append(result)
                        nested_results[func_name][str(dim)][ablation_name].append(
                            result
                        )
                        run_results.append(result)

                        main_pbar.update(1)
                        run_pbar.set_postfix_str(f"Fitness: {best_fit:.2e}")

                    run_pbar.close()
                    print_ablation_summary(func_name, dim, ablation_name, run_results)

                ablation_pbar.close()
                print_function_dimension_comparison(
                    func_name, dim, nested_results[func_name][str(dim)]
                )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("helios_as_ablation_results.csv", index=False)

    with open("helios_as_ablation_results.json", "w") as f:
        json.dump(nested_results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("HELIOS-AS ABLATION STUDY COMPLETE!")
    print("=" * 80)
    print_final_summary(results_df)
