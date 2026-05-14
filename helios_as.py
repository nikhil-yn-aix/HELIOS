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
        start_idx = self.history_count % self.config.surrogate_history_size

        end_idx = start_idx + num_new
        if end_idx <= self.config.surrogate_history_size:
            self.history_positions[start_idx:end_idx] = positions
            self.history_fitness[start_idx:end_idx] = fitnesses
        else:
            num_part1 = self.config.surrogate_history_size - start_idx
            self.history_positions[start_idx:] = positions[:num_part1]
            self.history_fitness[start_idx:] = fitnesses[:num_part1]
            num_part2 = num_new - num_part1
            self.history_positions[:num_part2] = positions[num_part1:]
            self.history_fitness[:num_part2] = fitnesses[num_part1:]

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
        pbar = tqdm(range(self.max_iterations), desc="Helios-AS", unit="iter")

        for iteration in pbar:
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

            pbar.set_postfix(
                {
                    "Best": f"{self.global_best_fitness:.4e}",
                    "Evals": self.evaluation_count,
                    "SF": self.mechanism_states.scaling_consecutive_failures,
                    "LS": self.mechanism_states.ls_total_calls,
                    "PE": self.mechanism_states.plateau_total_calls,
                    "BS": self.mechanism_states.basin_total_calls,
                    "ExS": f"{self.exploration_scale:.3f}",
                    "ExlS": f"{self.exploitation_scale:.3f}",
                }
            )

            if (
                self.mechanism_states.ls_stagnation_counter
                > self.config.stagnation_threshold
            ):
                break

        pbar.close()

        elapsed_time = time.time() - start_time
        return self.global_best_position, self.global_best_fitness


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


def levy_function(x: np.ndarray) -> np.ndarray:
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[..., 0]) ** 2
    term3 = (w[..., -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[..., -1]) ** 2)

    if x.ndim == 1:
        if len(x) == 1:
            return term1 + term3
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    else:
        if x.shape[-1] == 1:
            return term1 + term3
        term2 = np.sum(
            (w[..., :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[..., :-1] + 1) ** 2),
            axis=-1,
        )

    return term1 + term2 + term3


def adaptive_success_criteria(achieved_fitness, target_fitness, initial_fitness):
    relative_success = achieved_fitness <= target_fitness
    near_success = achieved_fitness <= target_fitness * 100
    improvement_success = (initial_fitness / max(achieved_fitness, 1e-16)) >= 1e4

    return relative_success, near_success, improvement_success


def helios_benchmark():
    print("Helios-AS - Adaptive Serial Benchmark\n")

    DIMENSIONS = 10
    NUM_RUNS = 10
    MAX_ITERATIONS = 2000
    MASTER_SEED = 2024

    config = HeliosASConfig(
        num_explorers=40,
        num_refiners=40,
        max_iterations=MAX_ITERATIONS,
        multimodal_cv_threshold=0.5,
        diversity_restart_threshold=0.04,
        stagnation_threshold=800,
    )

    benchmarks = {
        "Sphere": {"func": sphere_function, "bounds": [-100.0, 100.0], "target": 1e-10},
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
    }

    results_summary = []
    total_start_time = time.time()

    for func_name, details in benchmarks.items():
        print(f"\n{func_name.upper()} FUNCTION (D={DIMENSIONS})")
        bounds_D = np.array([[details["bounds"][0], details["bounds"][1]]] * DIMENSIONS)

        run_results = []
        run_performances = []
        target_successes = 0
        near_successes = 0
        improvement_successes = 0

        for i in range(NUM_RUNS):
            run_seed = MASTER_SEED + i

            print(f"Run {i+1:2d}: ", end="", flush=True)

            optimizer = HeliosAS(
                objective_func=details["func"],
                bounds=bounds_D,
                config=config,
                seed=run_seed,
            )

            best_pos, best_fit = optimizer.optimize()
            performance = optimizer.get_performance_summary()

            run_results.append(best_fit)
            run_performances.append(performance)

            target_success, near_success, improvement_success = (
                adaptive_success_criteria(
                    best_fit, details["target"], performance["improvement_ratio"]
                )
            )

            if target_success:
                target_successes += 1
                status = "TARGET"
            elif near_success:
                near_successes += 1
                status = "NEAR"
            elif improvement_success:
                improvement_successes += 1
                status = "IMPROVE"
            else:
                status = "FAIL"

            print(
                f"{best_fit:.2e} | Evals: {performance['total_evaluations']:4d} | "
                f"LS: {performance['local_search_calls']:2d} | "
                f"PE: {performance['plateau_escapes']:2d} | "
                f"BS: {performance['basin_searches']:2d} | {status}"
            )

        mean_fitness = np.mean(run_results)
        std_fitness = np.std(run_results)
        best_fitness = np.min(run_results)
        mean_evals = np.mean([p["total_evaluations"] for p in run_performances])
        mean_ls_calls = np.mean([p["local_search_calls"] for p in run_performances])
        mean_plateau_escapes = np.mean([p["plateau_escapes"] for p in run_performances])
        mean_basin_searches = np.mean([p["basin_searches"] for p in run_performances])

        target_success_rate = target_successes / NUM_RUNS
        near_success_rate = near_successes / NUM_RUNS
        improvement_success_rate = improvement_successes / NUM_RUNS
        total_success_rate = (
            target_successes + near_successes + improvement_successes
        ) / NUM_RUNS

        summary = {
            "Function": func_name,
            "Best": f"{best_fitness:.2e}",
            "Mean": f"{mean_fitness:.2e}",
            "Std": f"{std_fitness:.2e}",
            "Target_Success": f"{target_success_rate:.0%}",
            "Near_Success": f"{near_success_rate:.0%}",
            "Total_Success": f"{total_success_rate:.0%}",
            "Avg_Evals": f"{mean_evals:.0f}",
            "Avg_LS": f"{mean_ls_calls:.1f}",
            "Avg_PE": f"{mean_plateau_escapes:.1f}",
            "Avg_BS": f"{mean_basin_searches:.1f}",
        }

        results_summary.append(summary)

        print(
            f"SUMMARY: Best={best_fitness:.2e} | Target={target_success_rate:.0%} | "
            f"Near={near_success_rate:.0%} | Total={total_success_rate:.0%} | Evals={mean_evals:.0f}"
        )

    total_end_time = time.time()

    print(f"\n{'='*100}")
    print("HELIOS-AS ADAPTIVE SERIAL - PERFORMANCE REPORT")
    print(f"{'='*100}")

    results_df = pd.DataFrame(results_summary)
    print(results_df.to_string(index=False))

    overall_target_success = np.mean(
        [float(row["Target_Success"].strip("%")) / 100 for row in results_summary]
    )
    overall_total_success = np.mean(
        [float(row["Total_Success"].strip("%")) / 100 for row in results_summary]
    )

    print(f"\nOVERALL TARGET SUCCESS RATE: {overall_target_success:.1%}")
    print(f"OVERALL TOTAL SUCCESS RATE: {overall_total_success:.1%}")
    print(f"TOTAL BENCHMARK TIME: {total_end_time - total_start_time:.1f}s")

    print(f"\n{'='*100}")


def scaling_test():
    print("\nDIMENSIONAL SCALING PERFORMANCE")

    dimensions_list = [5, 10, 20, 30, 50, 100]

    for dim in dimensions_list:
        print(f"\nDimension: {dim}")
        bounds = np.array([[-10, 10]] * dim)

        config = HeliosASConfig(
            num_explorers=25 + dim,
            num_refiners=25 + dim,
            max_iterations=min(1500, 8000 // dim),
            stagnation_threshold=600,
        )

        start_time = time.time()
        optimizer = HeliosAS(sphere_function, bounds, config, seed=42)
        best_pos, best_fit = optimizer.optimize()
        end_time = time.time()

        performance = optimizer.get_performance_summary()
        evals_per_second = performance["total_evaluations"] / (end_time - start_time)

        print(
            f"  Result: {best_fit:.2e} | Time: {end_time-start_time:.2f}s | "
            f"Evals/s: {evals_per_second:.0f} | Total Evals: {performance['total_evaluations']}"
        )


if __name__ == "__main__":
    logger.info("Starting Helios-AS Adaptive Serial Benchmark")

    helios_benchmark()

    scaling_test()

    print("\nHelios-AS Adaptive Serial Complete")
    logger.info("Helios-AS benchmarking complete")
