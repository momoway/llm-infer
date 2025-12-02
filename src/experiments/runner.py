"""Experiment runner for simulations."""

import simpy
from typing import Dict, Any
from .config import SimulationConfig
from ..simulation import (
    RequestGenerator,
    LLMInferenceServer,
    constant_rate,
    step_rate,
    sinusoidal_rate,
)
from ..scheduling import get_policy
from ..metrics import MetricsCollector


class SimulationRunner:
    """Runs a single simulation with given configuration."""

    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation runner.

        Args:
            config: Simulation configuration
        """
        self.config = config

    def run(self) -> Dict[str, Any]:
        """
        Run the simulation and return metrics.

        Returns:
            Dictionary of metrics and results
        """
        # Create SimPy environment
        env = simpy.Environment()

        # Create arrival rate function
        arrival_rate_func = self._create_arrival_rate_func()

        # Create request generator
        generator = RequestGenerator(
            env=env,
            arrival_rate_func=arrival_rate_func,
            prompt_length_dist=self.config.prompt_length_dist,
            output_length_dist=self.config.output_length_dist,
            prompt_dist_params=self.config.prompt_dist_params,
            output_dist_params=self.config.output_dist_params,
            seed=self.config.random_seed,
        )

        # Create scheduling policy
        policy = get_policy(
            self.config.scheduling_policy,
            **self.config.scheduling_policy_params
        )

        # Create LLM server
        server = LLMInferenceServer(
            env=env,
            batch_size=self.config.batch_size,
            scheduling_policy=policy,
            batch_timeout=self.config.batch_timeout,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
        )

        # Start server process
        env.process(server.run())

        # Start request generation process (pass server to track queue stats)
        env.process(generator.run(server, max_requests=self.config.num_requests))

        # Run simulation until all requests are generated and processed
        # Add buffer time to ensure all requests are processed
        total_time = self.config.num_requests / self.config.arrival_rate * 2
        env.run(until=total_time)

        # Collect metrics
        metrics_collector = MetricsCollector(warmup_requests=self.config.warmup_requests)
        metrics = metrics_collector.compute_metrics(
            server.completed_requests,
            additional_stats=server.get_statistics()
        )

        # Add configuration to results
        metrics["config"] = self.config.to_dict()

        # Add starvation rate
        metrics["starvation_rate"] = metrics_collector.compute_starvation_rate(server.completed_requests)

        return metrics

    def _create_arrival_rate_func(self):
        """Create arrival rate function based on configuration."""
        pattern = self.config.arrival_pattern
        params = self.config.arrival_pattern_params

        if pattern == "constant":
            return constant_rate(self.config.arrival_rate)

        elif pattern == "step":
            step_time = params.get("step_time", 1000)
            step_rate = params.get("step_rate", self.config.arrival_rate * 2)
            return step_rate(self.config.arrival_rate, step_time, step_rate)

        elif pattern == "sinusoidal":
            amplitude = params.get("amplitude", self.config.arrival_rate * 0.8)
            period = params.get("period", 3600)
            return sinusoidal_rate(self.config.arrival_rate, amplitude, period)

        else:
            raise ValueError(f"Unknown arrival pattern: {pattern}")


class ExperimentRunner:
    """Runs multiple simulation replications for statistical rigor."""

    def __init__(self, configs: list):
        """
        Initialize experiment runner.

        Args:
            configs: List of simulation configurations
        """
        self.configs = configs

    def run_all(self, verbose: bool = True) -> list:
        """
        Run all simulation replications.

        Args:
            verbose: Whether to print progress

        Returns:
            List of metrics dictionaries from all runs
        """
        results = []

        for i, config in enumerate(self.configs):
            if verbose:
                print(f"Running replication {i+1}/{len(self.configs)}...")

            runner = SimulationRunner(config)
            metrics = runner.run()
            results.append(metrics)

            if verbose:
                print(f"  Completed: {metrics.get('total_requests', 0)} requests, "
                      f"Avg latency: {metrics.get('avg_latency', 0):.3f}s")

        return results

    def aggregate_results(self, results: list) -> Dict[str, Any]:
        """
        Aggregate results from multiple replications with confidence intervals.

        Args:
            results: List of metrics dictionaries

        Returns:
            Aggregated metrics with mean and 95% CI
        """
        import numpy as np
        from scipy import stats

        # Metrics to aggregate
        metric_keys = [
            "avg_latency", "p50_latency", "p95_latency", "p99_latency",
            "avg_queue_wait", "throughput_req_per_sec", "fairness_index",
            "starvation_rate", "max_queue_length"
        ]

        aggregated = {}

        for key in metric_keys:
            values = [r.get(key) for r in results if r.get(key) is not None]

            if not values:
                continue

            mean = np.mean(values)
            std_error = np.std(values, ddof=1) / np.sqrt(len(values))
            t_value = stats.t.ppf(0.975, len(values) - 1)  # 95% CI
            margin = t_value * std_error

            aggregated[key] = {
                "mean": mean,
                "std": np.std(values, ddof=1),
                "ci_lower": mean - margin,
                "ci_upper": mean + margin,
                "values": values,
            }

        return aggregated
