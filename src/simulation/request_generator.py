"""Request generator with various arrival patterns and distributions."""

import numpy as np
import simpy
from typing import Callable, Optional
from .request import Request


class RequestGenerator:
    """Generates requests with configurable arrival patterns and length distributions."""

    def __init__(
        self,
        env: simpy.Environment,
        arrival_rate_func: Callable[[float], float],
        prompt_length_dist: str = "lognormal",
        output_length_dist: str = "truncated_normal",
        prompt_dist_params: Optional[dict] = None,
        output_dist_params: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize request generator.

        Args:
            env: SimPy environment
            arrival_rate_func: Function lambda(t) -> rate at time t
            prompt_length_dist: Distribution type for prompt lengths
            output_length_dist: Distribution type for output lengths
            prompt_dist_params: Parameters for prompt distribution
            output_dist_params: Parameters for output distribution
            seed: Random seed for reproducibility
        """
        self.env = env
        self.arrival_rate_func = arrival_rate_func
        self.prompt_length_dist = prompt_length_dist
        self.output_length_dist = output_length_dist

        # Default distribution parameters from report
        self.prompt_dist_params = prompt_dist_params or {"mu": 4, "sigma": 1.5}
        self.output_dist_params = output_dist_params or {"mu": 100, "sigma": 30, "min": 10, "max": 500}

        self.rng = np.random.default_rng(seed)
        self.request_id_counter = 0

    def generate_prompt_length(self) -> int:
        """Generate prompt length based on configured distribution."""
        if self.prompt_length_dist == "lognormal":
            mu = self.prompt_dist_params.get("mu", 4)
            sigma = self.prompt_dist_params.get("sigma", 1.5)
            length = int(self.rng.lognormal(mu, sigma))
            return max(1, length)

        elif self.prompt_length_dist == "uniform":
            min_val = self.prompt_dist_params.get("min", 10)
            max_val = self.prompt_dist_params.get("max", 500)
            return int(self.rng.uniform(min_val, max_val))

        elif self.prompt_length_dist == "powerlaw":
            alpha = self.prompt_dist_params.get("alpha", 2.0)
            min_val = self.prompt_dist_params.get("min", 10)
            # Power law distribution
            length = int(min_val * (1.0 - self.rng.random()) ** (-1.0 / (alpha - 1)))
            max_val = self.prompt_dist_params.get("max", 2000)
            return min(max_val, max(min_val, length))

        elif self.prompt_length_dist == "bimodal":
            # For experiment 2: 70% short, 30% long
            if self.rng.random() < 0.7:
                return int(self.rng.uniform(10, 50))
            else:
                return int(self.rng.uniform(500, 2000))

        else:
            raise ValueError(f"Unknown prompt distribution: {self.prompt_length_dist}")

    def generate_output_length(self) -> int:
        """Generate expected output length based on configured distribution."""
        if self.output_length_dist == "truncated_normal":
            mu = self.output_dist_params.get("mu", 100)
            sigma = self.output_dist_params.get("sigma", 30)
            min_val = self.output_dist_params.get("min", 10)
            max_val = self.output_dist_params.get("max", 500)

            length = int(self.rng.normal(mu, sigma))
            return max(min_val, min(max_val, length))

        elif self.output_length_dist == "uniform":
            min_val = self.output_dist_params.get("min", 10)
            max_val = self.output_dist_params.get("max", 500)
            return int(self.rng.uniform(min_val, max_val))

        elif self.output_length_dist == "lognormal":
            mu = self.output_dist_params.get("mu", 4)
            sigma = self.output_dist_params.get("sigma", 1.0)
            length = int(self.rng.lognormal(mu, sigma))
            min_val = self.output_dist_params.get("min", 10)
            max_val = self.output_dist_params.get("max", 500)
            return max(min_val, min(max_val, length))

        else:
            raise ValueError(f"Unknown output distribution: {self.output_length_dist}")

    def generate_request(self) -> Request:
        """Generate a single request with current timestamp."""
        request = Request(
            request_id=self.request_id_counter,
            arrival_time=self.env.now,
            prompt_length=self.generate_prompt_length(),
            expected_output_length=self.generate_output_length(),
        )
        self.request_id_counter += 1
        return request

    def run(self, queue: simpy.Store, max_requests: Optional[int] = None):
        """
        Generate requests according to non-homogeneous Poisson process.

        Args:
            queue: SimPy store to put generated requests
            max_requests: Maximum number of requests to generate (None for unlimited)
        """
        generated = 0

        while max_requests is None or generated < max_requests:
            # Get current arrival rate
            current_rate = self.arrival_rate_func(self.env.now)

            # Inter-arrival time for Poisson process
            if current_rate > 0:
                inter_arrival = self.rng.exponential(1.0 / current_rate)
            else:
                inter_arrival = float('inf')

            # Wait for next arrival
            yield self.env.timeout(inter_arrival)

            # Generate and enqueue request
            request = self.generate_request()
            request.queue_entry_time = self.env.now
            queue.put(request)

            generated += 1


def constant_rate(rate: float) -> Callable[[float], float]:
    """Constant arrival rate function."""
    return lambda t: rate


def step_rate(base_rate: float, step_time: float, step_rate: float) -> Callable[[float], float]:
    """Step function arrival rate."""
    return lambda t: base_rate if t < step_time else step_rate


def sinusoidal_rate(base_rate: float, amplitude: float, period: float) -> Callable[[float], float]:
    """Sinusoidal arrival rate (for experiment 5)."""
    return lambda t: base_rate + amplitude * np.sin(2 * np.pi * t / period)
