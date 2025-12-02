"""Configuration dataclasses for experiments."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""

    # Workload parameters
    arrival_rate: float = 10.0  # requests per second
    arrival_pattern: str = "constant"  # constant, step, sinusoidal
    arrival_pattern_params: Dict[str, Any] = field(default_factory=dict)

    # Request distribution parameters
    prompt_length_dist: str = "lognormal"
    prompt_dist_params: Dict[str, Any] = field(default_factory=lambda: {"mu": 4, "sigma": 1.5})
    output_length_dist: str = "truncated_normal"
    output_dist_params: Dict[str, Any] = field(
        default_factory=lambda: {"mu": 100, "sigma": 30, "min": 10, "max": 500}
    )

    # Server parameters
    batch_size: int = 32
    batch_timeout: Optional[float] = None
    scheduling_policy: str = "FCFS"
    scheduling_policy_params: Dict[str, Any] = field(default_factory=dict)

    # Timing model parameters (from report)
    alpha: float = 0.001  # Prefill per-token time (s/token)
    beta: float = 0.05    # Prefill overhead (s)
    gamma: float = 0.0005 # Decode step time (s/step)

    # Simulation parameters
    num_requests: int = 10000
    warmup_requests: int = 500
    random_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "arrival_rate": self.arrival_rate,
            "arrival_pattern": self.arrival_pattern,
            "batch_size": self.batch_size,
            "scheduling_policy": self.scheduling_policy,
            "num_requests": self.num_requests,
            "warmup_requests": self.warmup_requests,
            "random_seed": self.random_seed,
        }


@dataclass
class ExperimentConfig:
    """Configuration for a multi-run experiment."""

    name: str
    description: str
    base_config: SimulationConfig
    num_replications: int = 30  # For statistical rigor (95% CI)
    random_seed_start: int = 42

    def get_replication_configs(self) -> list:
        """Generate configs for all replications with different random seeds."""
        configs = []
        for i in range(self.num_replications):
            config = SimulationConfig(**vars(self.base_config))
            config.random_seed = self.random_seed_start + i
            configs.append(config)
        return configs
