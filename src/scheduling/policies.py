"""Scheduling policies for LLM inference request batching."""

from abc import ABC, abstractmethod
from typing import List
from ..simulation.request import Request


class SchedulingPolicy(ABC):
    """Abstract base class for scheduling policies."""

    @abstractmethod
    def sort_requests(self, requests: List[Request]) -> List[Request]:
        """
        Sort requests according to scheduling policy.

        Args:
            requests: List of requests to sort

        Returns:
            Sorted list of requests
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the scheduling policy."""
        pass


class FCFSPolicy(SchedulingPolicy):
    """First-Come-First-Serve scheduling policy."""

    def sort_requests(self, requests: List[Request]) -> List[Request]:
        """Sort by arrival time (earliest first)."""
        return sorted(requests, key=lambda r: r.arrival_time)

    @property
    def name(self) -> str:
        return "FCFS"


class SJFPolicy(SchedulingPolicy):
    """Shortest Job First scheduling policy (based on prompt length)."""

    def sort_requests(self, requests: List[Request]) -> List[Request]:
        """Sort by prompt length (shortest first)."""
        return sorted(requests, key=lambda r: r.prompt_length)

    @property
    def name(self) -> str:
        return "SJF"


class PredictedSJFPolicy(SchedulingPolicy):
    """
    Predicted Shortest Job First policy.

    Estimates total processing time based on prompt and expected output length.
    Uses the timing model from the report:
    - Prefill time: α * prompt_length + β
    - Decode time: γ * expected_output_length
    """

    def __init__(self, alpha: float = 0.001, beta: float = 0.05, gamma: float = 0.0005):
        """
        Initialize with timing model parameters.

        Args:
            alpha: Per-token prefill time (s/token)
            beta: Prefill overhead (s)
            gamma: Per-step decode time (s/step)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def estimate_processing_time(self, request: Request) -> float:
        """Estimate total processing time for a request."""
        prefill_time = self.alpha * request.prompt_length + self.beta
        decode_time = self.gamma * request.expected_output_length
        return prefill_time + decode_time

    def sort_requests(self, requests: List[Request]) -> List[Request]:
        """Sort by estimated processing time (shortest first)."""
        return sorted(requests, key=lambda r: self.estimate_processing_time(r))

    @property
    def name(self) -> str:
        return "Predicted-SJF"


class PriorityPolicy(SchedulingPolicy):
    """
    Priority-based scheduling policy (for SLA-aware scheduling).

    Requests can have different priority levels.
    This is a simple implementation where priority could be based on:
    - Request metadata (not yet implemented in Request class)
    - Waiting time (aging to prevent starvation)
    - Estimated completion time
    """

    def __init__(self, use_aging: bool = True, aging_weight: float = 0.1):
        """
        Initialize priority policy.

        Args:
            use_aging: Whether to increase priority based on wait time
            aging_weight: Weight for aging factor
        """
        self.use_aging = use_aging
        self.aging_weight = aging_weight

    def compute_priority(self, request: Request, current_time: float) -> float:
        """
        Compute priority score (lower is higher priority).

        Args:
            request: Request to compute priority for
            current_time: Current simulation time

        Returns:
            Priority score (lower = higher priority)
        """
        # Base priority could be extended with request.priority field
        base_priority = 0.0

        # Aging: reduce priority score based on wait time
        if self.use_aging and request.queue_entry_time is not None:
            wait_time = current_time - request.queue_entry_time
            base_priority -= self.aging_weight * wait_time

        # Could add other factors here (request size, estimated time, etc.)

        return base_priority

    def sort_requests(self, requests: List[Request]) -> List[Request]:
        """Sort by priority (lower priority score = higher priority)."""
        if not requests:
            return requests

        # Get current time from first request (approximation)
        current_time = requests[0].queue_entry_time or 0.0

        return sorted(requests, key=lambda r: self.compute_priority(r, current_time))

    @property
    def name(self) -> str:
        return "Priority"


def get_policy(policy_name: str, **kwargs) -> SchedulingPolicy:
    """
    Factory function to get scheduling policy by name.

    Args:
        policy_name: Name of the policy ("FCFS", "SJF", "Predicted-SJF", "Priority")
        **kwargs: Additional parameters for the policy

    Returns:
        Scheduling policy instance
    """
    policies = {
        "FCFS": FCFSPolicy,
        "SJF": SJFPolicy,
        "Predicted-SJF": PredictedSJFPolicy,
        "Priority": PriorityPolicy,
    }

    if policy_name not in policies:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(policies.keys())}")

    return policies[policy_name](**kwargs)
