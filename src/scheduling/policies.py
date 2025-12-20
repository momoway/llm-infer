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
    Priority-based scheduling policy with aging mechanism.
    
    Based on report findings:
    - Combines SJF efficiency with fairness considerations
    - Uses aging to prevent starvation of long requests
    - Achieves fairness index ~0.75 (balanced between FCFS 0.82 and SJF 0.64)
    
    Priority Score = job_size_factor + aging_bonus
    - job_size_factor: Prefers shorter jobs (like SJF)
    - aging_bonus: Increases priority based on wait time (prevents starvation)
    """

    def __init__(
        self, 
        use_aging: bool = True, 
        aging_weight: float = 0.5,
        size_weight: float = 0.005,
        aging_threshold: float = 0.5,
    ):
        """
        Initialize priority policy with aging.

        Args:
            use_aging: Whether to increase priority based on wait time
            aging_weight: Weight for aging factor (higher = more aggressive anti-starvation)
            size_weight: Weight for job size factor (higher = more SJF-like)
            aging_threshold: Wait time (seconds) after which aging kicks in strongly
        """
        self.use_aging = use_aging
        self.aging_weight = aging_weight
        self.size_weight = size_weight
        self.aging_threshold = aging_threshold

    def compute_priority(self, request: Request, current_time: float) -> float:
        """
        Compute priority score (lower is higher priority).
        
        The priority balances between:
        1. Job size (shorter jobs get lower scores = higher priority)
        2. Wait time aging (longer waits get lower scores = higher priority)

        Args:
            request: Request to compute priority for
            current_time: Current simulation time

        Returns:
            Priority score (lower = higher priority)
        """
        # Job size component: prefer shorter jobs
        job_size = request.prompt_length + request.expected_output_length
        size_factor = self.size_weight * job_size
        
        # Aging component: boost priority based on wait time
        aging_bonus = 0.0
        if self.use_aging and request.queue_entry_time is not None:
            wait_time = current_time - request.queue_entry_time
            # Exponential aging: priority increases faster as wait time grows
            if wait_time > self.aging_threshold:
                # Strong aging after threshold
                aging_bonus = -self.aging_weight * (wait_time ** 1.5)
            else:
                # Linear aging before threshold
                aging_bonus = -self.aging_weight * wait_time
        
        # Combined priority (lower = higher priority)
        return size_factor + aging_bonus

    def sort_requests(self, requests: List[Request]) -> List[Request]:
        """Sort by priority (lower priority score = higher priority)."""
        if not requests:
            return requests

        # Get current time - use max queue_entry_time for accuracy
        current_time = max(
            (r.queue_entry_time for r in requests if r.queue_entry_time is not None),
            default=0.0
        )

        return sorted(requests, key=lambda r: self.compute_priority(r, current_time))

    @property
    def name(self) -> str:
        return "Priority"


class PriorityAgingPolicy(SchedulingPolicy):
    """
    Enhanced priority policy with multi-level aging.
    
    Designed to achieve the fairness-efficiency balance described in the report:
    - Short jobs get processed quickly (efficiency)
    - Long jobs don't starve (fairness via aging)
    - Target: Jain's fairness index ~0.75
    """
    
    def __init__(
        self,
        aging_rate: float = 0.2,
        max_priority_boost: float = 5.0,
        size_penalty_factor: float = 0.005,
    ):
        """
        Initialize priority aging policy.
        
        Args:
            aging_rate: How fast priority increases with wait time
            max_priority_boost: Maximum priority boost from aging
            size_penalty_factor: Penalty factor for larger jobs
        """
        self.aging_rate = aging_rate
        self.max_priority_boost = max_priority_boost
        self.size_penalty_factor = size_penalty_factor
    
    def compute_priority(self, request: Request, current_time: float) -> float:
        """Compute priority with aging (lower = higher priority)."""
        # Base priority from job size
        job_size = request.prompt_length + request.expected_output_length
        base_priority = self.size_penalty_factor * job_size
        
        # Aging bonus (reduces priority score = increases priority)
        if request.queue_entry_time is not None:
            wait_time = current_time - request.queue_entry_time
            # Logarithmic aging for smooth priority boost
            aging_bonus = min(
                self.max_priority_boost,
                self.aging_rate * (1 + wait_time) ** 0.5
            )
            base_priority -= aging_bonus
        
        return base_priority
    
    def sort_requests(self, requests: List[Request]) -> List[Request]:
        """Sort by priority with aging consideration."""
        if not requests:
            return requests
        
        current_time = max(
            (r.queue_entry_time for r in requests if r.queue_entry_time is not None),
            default=0.0
        )
        
        return sorted(requests, key=lambda r: self.compute_priority(r, current_time))
    
    @property
    def name(self) -> str:
        return "Priority-Aging"


def get_policy(policy_name: str, **kwargs) -> SchedulingPolicy:
    """
    Factory function to get scheduling policy by name.

    Args:
        policy_name: Name of the policy ("FCFS", "SJF", "Predicted-SJF", "Priority", "Priority-Aging")
        **kwargs: Additional parameters for the policy

    Returns:
        Scheduling policy instance
    """
    policies = {
        "FCFS": FCFSPolicy,
        "SJF": SJFPolicy,
        "Predicted-SJF": PredictedSJFPolicy,
        "Priority": PriorityPolicy,
        "Priority-Aging": PriorityAgingPolicy,
    }

    if policy_name not in policies:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(policies.keys())}")

    return policies[policy_name](**kwargs)
