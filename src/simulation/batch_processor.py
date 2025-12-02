"""Batch processor for LLM inference with prefill and decode phases."""

import simpy
from typing import List, Optional
from .request import Request


class BatchProcessor:
    """
    Simulates GPU inference with two-phase execution model.

    Phase 1 - Prefill: Process input prompts in parallel (compute-bound)
    Phase 2 - Decode: Generate tokens autoregressively (memory-bound)
    """

    def __init__(
        self,
        env: simpy.Environment,
        alpha: float = 0.001,  # Prefill time per token (s/token)
        beta: float = 0.05,     # Prefill overhead (s)
        gamma: float = 0.0005,  # Decode time per step (s/step)
    ):
        """
        Initialize batch processor with timing parameters from report.

        Args:
            env: SimPy environment
            alpha: Prefill per-token processing time
            beta: Fixed prefill overhead for batch setup
            gamma: Decode step time (constant per iteration)
        """
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Statistics
        self.total_batches_processed = 0
        self.total_requests_processed = 0

    def compute_prefill_time(self, batch: List[Request]) -> float:
        """
        Compute prefill time for a batch.

        Formula: T_prefill = α * Σ(l_prompt_i) + β

        Args:
            batch: List of requests in the batch

        Returns:
            Prefill time in seconds
        """
        total_prompt_tokens = sum(req.prompt_length for req in batch)
        return self.alpha * total_prompt_tokens + self.beta

    def compute_decode_time(self, batch: List[Request]) -> float:
        """
        Compute decode time for a batch.

        Formula: T_decode_total = γ * max(l_output_i)

        The batch must perform max(output_length) decode steps,
        waiting for the longest request to complete.

        Args:
            batch: List of requests in the batch

        Returns:
            Total decode time in seconds
        """
        max_output_length = max(req.expected_output_length for req in batch)
        return self.gamma * max_output_length

    def process_batch(self, batch: List[Request]) -> float:
        """
        Process a batch of requests through prefill and decode phases.

        Args:
            batch: List of requests to process

        Returns:
            Total processing time (prefill + decode)
        """
        if not batch:
            return 0.0

        processing_start = self.env.now

        # Mark all requests as started processing
        for req in batch:
            req.processing_start_time = processing_start
            # Set actual output length (in real system, this varies)
            # For simulation, we use expected length
            req.actual_output_length = req.expected_output_length

        # Phase 1: Prefill (parallel processing of all prompts)
        prefill_time = self.compute_prefill_time(batch)
        yield self.env.timeout(prefill_time)

        # Phase 2: Decode (iterative token generation)
        decode_time = self.compute_decode_time(batch)
        yield self.env.timeout(decode_time)

        # Mark all requests as completed
        processing_end = self.env.now
        for req in batch:
            req.processing_end_time = processing_end

        # Update statistics
        self.total_batches_processed += 1
        self.total_requests_processed += len(batch)

        total_time = prefill_time + decode_time
        return total_time

    def get_statistics(self) -> dict:
        """Get processor statistics."""
        return {
            "total_batches_processed": self.total_batches_processed,
            "total_requests_processed": self.total_requests_processed,
            "avg_batch_size": (
                self.total_requests_processed / self.total_batches_processed
                if self.total_batches_processed > 0
                else 0
            ),
        }
