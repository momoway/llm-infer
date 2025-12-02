"""Main LLM inference server simulation."""

import simpy
from typing import List, Optional
from .request import Request
from .batch_processor import BatchProcessor
from ..scheduling.policies import SchedulingPolicy, FCFSPolicy


class LLMInferenceServer:
    """
    Simulates an LLM inference server with continuous batching.

    Implements the queue management and batch processing pipeline
    described in the report.
    """

    def __init__(
        self,
        env: simpy.Environment,
        batch_size: int,
        scheduling_policy: Optional[SchedulingPolicy] = None,
        batch_timeout: Optional[float] = None,
        alpha: float = 0.001,
        beta: float = 0.05,
        gamma: float = 0.0005,
    ):
        """
        Initialize LLM inference server.

        Args:
            env: SimPy environment
            batch_size: Maximum batch size
            scheduling_policy: Scheduling policy for request ordering
            batch_timeout: Timeout to form batch even if not full (None = no timeout)
            alpha: Prefill per-token time
            beta: Prefill overhead
            gamma: Decode step time
        """
        self.env = env
        self.batch_size = batch_size
        self.scheduling_policy = scheduling_policy or FCFSPolicy()
        self.batch_timeout = batch_timeout

        # Request queue
        self.request_queue: List[Request] = []
        self.queue_store = simpy.Store(env)

        # Batch processor
        self.processor = BatchProcessor(env, alpha, beta, gamma)

        # Completed requests
        self.completed_requests: List[Request] = []

        # Server statistics
        self.max_queue_length = 0
        self.current_queue_length = 0  # Track current queue length manually
        self.total_queue_wait_time = 0.0

    def run(self):
        """Main server loop: collect requests, form batches, process."""
        while True:
            # Collect requests for next batch
            batch = yield self.env.process(self.collect_batch())

            if not batch:
                # No more requests, wait a bit
                yield self.env.timeout(0.1)
                continue

            # Sort batch according to scheduling policy
            # Note: In real continuous batching, sorting happens at queue level
            # Here we sort just before processing for simplicity
            sorted_batch = self.scheduling_policy.sort_requests(batch)

            # Process the batch
            yield self.env.process(self.processor.process_batch(sorted_batch))

            # Move completed requests
            self.completed_requests.extend(sorted_batch)

    def collect_batch(self):
        """
        Collect requests to form a batch.

        Strategies:
        1. Wait until batch_size requests accumulated
        2. Use timeout to process partial batches
        """
        batch = []
        deadline = self.env.now + (self.batch_timeout if self.batch_timeout else float('inf'))

        while len(batch) < self.batch_size:
            # Wait for new request from generator
            try:
                # Wait for request or timeout
                remaining_time = max(0, deadline - self.env.now)
                if remaining_time <= 0 and batch:
                    # Timeout reached, process partial batch
                    break

                # Try to get a request with timeout
                if self.batch_timeout:
                    timeout_event = self.env.timeout(remaining_time)
                    get_event = self.queue_store.get()
                    result = yield timeout_event | get_event

                    if get_event in result:
                        req = result[get_event]
                        batch.append(req)
                        # Decrement queue length when removing from queue
                        self.current_queue_length -= 1
                    else:
                        # Timeout, process partial batch if any
                        if batch:
                            break
                else:
                    # No timeout, just wait for request
                    req = yield self.queue_store.get()
                    batch.append(req)
                    # Decrement queue length when removing from queue
                    self.current_queue_length -= 1

            except simpy.Interrupt:
                break

        return batch

    def enqueue_request(self, request: Request):
        """
        Add request to queue and track statistics.

        Note: Store has unlimited capacity by default, so put() succeeds immediately.
        """
        # Increment queue length
        self.current_queue_length += 1
        # Update max queue length
        self.max_queue_length = max(self.max_queue_length, self.current_queue_length)
        # Actually add to the store (immediate for unlimited Store)
        self.queue_store.put(request)

    def get_statistics(self) -> dict:
        """Get comprehensive server statistics."""
        completed = self.completed_requests

        if not completed:
            return {
                "total_requests": 0,
                "completed_requests": 0,
                "avg_latency": 0,
                "max_queue_length": self.max_queue_length,
            }

        latencies = [req.total_latency for req in completed if req.total_latency is not None]
        queue_waits = [req.queue_wait_time for req in completed if req.queue_wait_time is not None]
        processing_times = [req.processing_time for req in completed if req.processing_time is not None]

        stats = {
            "total_requests": len(completed),
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "avg_queue_wait": sum(queue_waits) / len(queue_waits) if queue_waits else 0,
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "max_queue_length": self.max_queue_length,
            "scheduling_policy": self.scheduling_policy.name,
            "batch_size": self.batch_size,
        }

        # Add percentiles
        if latencies:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            stats["p50_latency"] = latencies_sorted[int(0.50 * n)]
            stats["p95_latency"] = latencies_sorted[int(0.95 * n)]
            stats["p99_latency"] = latencies_sorted[int(0.99 * n)]

        # Add processor statistics
        stats.update(self.processor.get_statistics())

        return stats
