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
    described in the report. Supports various scheduling policies
    that affect which requests are selected for batching.
    """

    def __init__(
        self,
        env: simpy.Environment,
        batch_size: int,
        scheduling_policy: Optional[SchedulingPolicy] = None,
        batch_timeout: Optional[float] = None,
        alpha: float = 0.00015,
        beta: float = 0.008,
        gamma: float = 0.010,
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

        # Request queue - use a list for proper scheduling policy support
        self.request_queue: List[Request] = []
        # Event to signal new request arrival
        self.new_request_event = simpy.Event(env)

        # Batch processor
        self.processor = BatchProcessor(env, alpha, beta, gamma)

        # Completed requests
        self.completed_requests: List[Request] = []

        # Server statistics
        self.max_queue_length = 0
        self.current_queue_length = 0
        self.total_queue_wait_time = 0.0

    def run(self):
        """Main server loop: collect requests, form batches, process."""
        while True:
            # Collect requests for next batch using scheduling policy
            batch = yield self.env.process(self.collect_batch())

            if not batch:
                # No requests available, wait for new arrivals
                yield self.env.timeout(0.1)
                continue

            # Process the batch
            yield self.env.process(self.processor.process_batch(batch))

            # Move completed requests
            self.completed_requests.extend(batch)

    def select_batch_from_queue(self) -> List[Request]:
        """
        Select requests from queue according to scheduling policy.
        
        Returns:
            List of requests to form the next batch
        """
        if not self.request_queue:
            return []
        
        # Sort queue according to scheduling policy
        sorted_queue = self.scheduling_policy.sort_requests(self.request_queue.copy())
        
        # Select top batch_size requests
        batch_size = min(self.batch_size, len(sorted_queue))
        selected = sorted_queue[:batch_size]
        
        # Remove selected requests from queue
        for req in selected:
            self.request_queue.remove(req)
            self.current_queue_length -= 1
        
        return selected

    def collect_batch(self):
        """
        Collect requests to form a batch using the scheduling policy.

        Strategy:
        1. If queue already has batch_size requests, form batch immediately
        2. Otherwise, wait for requests to accumulate up to batch_size
        3. Use batch_timeout to avoid waiting too long for partial batches
        4. Select batch according to scheduling policy
        """
        timeout = self.batch_timeout if self.batch_timeout else 1.0
        start_time = self.env.now
        
        # Wait until we have enough requests or timeout
        while len(self.request_queue) < self.batch_size:
            elapsed = self.env.now - start_time
            if elapsed >= timeout:
                # Timeout - form partial batch with what we have
                break
            
            remaining = timeout - elapsed
            
            # If queue is empty, wait for at least one request
            if not self.request_queue:
                self.new_request_event = simpy.Event(self.env)
                timeout_event = self.env.timeout(remaining)
                result = yield self.new_request_event | timeout_event
                
                if timeout_event in result and timeout_event.processed:
                    # Timeout with empty queue
                    break
                continue
            
            # We have some requests, wait a bit more for batch to fill
            # Use shorter wait when we have more requests (dynamic batching)
            fill_ratio = len(self.request_queue) / self.batch_size
            wait_time = min(remaining, 0.2 * (1 - fill_ratio))
            
            if wait_time <= 0.001:
                break
            
            self.new_request_event = simpy.Event(self.env)
            timeout_event = self.env.timeout(wait_time)
            yield self.new_request_event | timeout_event
        
        # Select batch according to scheduling policy
        if self.request_queue:
            batch = self.select_batch_from_queue()
        else:
            batch = []
        
        return batch

    def enqueue_request(self, request: Request):
        """
        Add request to queue and track statistics.
        
        Requests are added to a list and selected according to the
        scheduling policy when forming batches.
        """
        # Add to queue
        self.request_queue.append(request)
        self.current_queue_length += 1
        
        # Update max queue length
        self.max_queue_length = max(self.max_queue_length, self.current_queue_length)
        
        # Signal that a new request has arrived
        if not self.new_request_event.triggered:
            self.new_request_event.succeed()

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
