"""Adaptive batching LLM inference server."""

import simpy
from typing import List, Optional
from .request import Request
from .batch_processor import BatchProcessor
from ..scheduling.policies import SchedulingPolicy, FCFSPolicy


class AdaptiveBatchingServer:
    """
    LLM Server with adaptive batch sizing based on queue depth.
    
    Based on the report's findings:
    - Increases batch size when queue is deep to maximize throughput
    - Decreases batch size when queue is shallow for lower latency
    - Implements statistical multiplexing for non-stationary loads
    """
    
    def __init__(
        self,
        env: simpy.Environment,
        base_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 128,
        queue_threshold_low: int = 10,
        queue_threshold_high: int = 50,
        scheduling_policy: Optional[SchedulingPolicy] = None,
        batch_timeout: float = 0.5,
        alpha: float = 0.00015,
        beta: float = 0.008,
        gamma: float = 0.010,
    ):
        """
        Initialize adaptive batching server.
        
        Args:
            env: SimPy environment
            base_batch_size: Default batch size during normal load
            min_batch_size: Minimum batch size during low load
            max_batch_size: Maximum batch size during high load
            queue_threshold_low: Queue length below which to use smaller batches
            queue_threshold_high: Queue length above which to use larger batches
            scheduling_policy: Policy for ordering requests within batch
            batch_timeout: Maximum time to wait for batch to fill
            alpha: Prefill per-token time
            beta: Prefill overhead
            gamma: Decode step time
        """
        self.env = env
        self.base_batch_size = base_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.queue_threshold_low = queue_threshold_low
        self.queue_threshold_high = queue_threshold_high
        self.scheduling_policy = scheduling_policy or FCFSPolicy()
        self.batch_timeout = batch_timeout
        
        # Request queue
        self.request_queue: List[Request] = []
        self.queue_store = simpy.Store(env)
        
        # Batch processor
        self.processor = BatchProcessor(env, alpha, beta, gamma)
        
        # Completed requests
        self.completed_requests: List[Request] = []
        
        # Statistics
        self.max_queue_length = 0
        self.current_queue_length = 0
        self.current_batch_size = base_batch_size
        
        # Track batch size history for analysis
        self.batch_size_history: List[tuple] = []  # (time, batch_size, queue_length)
        self.batch_sizes_used: List[int] = []
    
    def get_adaptive_batch_size(self) -> int:
        """
        Determine batch size based on current queue depth.
        
        Strategy:
        - Queue < threshold_low: Use smaller batch for lower latency
        - Queue > threshold_high: Use larger batch for higher throughput
        - Otherwise: Use base batch size
        """
        queue_len = self.current_queue_length
        
        if queue_len < self.queue_threshold_low:
            # Queue is short, optimize for latency
            new_size = max(self.min_batch_size, self.base_batch_size // 2)
        elif queue_len > self.queue_threshold_high:
            # Queue is deep, optimize for throughput
            # Scale batch size based on queue depth
            scale_factor = min(4, 1 + (queue_len - self.queue_threshold_high) / 50)
            new_size = min(self.max_batch_size, int(self.base_batch_size * scale_factor))
        else:
            # Normal operation
            new_size = self.base_batch_size
        
        # Record batch size change
        if new_size != self.current_batch_size:
            self.batch_size_history.append((self.env.now, new_size, queue_len))
        
        self.current_batch_size = new_size
        return new_size
    
    def run(self):
        """Main server loop with adaptive batching."""
        while True:
            # Get adaptive batch size based on current queue state
            batch_size = self.get_adaptive_batch_size()
            
            # Collect requests for batch
            batch = yield self.env.process(self.collect_batch(batch_size))
            
            if not batch:
                yield self.env.timeout(0.1)
                continue
            
            # Record batch size used
            self.batch_sizes_used.append(len(batch))
            
            # Sort batch according to scheduling policy
            sorted_batch = self.scheduling_policy.sort_requests(batch)
            
            # Process the batch
            yield self.env.process(self.processor.process_batch(sorted_batch))
            
            # Move to completed
            self.completed_requests.extend(sorted_batch)
    
    def collect_batch(self, batch_size: int):
        """Collect requests up to batch_size with timeout."""
        batch = []
        deadline = self.env.now + self.batch_timeout
        
        while len(batch) < batch_size:
            remaining_time = max(0, deadline - self.env.now)
            
            if remaining_time <= 0 and batch:
                break
            
            try:
                timeout_event = self.env.timeout(remaining_time)
                get_event = self.queue_store.get()
                result = yield timeout_event | get_event
                
                if get_event in result:
                    req = result[get_event]
                    batch.append(req)
                    self.current_queue_length -= 1
                else:
                    if batch:
                        break
            except simpy.Interrupt:
                break
        
        return batch
    
    def enqueue_request(self, request: Request):
        """Add request to queue and track statistics."""
        self.current_queue_length += 1
        self.max_queue_length = max(self.max_queue_length, self.current_queue_length)
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
            "batch_size_mode": "adaptive",
            "base_batch_size": self.base_batch_size,
            "batch_size_changes": len(self.batch_size_history),
        }
        
        # Add percentiles
        if latencies:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            stats["p50_latency"] = latencies_sorted[int(0.50 * n)]
            stats["p95_latency"] = latencies_sorted[int(0.95 * n)]
            stats["p99_latency"] = latencies_sorted[int(0.99 * n)]
        
        # Add batch size statistics
        if self.batch_sizes_used:
            stats["avg_batch_size"] = sum(self.batch_sizes_used) / len(self.batch_sizes_used)
            stats["min_batch_size_used"] = min(self.batch_sizes_used)
            stats["max_batch_size_used"] = max(self.batch_sizes_used)
        
        # Add processor statistics
        stats.update(self.processor.get_statistics())
        
        return stats


