"""Metrics collection and analysis for LLM inference simulation."""

import numpy as np
from typing import List, Dict, Any
from ..simulation.request import Request


class MetricsCollector:
    """Collects and analyzes metrics from simulation runs."""

    def __init__(self, warmup_requests: int = 500):
        """
        Initialize metrics collector.

        Args:
            warmup_requests: Number of requests to skip for warm-up period
        """
        self.warmup_requests = warmup_requests

    def compute_metrics(self, requests: List[Request], additional_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compute comprehensive metrics from completed requests.

        Args:
            requests: List of completed requests
            additional_stats: Additional statistics from server

        Returns:
            Dictionary of metrics
        """
        # Skip warmup period
        requests_after_warmup = requests[self.warmup_requests:] if len(requests) > self.warmup_requests else requests

        if not requests_after_warmup:
            return {"error": "No requests after warmup period"}

        metrics = {}

        # Latency metrics
        latencies = [r.total_latency for r in requests_after_warmup if r.total_latency is not None]
        if latencies:
            metrics.update(self._compute_latency_metrics(latencies))

        # Queue wait time metrics
        queue_waits = [r.queue_wait_time for r in requests_after_warmup if r.queue_wait_time is not None]
        if queue_waits:
            metrics.update(self._compute_queue_metrics(queue_waits))

        # Processing time metrics
        processing_times = [r.processing_time for r in requests_after_warmup if r.processing_time is not None]
        if processing_times:
            metrics.update(self._compute_processing_metrics(processing_times))

        # Throughput metrics
        metrics.update(self._compute_throughput_metrics(requests_after_warmup))

        # Fairness metrics (Jain's fairness index)
        if latencies:
            metrics["fairness_index"] = self._compute_fairness_index(latencies)

        # Token metrics
        metrics.update(self._compute_token_metrics(requests_after_warmup))

        # Add additional stats if provided
        if additional_stats:
            metrics.update(additional_stats)

        return metrics

    def _compute_latency_metrics(self, latencies: List[float]) -> Dict[str, float]:
        """Compute latency statistics."""
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        return {
            "avg_latency": np.mean(latencies),
            "median_latency": np.median(latencies),
            "std_latency": np.std(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "p50_latency": latencies_sorted[int(0.50 * n)],
            "p95_latency": latencies_sorted[int(0.95 * n)],
            "p99_latency": latencies_sorted[int(0.99 * n)],
        }

    def _compute_queue_metrics(self, queue_waits: List[float]) -> Dict[str, float]:
        """Compute queue wait time statistics."""
        return {
            "avg_queue_wait": np.mean(queue_waits),
            "median_queue_wait": np.median(queue_waits),
            "max_queue_wait": max(queue_waits),
            "p95_queue_wait": np.percentile(queue_waits, 95),
            "p99_queue_wait": np.percentile(queue_waits, 99),
        }

    def _compute_processing_metrics(self, processing_times: List[float]) -> Dict[str, float]:
        """Compute processing time statistics."""
        return {
            "avg_processing_time": np.mean(processing_times),
            "median_processing_time": np.median(processing_times),
            "max_processing_time": max(processing_times),
        }

    def _compute_throughput_metrics(self, requests: List[Request]) -> Dict[str, float]:
        """Compute throughput metrics."""
        if not requests:
            return {}

        # Get time span
        start_time = min(r.arrival_time for r in requests)
        end_time = max(r.processing_end_time for r in requests if r.processing_end_time is not None)

        time_span = end_time - start_time if end_time > start_time else 1.0

        # Calculate throughput
        throughput = len(requests) / time_span

        # Token throughput
        total_tokens = sum(r.total_tokens for r in requests)
        token_throughput = total_tokens / time_span

        return {
            "throughput_req_per_sec": throughput,
            "token_throughput_per_sec": token_throughput,
            "total_requests": len(requests),
            "simulation_time": time_span,
        }

    def _compute_fairness_index(self, latencies: List[float]) -> float:
        """
        Compute Jain's fairness index.

        Formula: (Σx_i)^2 / (n * Σx_i^2)

        Returns value between 0 and 1, where 1 is perfectly fair.

        Args:
            latencies: List of latencies

        Returns:
            Fairness index (0 to 1)
        """
        n = len(latencies)
        if n == 0:
            return 0.0

        sum_latencies = sum(latencies)
        sum_squared_latencies = sum(x * x for x in latencies)

        if sum_squared_latencies == 0:
            return 1.0

        return (sum_latencies ** 2) / (n * sum_squared_latencies)

    def _compute_token_metrics(self, requests: List[Request]) -> Dict[str, float]:
        """Compute token-related metrics."""
        prompt_lengths = [r.prompt_length for r in requests]
        output_lengths = [r.actual_output_length or r.expected_output_length for r in requests]

        return {
            "avg_prompt_length": np.mean(prompt_lengths),
            "avg_output_length": np.mean(output_lengths),
            "total_tokens_processed": sum(r.total_tokens for r in requests),
        }

    def compute_starvation_rate(self, requests: List[Request], threshold_multiplier: float = 3.0) -> float:
        """
        Compute starvation rate (percentage of requests with excessive wait times).

        A request is considered starved if its latency exceeds threshold_multiplier times
        the median latency.

        Args:
            requests: List of completed requests
            threshold_multiplier: Multiplier for median latency to define starvation

        Returns:
            Starvation rate (0 to 1)
        """
        latencies = [r.total_latency for r in requests if r.total_latency is not None]

        if not latencies:
            return 0.0

        median_latency = np.median(latencies)
        threshold = threshold_multiplier * median_latency

        starved_count = sum(1 for lat in latencies if lat > threshold)

        return starved_count / len(latencies)

    def compute_confidence_interval(
        self, values: List[float], confidence: float = 0.95
    ) -> tuple[float, float, float]:
        """
        Compute confidence interval for a list of values.

        Args:
            values: List of values
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        mean = np.mean(values)
        std_error = np.std(values, ddof=1) / np.sqrt(len(values))

        # Using t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)

        margin = t_value * std_error
        return mean, mean - margin, mean + margin
