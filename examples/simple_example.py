#!/usr/bin/env python3
"""
Simple example demonstrating basic LLM inference simulation.

This script runs a simple simulation with default parameters and prints
basic metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiments.config import SimulationConfig
from src.experiments.runner import SimulationRunner


def main():
    """Run a simple simulation example."""
    print("=" * 60)
    print("Simple LLM Inference Simulation Example")
    print("=" * 60)

    # Create configuration
    config = SimulationConfig(
        arrival_rate=5.0,           # 5 requests per second
        batch_size=32,              # Batch size of 32
        scheduling_policy="FCFS",   # First-Come-First-Serve
        num_requests=1000,          # Generate 1000 requests
        warmup_requests=100,        # Skip first 100 for warmup
        random_seed=42,             # For reproducibility
    )

    print("\nSimulation Configuration:")
    print(f"  Arrival rate: {config.arrival_rate} req/s")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Scheduling policy: {config.scheduling_policy}")
    print(f"  Number of requests: {config.num_requests}")
    print(f"  Warmup requests: {config.warmup_requests}")

    # Run simulation
    print("\nRunning simulation...")
    runner = SimulationRunner(config)
    metrics = runner.run()

    # Print results
    print("\n" + "=" * 60)
    print("Simulation Results")
    print("=" * 60)

    print(f"\nLatency Metrics:")
    print(f"  Average latency: {metrics.get('avg_latency', 0):.3f} s")
    print(f"  Median latency:  {metrics.get('median_latency', 0):.3f} s")
    print(f"  P95 latency:     {metrics.get('p95_latency', 0):.3f} s")
    print(f"  P99 latency:     {metrics.get('p99_latency', 0):.3f} s")
    print(f"  Max latency:     {metrics.get('max_latency', 0):.3f} s")

    print(f"\nQueue Metrics:")
    print(f"  Average queue wait: {metrics.get('avg_queue_wait', 0):.3f} s")
    print(f"  Max queue length:   {metrics.get('max_queue_length', 0)}")

    print(f"\nThroughput Metrics:")
    print(f"  Throughput:       {metrics.get('throughput_req_per_sec', 0):.2f} req/s")
    print(f"  Token throughput: {metrics.get('token_throughput_per_sec', 0):.2f} tokens/s")

    print(f"\nProcessing Metrics:")
    print(f"  Total requests processed: {metrics.get('total_requests', 0)}")
    print(f"  Total batches:            {metrics.get('total_batches_processed', 0)}")
    print(f"  Average batch size:       {metrics.get('avg_batch_size', 0):.2f}")

    print(f"\nFairness Metrics:")
    print(f"  Fairness index:    {metrics.get('fairness_index', 0):.3f}")
    print(f"  Starvation rate:   {metrics.get('starvation_rate', 0):.3%}")

    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
