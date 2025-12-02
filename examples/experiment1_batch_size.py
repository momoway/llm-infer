#!/usr/bin/env python3
"""
Experiment 1: Batch Size Sensitivity Analysis

Parameters: Batch sizes B ∈ {1, 2, 4, 8, 16, 32, 64, 128}
Fixed conditions: Constant arrival rate λ = 10 req/s, FCFS scheduling
Metrics: Average latency, p50/p95/p99 latency, throughput, GPU utilization
Hypothesis: Optimal batch size will be between 16-32 for this load level
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiments.config import SimulationConfig, ExperimentConfig
from src.experiments.runner import SimulationRunner, ExperimentRunner
import matplotlib.pyplot as plt
import json


def run_experiment():
    """Run Experiment 1: Batch Size Sensitivity Analysis."""
    print("=" * 60)
    print("Experiment 1: Batch Size Sensitivity Analysis")
    print("=" * 60)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    results_by_batch_size = {}

    for batch_size in batch_sizes:
        print(f"\n--- Testing batch size: {batch_size} ---")

        # Create base configuration
        base_config = SimulationConfig(
            arrival_rate=10.0,
            arrival_pattern="constant",
            batch_size=batch_size,
            scheduling_policy="FCFS",
            num_requests=10000,
            warmup_requests=500,
        )

        # Create experiment with 30 replications
        experiment = ExperimentConfig(
            name=f"batch_size_{batch_size}",
            description=f"Batch size sensitivity analysis with B={batch_size}",
            base_config=base_config,
            num_replications=5,  # Use 5 for quick test, change to 30 for full experiment
            random_seed_start=42,
        )

        # Run replications
        runner = ExperimentRunner(experiment.get_replication_configs())
        results = runner.run_all(verbose=True)

        # Aggregate results
        aggregated = runner.aggregate_results(results)
        results_by_batch_size[batch_size] = aggregated

        # Print summary
        print(f"\nResults for batch size {batch_size}:")
        print(f"  Avg Latency: {aggregated['avg_latency']['mean']:.3f}s "
              f"(95% CI: [{aggregated['avg_latency']['ci_lower']:.3f}, "
              f"{aggregated['avg_latency']['ci_upper']:.3f}])")
        print(f"  P99 Latency: {aggregated['p99_latency']['mean']:.3f}s")
        print(f"  Throughput: {aggregated['throughput_req_per_sec']['mean']:.2f} req/s")

    # Save results
    output_file = "experiment1_results.json"
    with open(output_file, 'w') as f:
        # Convert to serializable format
        serializable_results = {
            str(k): {
                metric: {
                    "mean": v["mean"],
                    "std": v["std"],
                    "ci_lower": v["ci_lower"],
                    "ci_upper": v["ci_upper"]
                }
                for metric, v in metrics.items()
            }
            for k, metrics in results_by_batch_size.items()
        }
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    # Plot results
    plot_results(results_by_batch_size)


def plot_results(results_by_batch_size):
    """Plot key metrics vs batch size."""
    batch_sizes = sorted(results_by_batch_size.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Average Latency
    ax = axes[0, 0]
    means = [results_by_batch_size[b]['avg_latency']['mean'] for b in batch_sizes]
    ci_lower = [results_by_batch_size[b]['avg_latency']['ci_lower'] for b in batch_sizes]
    ci_upper = [results_by_batch_size[b]['avg_latency']['ci_upper'] for b in batch_sizes]
    ax.plot(batch_sizes, means, 'o-', linewidth=2)
    ax.fill_between(batch_sizes, ci_lower, ci_upper, alpha=0.3)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Average Latency (s)')
    ax.set_title('Average Latency vs Batch Size')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: P99 Latency
    ax = axes[0, 1]
    means = [results_by_batch_size[b]['p99_latency']['mean'] for b in batch_sizes]
    ci_lower = [results_by_batch_size[b]['p99_latency']['ci_lower'] for b in batch_sizes]
    ci_upper = [results_by_batch_size[b]['p99_latency']['ci_upper'] for b in batch_sizes]
    ax.plot(batch_sizes, means, 'o-', linewidth=2, color='orange')
    ax.fill_between(batch_sizes, ci_lower, ci_upper, alpha=0.3, color='orange')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('P99 Latency (s)')
    ax.set_title('P99 Latency vs Batch Size')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Plot 3: Throughput
    ax = axes[1, 0]
    means = [results_by_batch_size[b]['throughput_req_per_sec']['mean'] for b in batch_sizes]
    ci_lower = [results_by_batch_size[b]['throughput_req_per_sec']['ci_lower'] for b in batch_sizes]
    ci_upper = [results_by_batch_size[b]['throughput_req_per_sec']['ci_upper'] for b in batch_sizes]
    ax.plot(batch_sizes, means, 'o-', linewidth=2, color='green')
    ax.fill_between(batch_sizes, ci_lower, ci_upper, alpha=0.3, color='green')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('Throughput vs Batch Size')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Plot 4: Max Queue Length
    ax = axes[1, 1]
    means = [results_by_batch_size[b]['max_queue_length']['mean'] for b in batch_sizes]
    ax.bar(range(len(batch_sizes)), means, color='red', alpha=0.6)
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Max Queue Length')
    ax.set_title('Max Queue Length vs Batch Size')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('experiment1_batch_size_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Plots saved to experiment1_batch_size_analysis.png")


if __name__ == "__main__":
    run_experiment()
