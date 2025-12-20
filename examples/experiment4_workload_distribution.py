#!/usr/bin/env python3
"""
Experiment 4: Workload Distribution Sensitivity

Distributions: Uniform, LogNormal (various σ), PowerLaw (various α)
Fixed conditions: Batch size B=32, FCFS scheduling, λ=10 req/s
Analysis: Robustness of optimal batch sizes across different workload characteristics

Based on report:
- Heavy-tailed distributions (PowerLaw) increase variance and tail latency
- LogNormal captures real-world LLM workload patterns
- Distribution affects optimal batch size selection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiments.config import SimulationConfig
from src.experiments.runner import SimulationRunner
import matplotlib.pyplot as plt
import json
import numpy as np


def run_experiment():
    """Run Experiment 4: Workload Distribution Sensitivity."""
    print("=" * 60)
    print("Experiment 4: Workload Distribution Sensitivity")
    print("=" * 60)
    print("\nComparing: Uniform, LogNormal (σ=0.5, 1.0, 1.5), PowerLaw (α=1.5, 2.0, 2.5)")
    
    # Define distributions to test (calibrated for realistic workloads)
    distributions = [
        ("Uniform", "uniform", {"min": 10, "max": 200}),
        ("LogNormal(σ=0.5)", "lognormal", {"mu": 3.5, "sigma": 0.5}),
        ("LogNormal(σ=0.8)", "lognormal", {"mu": 3.5, "sigma": 0.8}),
        ("LogNormal(σ=1.2)", "lognormal", {"mu": 3.5, "sigma": 1.2}),
        ("PowerLaw(α=1.8)", "powerlaw", {"alpha": 1.8, "min": 10, "max": 500}),
        ("PowerLaw(α=2.2)", "powerlaw", {"alpha": 2.2, "min": 10, "max": 500}),
        ("PowerLaw(α=2.5)", "powerlaw", {"alpha": 2.5, "min": 10, "max": 500}),
    ]
    
    results_by_dist = {}
    
    for name, dist_type, dist_params in distributions:
        print(f"\n--- Testing distribution: {name} ---")
        
        config = SimulationConfig(
            arrival_rate=10.0,
            arrival_pattern="constant",
            batch_size=32,
            scheduling_policy="FCFS",
            prompt_length_dist=dist_type,
            prompt_dist_params=dist_params,
            num_requests=5000,
            warmup_requests=250,
            random_seed=42,
        )
        
        # Run simulation
        runner = SimulationRunner(config)
        metrics = runner.run()
        results_by_dist[name] = metrics
        
        # Print key metrics
        print(f"  Avg Latency: {metrics.get('avg_latency', 0):.3f}s")
        print(f"  P99 Latency: {metrics.get('p99_latency', 0):.3f}s")
        print(f"  Std Latency: {metrics.get('std_latency', 0):.3f}s")
        print(f"  Avg Prompt Length: {metrics.get('avg_prompt_length', 0):.1f} tokens")
    
    # Save results
    output_file = "experiment4_results.json"
    with open(output_file, 'w') as f:
        serializable_results = {
            k: {
                "avg_latency": v.get("avg_latency", 0),
                "p50_latency": v.get("p50_latency", 0),
                "p95_latency": v.get("p95_latency", 0),
                "p99_latency": v.get("p99_latency", 0),
                "std_latency": v.get("std_latency", 0),
                "avg_prompt_length": v.get("avg_prompt_length", 0),
                "throughput_req_per_sec": v.get("throughput_req_per_sec", 0),
            }
            for k, v in results_by_dist.items()
        }
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Plot results
    plot_results(results_by_dist)
    
    # Print comparison
    print_comparison(results_by_dist)


def plot_results(results_by_dist):
    """Plot distribution sensitivity results."""
    dists = list(results_by_dist.keys())
    x = np.arange(len(dists))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Latency Percentiles
    ax = axes[0, 0]
    p50 = [results_by_dist[d].get('p50_latency', 0) for d in dists]
    p95 = [results_by_dist[d].get('p95_latency', 0) for d in dists]
    p99 = [results_by_dist[d].get('p99_latency', 0) for d in dists]
    
    width = 0.25
    ax.bar(x - width, p50, width, label='P50', color='green', alpha=0.7)
    ax.bar(x, p95, width, label='P95', color='orange', alpha=0.7)
    ax.bar(x + width, p99, width, label='P99', color='red', alpha=0.7)
    ax.set_ylabel('Latency (s)')
    ax.set_title('Latency Percentiles by Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(dists, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Latency Variance
    ax = axes[0, 1]
    std_lat = [results_by_dist[d].get('std_latency', 0) for d in dists]
    ax.bar(x, std_lat, color='purple', alpha=0.7)
    ax.set_ylabel('Latency Std Dev (s)')
    ax.set_title('Latency Variability by Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(dists, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: P99/P50 Ratio (tail heaviness)
    ax = axes[1, 0]
    ratio = [results_by_dist[d].get('p99_latency', 1) / max(results_by_dist[d].get('p50_latency', 1), 0.001) 
             for d in dists]
    ax.bar(x, ratio, color='coral', alpha=0.7)
    ax.set_ylabel('P99/P50 Ratio')
    ax.set_title('Tail Heaviness by Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(dists, rotation=45, ha='right')
    ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Good (ratio < 2)')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Poor (ratio > 5)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Throughput
    ax = axes[1, 1]
    throughput = [results_by_dist[d].get('throughput_req_per_sec', 0) for d in dists]
    ax.bar(x, throughput, color='steelblue', alpha=0.7)
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('Throughput by Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(dists, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiment4_workload_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Plots saved to experiment4_workload_distribution.png")


def print_comparison(results_by_dist):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("WORKLOAD DISTRIBUTION COMPARISON")
    print("=" * 90)
    print(f"{'Distribution':<20} {'Avg Lat':<10} {'P99 Lat':<10} {'Std Lat':<10} {'P99/P50':<10} {'Throughput':<10}")
    print("-" * 90)
    
    for name, metrics in results_by_dist.items():
        avg = metrics.get('avg_latency', 0)
        p99 = metrics.get('p99_latency', 0)
        std = metrics.get('std_latency', 0)
        p50 = max(metrics.get('p50_latency', 1), 0.001)
        ratio = p99 / p50
        throughput = metrics.get('throughput_req_per_sec', 0)
        
        print(f"{name:<20} {avg:<10.3f} {p99:<10.3f} {std:<10.3f} {ratio:<10.2f} {throughput:<10.2f}")
    
    print("=" * 90)
    print("\nKey Findings:")
    
    # Find distribution with highest variance
    max_std_dist = max(results_by_dist.items(), key=lambda x: x[1].get('std_latency', 0))
    print(f"  - Highest variance: {max_std_dist[0]} (σ = {max_std_dist[1].get('std_latency', 0):.3f}s)")
    
    # Find distribution with lowest p99
    min_p99_dist = min(results_by_dist.items(), key=lambda x: x[1].get('p99_latency', float('inf')))
    print(f"  - Best tail latency: {min_p99_dist[0]} (p99 = {min_p99_dist[1].get('p99_latency', 0):.3f}s)")


if __name__ == "__main__":
    run_experiment()


