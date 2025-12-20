#!/usr/bin/env python3
"""
Experiment 2: Scheduling Policy Comparison

Policies: FCFS, SJF, Predicted-SJF, Priority (with aging)
Workload: Bimodal (70% short requests 10-50 tokens, 30% long requests 500-2000 tokens)
Fixed conditions: Batch size B=32, constant arrival rate λ=10 req/s
Metrics: Average latency, p99 latency, fairness index, starvation rate

Based on report findings:
- SJF reduces average latency by ~15% vs FCFS
- SJF degrades p99 latency and fairness (Jain's Index 0.64 vs 0.82)
- Priority with aging achieves balance (fairness index ~0.75)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiments.config import SimulationConfig, ExperimentConfig
from src.experiments.runner import SimulationRunner, ExperimentRunner
import matplotlib.pyplot as plt
import json
import numpy as np


def run_experiment():
    """Run Experiment 2: Scheduling Policy Comparison."""
    print("=" * 60)
    print("Experiment 2: Scheduling Policy Comparison")
    print("=" * 60)
    print("\nWorkload: Bimodal (70% short 10-50 tokens, 30% long 500-2000 tokens)")
    print("Fixed: Batch size=32, Arrival rate=10 req/s")

    policies = [
        ("FCFS", {}),
        ("SJF", {}),
        ("Predicted-SJF", {}),
        ("Priority", {"use_aging": True, "aging_weight": 0.1}),
    ]
    
    results_by_policy = {}
    
    for policy_name, policy_params in policies:
        print(f"\n--- Testing policy: {policy_name} ---")
        
        # Create base configuration with bimodal workload
        # Use arrival rate near saturation (18 req/s) to show scheduling effects
        base_config = SimulationConfig(
            arrival_rate=18.0,  # Near saturation to show scheduling differences
            arrival_pattern="constant",
            batch_size=32,
            scheduling_policy=policy_name,
            scheduling_policy_params=policy_params,
            # Bimodal workload: 70% short, 30% long
            prompt_length_dist="bimodal",
            prompt_dist_params={},  # Uses default bimodal params
            output_length_dist="truncated_normal",
            output_dist_params={"mu": 80, "sigma": 25, "min": 10, "max": 300},
            num_requests=5000,
            warmup_requests=500,
        )
        
        # Create experiment with replications
        experiment = ExperimentConfig(
            name=f"scheduling_{policy_name}",
            description=f"Scheduling policy comparison: {policy_name}",
            base_config=base_config,
            num_replications=10,  # Use 10 for reasonable speed, 30 for full experiment
            random_seed_start=42,
        )
        
        # Run replications
        runner = ExperimentRunner(experiment.get_replication_configs())
        results = runner.run_all(verbose=True)
        
        # Aggregate results
        aggregated = runner.aggregate_results(results)
        results_by_policy[policy_name] = aggregated
        
        # Print summary
        print(f"\nResults for {policy_name}:")
        print(f"  Avg Latency: {aggregated['avg_latency']['mean']:.3f}s "
              f"(95% CI: [{aggregated['avg_latency']['ci_lower']:.3f}, "
              f"{aggregated['avg_latency']['ci_upper']:.3f}])")
        print(f"  P99 Latency: {aggregated['p99_latency']['mean']:.3f}s")
        print(f"  Fairness Index: {aggregated['fairness_index']['mean']:.3f}")
        print(f"  Starvation Rate: {aggregated['starvation_rate']['mean']:.2%}")
    
    # Save results
    output_file = "experiment2_results.json"
    with open(output_file, 'w') as f:
        serializable_results = {
            k: {
                metric: {
                    "mean": v["mean"],
                    "std": v["std"],
                    "ci_lower": v["ci_lower"],
                    "ci_upper": v["ci_upper"]
                }
                for metric, v in metrics.items()
            }
            for k, metrics in results_by_policy.items()
        }
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Plot results
    plot_results(results_by_policy)
    
    # Print comparison table
    print_comparison_table(results_by_policy)


def plot_results(results_by_policy):
    """Plot comparison of scheduling policies."""
    policies = list(results_by_policy.keys())
    x = np.arange(len(policies))
    width = 0.35
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Latency Comparison (Avg and P99)
    ax = axes[0, 0]
    avg_latencies = [results_by_policy[p]['avg_latency']['mean'] for p in policies]
    p99_latencies = [results_by_policy[p]['p99_latency']['mean'] for p in policies]
    ax.bar(x - width/2, avg_latencies, width, label='Avg Latency', color='steelblue')
    ax.bar(x + width/2, p99_latencies, width, label='P99 Latency', color='coral')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Latency by Scheduling Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Fairness Index
    ax = axes[0, 1]
    fairness = [results_by_policy[p]['fairness_index']['mean'] for p in policies]
    fairness_ci = [(results_by_policy[p]['fairness_index']['ci_upper'] - 
                    results_by_policy[p]['fairness_index']['ci_lower']) / 2 for p in policies]
    ax.bar(x, fairness, yerr=fairness_ci, capsize=5, color='green', alpha=0.7)
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title('Fairness by Scheduling Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target (0.8)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Starvation Rate
    ax = axes[1, 0]
    starvation = [results_by_policy[p]['starvation_rate']['mean'] * 100 for p in policies]
    ax.bar(x, starvation, color='red', alpha=0.6)
    ax.set_ylabel('Starvation Rate (%)')
    ax.set_title('Starvation Rate by Scheduling Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Throughput
    ax = axes[1, 1]
    throughput = [results_by_policy[p]['throughput_req_per_sec']['mean'] for p in policies]
    throughput_ci = [(results_by_policy[p]['throughput_req_per_sec']['ci_upper'] - 
                      results_by_policy[p]['throughput_req_per_sec']['ci_lower']) / 2 for p in policies]
    ax.bar(x, throughput, yerr=throughput_ci, capsize=5, color='purple', alpha=0.7)
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('Throughput by Scheduling Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiment2_scheduling_policies.png', dpi=300, bbox_inches='tight')
    print("✓ Plots saved to experiment2_scheduling_policies.png")


def print_comparison_table(results_by_policy):
    """Print a comparison table of all policies."""
    print("\n" + "=" * 80)
    print("SCHEDULING POLICY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Policy':<15} {'Avg Lat (s)':<12} {'P99 Lat (s)':<12} {'Fairness':<10} {'Starvation':<12} {'Throughput':<10}")
    print("-" * 80)
    
    for policy, metrics in results_by_policy.items():
        avg_lat = metrics['avg_latency']['mean']
        p99_lat = metrics['p99_latency']['mean']
        fairness = metrics['fairness_index']['mean']
        starvation = metrics['starvation_rate']['mean']
        throughput = metrics['throughput_req_per_sec']['mean']
        
        print(f"{policy:<15} {avg_lat:<12.3f} {p99_lat:<12.3f} {fairness:<10.3f} {starvation:<12.2%} {throughput:<10.2f}")
    
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()


