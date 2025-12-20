#!/usr/bin/env python3
"""
Experiment 3: Load Stress Testing

Scenario: Gradual load increase from λ = 1 to 50 req/s
Fixed conditions: Batch size B=32, FCFS scheduling
Goal: Identify system saturation point and early warning indicators

Based on report findings:
- System maintains stability up to ~18 req/s
- Beyond saturation, queue grows linearly to infinity
- Latency degrades exponentially past saturation point
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
    """Run Experiment 3: Load Stress Testing."""
    print("=" * 60)
    print("Experiment 3: Load Stress Testing")
    print("=" * 60)
    print("\nTesting arrival rates from 1 to 50 req/s")
    print("Fixed: Batch size=32, FCFS scheduling")
    
    # Test a range of arrival rates
    arrival_rates = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
    results_by_rate = {}
    
    for rate in arrival_rates:
        print(f"\n--- Testing arrival rate: {rate} req/s ---")
        
        # For higher rates, we need more requests to see the effect
        num_requests = min(5000, int(rate * 500))  # Scale with rate, cap at 5000
        
        config = SimulationConfig(
            arrival_rate=float(rate),
            arrival_pattern="constant",
            batch_size=32,
            scheduling_policy="FCFS",
            num_requests=num_requests,
            warmup_requests=min(200, num_requests // 5),
            random_seed=42,
        )
        
        # Run simulation
        runner = SimulationRunner(config)
        metrics = runner.run()
        results_by_rate[rate] = metrics
        
        # Print key metrics
        print(f"  Avg Latency: {metrics.get('avg_latency', 0):.3f}s")
        print(f"  P99 Latency: {metrics.get('p99_latency', 0):.3f}s")
        print(f"  Max Queue Length: {metrics.get('max_queue_length', 0)}")
        print(f"  Throughput: {metrics.get('throughput_req_per_sec', 0):.2f} req/s")
        
        # Early warning: if queue is growing very large, system is saturating
        if metrics.get('max_queue_length', 0) > 1000:
            print(f"  ⚠️  WARNING: Queue explosion detected!")
    
    # Save results
    output_file = "experiment3_results.json"
    with open(output_file, 'w') as f:
        serializable_results = {
            str(k): {
                "arrival_rate": k,
                "avg_latency": v.get("avg_latency", 0),
                "p99_latency": v.get("p99_latency", 0),
                "max_queue_length": v.get("max_queue_length", 0),
                "throughput_req_per_sec": v.get("throughput_req_per_sec", 0),
                "avg_queue_wait": v.get("avg_queue_wait", 0),
            }
            for k, v in results_by_rate.items()
        }
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Plot results
    plot_results(results_by_rate)
    
    # Find saturation point
    find_saturation_point(results_by_rate)


def plot_results(results_by_rate):
    """Plot stress test results."""
    rates = sorted(results_by_rate.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Average and P99 Latency
    ax = axes[0, 0]
    avg_latencies = [results_by_rate[r].get('avg_latency', 0) for r in rates]
    p99_latencies = [results_by_rate[r].get('p99_latency', 0) for r in rates]
    ax.plot(rates, avg_latencies, 'o-', label='Avg Latency', linewidth=2)
    ax.plot(rates, p99_latencies, 's--', label='P99 Latency', linewidth=2)
    ax.set_xlabel('Arrival Rate (req/s)')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Latency vs Arrival Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see degradation
    
    # Plot 2: Max Queue Length
    ax = axes[0, 1]
    queue_lengths = [results_by_rate[r].get('max_queue_length', 0) for r in rates]
    ax.plot(rates, queue_lengths, 'o-', color='red', linewidth=2)
    ax.set_xlabel('Arrival Rate (req/s)')
    ax.set_ylabel('Max Queue Length')
    ax.set_title('Queue Buildup vs Arrival Rate')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Warning threshold')
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
    ax.legend()
    
    # Plot 3: Throughput
    ax = axes[1, 0]
    throughputs = [results_by_rate[r].get('throughput_req_per_sec', 0) for r in rates]
    ax.plot(rates, throughputs, 'o-', color='green', linewidth=2, label='Actual Throughput')
    ax.plot(rates, rates, 'k--', alpha=0.5, label='Ideal (λ = throughput)')
    ax.set_xlabel('Arrival Rate (req/s)')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('Throughput vs Arrival Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Queue Wait Time
    ax = axes[1, 1]
    wait_times = [results_by_rate[r].get('avg_queue_wait', 0) for r in rates]
    ax.plot(rates, wait_times, 'o-', color='purple', linewidth=2)
    ax.set_xlabel('Arrival Rate (req/s)')
    ax.set_ylabel('Avg Queue Wait Time (s)')
    ax.set_title('Queue Wait Time vs Arrival Rate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment3_load_stress.png', dpi=300, bbox_inches='tight')
    print("✓ Plots saved to experiment3_load_stress.png")


def find_saturation_point(results_by_rate):
    """Identify the saturation point based on queue growth."""
    print("\n" + "=" * 60)
    print("SATURATION ANALYSIS")
    print("=" * 60)
    
    rates = sorted(results_by_rate.keys())
    
    # Find the rate where queue starts growing significantly
    prev_queue = 0
    saturation_rate = None
    
    for rate in rates:
        queue_len = results_by_rate[rate].get('max_queue_length', 0)
        throughput = results_by_rate[rate].get('throughput_req_per_sec', 0)
        
        # Saturation indicators:
        # 1. Queue length > 100
        # 2. Throughput significantly below arrival rate
        if queue_len > 100 and saturation_rate is None:
            saturation_rate = rate
            print(f"\n⚠️  Saturation detected at λ = {rate} req/s")
            print(f"   Queue length: {queue_len}")
            print(f"   Throughput: {throughput:.2f} req/s (expected: {rate})")
        
        prev_queue = queue_len
    
    if saturation_rate:
        # Find the safe operating point (80% of saturation)
        safe_rate = saturation_rate * 0.8
        print(f"\n📊 Recommended max operating rate: {safe_rate:.1f} req/s")
        print(f"   (80% of saturation point for safety margin)")
    else:
        print("\n✅ No saturation detected in tested range")
        print("   System can handle up to 50 req/s with B=32")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_experiment()


