# LLM Inference Simulation

A discrete-event simulation framework for analyzing LLM inference systems with static and adaptive batching, built with SimPy. This project models the performance characteristics of systems like vLLM and Orca, focusing on batching strategies and scheduling policies.

## Project Overview

This simulation framework implements a two-phase LLM inference model based on the INDENG 174 Final Report:

1. **Prefill Phase**: Parallel processing of input prompts (compute-bound)
   - Formula: `T_prefill = α × Σ(l_prompt_i) + β`
2. **Decode Phase**: Autoregressive token generation (memory-bound)
   - Formula: `T_decode = γ × max(l_output_i)`

### Key Features

- **Multiple Scheduling Policies**: FCFS, SJF, Predicted-SJF, Priority with Aging
- **Adaptive Batching**: Dynamic batch size adjustment based on queue depth
- **Flexible Workload Generation**: LogNormal, TruncatedNormal, PowerLaw, Bimodal distributions
- **Non-stationary Arrival Patterns**: Constant, step, and sinusoidal arrival rates
- **Comprehensive Metrics**: Latency percentiles, throughput, fairness index, starvation rate
- **Statistical Rigor**: Built-in support for multiple replications with 95% confidence intervals

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip

### Setup

1. Clone or navigate to the repository:
```bash
cd llm-infer
```

2. Create and activate conda environment:
```bash
conda activate llm-infer
```

3. Or install dependencies with pip:
```bash
pip install -r requirements.txt
```

## Project Structure

```
llm-infer/
├── src/
│   ├── simulation/              # Core simulation components
│   │   ├── request.py           # Request data structure
│   │   ├── request_generator.py # Workload generation
│   │   ├── batch_processor.py   # Two-phase processing model
│   │   ├── llm_server.py        # Main server simulation
│   │   └── adaptive_server.py   # Adaptive batching server
│   ├── scheduling/              # Scheduling policies
│   │   └── policies.py          # FCFS, SJF, Predicted-SJF, Priority
│   ├── metrics/                 # Metrics collection and analysis
│   │   └── collector.py         # MetricsCollector class
│   └── experiments/             # Experiment configuration and runners
│       ├── config.py            # Configuration dataclasses
│       └── runner.py            # Simulation and experiment runners
├── examples/                    # Experiment scripts
│   ├── simple_example.py        # Basic usage example
│   ├── experiment1_batch_size.py
│   ├── experiment2_scheduling_policies.py
│   ├── experiment3_load_stress.py
│   ├── experiment4_workload_distribution.py
│   └── experiment5_adaptive_batching.py
├── llm-infer-report/            # LaTeX report files
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

Run a simple simulation:

```bash
conda activate llm-infer
python examples/simple_example.py
```

This will run a basic simulation with default parameters and display key metrics.

## Usage

### Basic Simulation

```python
from src.experiments.config import SimulationConfig
from src.experiments.runner import SimulationRunner

# Create configuration
config = SimulationConfig(
    arrival_rate=10.0,          # 10 requests per second
    batch_size=32,              # Batch size of 32
    scheduling_policy="FCFS",   # First-Come-First-Serve
    num_requests=10000,         # Generate 10000 requests
    warmup_requests=500,        # Skip first 500 for warmup
)

# Run simulation
runner = SimulationRunner(config)
metrics = runner.run()

# Access results
print(f"Average latency: {metrics['avg_latency']:.3f}s")
print(f"P99 latency: {metrics['p99_latency']:.3f}s")
print(f"Throughput: {metrics['throughput_req_per_sec']:.2f} req/s")
```

### Running Experiments

```python
from src.experiments.config import SimulationConfig, ExperimentConfig
from src.experiments.runner import ExperimentRunner

# Create base configuration
base_config = SimulationConfig(
    arrival_rate=10.0,
    batch_size=32,
    scheduling_policy="FCFS",
)

# Create experiment with multiple replications
experiment = ExperimentConfig(
    name="example_experiment",
    description="Example experiment",
    base_config=base_config,
    num_replications=10,
)

# Run all replications
runner = ExperimentRunner(experiment.get_replication_configs())
results = runner.run_all()

# Aggregate results with confidence intervals
aggregated = runner.aggregate_results(results)
print(f"Mean latency: {aggregated['avg_latency']['mean']:.3f}s")
print(f"95% CI: [{aggregated['avg_latency']['ci_lower']:.3f}, "
      f"{aggregated['avg_latency']['ci_upper']:.3f}]")
```

## Experiments

The project includes five experiments as described in the INDENG 174 Final Report:

### Experiment 1: Batch Size Sensitivity Analysis
- **Parameters**: Batch sizes ∈ {1, 2, 4, 8, 16, 32, 64, 128}
- **Fixed conditions**: λ = 10 req/s, FCFS scheduling
- **Script**: `examples/experiment1_batch_size.py`
- **Key Findings**: 
  - Optimal batch size: **16-32**
  - B ≤ 8 causes queue explosion (876s latency at B=1)
  - B ≥ 16 achieves stable throughput (~9.9 req/s)

### Experiment 2: Scheduling Policy Comparison
- **Policies**: FCFS, SJF, Predicted-SJF, Priority with Aging
- **Workload**: Bimodal (70% short 10-50 tokens, 30% long 200-500 tokens)
- **Fixed conditions**: λ = 18 req/s, B = 32
- **Script**: `examples/experiment2_scheduling_policies.py`
- **Key Findings**:
  - SJF reduces avg latency by **37%** (9.15s vs 14.65s)
  - But fairness drops dramatically: **0.87 → 0.09** (Jain's Index)
  - **7%** starvation rate with SJF
  - Priority with aging: balanced (0.84 fairness, 6% latency reduction)

### Experiment 3: Load Stress Testing
- **Scenario**: Gradual load increase (λ = 1 to 50 req/s)
- **Fixed conditions**: B = 32, FCFS
- **Script**: `examples/experiment3_load_stress.py`
- **Key Findings**:
  - System saturates at **~20 req/s**
  - Recommended max operating rate: **16 req/s** (80% safety margin)
  - Queue explosion occurs beyond saturation point

### Experiment 4: Workload Distribution Sensitivity
- **Distributions**: Uniform, LogNormal (σ=0.5, 1.0, 1.5), PowerLaw (α=1.5, 2.0, 2.5)
- **Script**: `examples/experiment4_workload_distribution.py`
- **Key Findings**: Heavy-tailed distributions increase tail latency significantly

### Experiment 5: Adaptive Batching under Traffic Spikes
- **Scenario**: Normal load (10 req/s) → Spike (25 req/s) → Recovery
- **Comparison**: Static B=32 vs Adaptive (B=32→64 during spikes)
- **Script**: `examples/experiment5_adaptive_batching.py`
- **Key Findings**:
  - Adaptive batching reduces aggregate latency by **54%**
  - Max queue depth reduced by **84%**
  - B=64 handles spikes well (3.1s vs 9.9s latency)

## Configuration Options

### Timing Model Parameters (Calibrated)

Based on realistic LLM inference characteristics:
- `α = 0.00015` s/token (prefill per-token time)
- `β = 0.008` s (prefill overhead)
- `γ = 0.010` s/step (decode step time)

These values result in a saturation point of ~20 req/s at B=32.

### Arrival Patterns
- `constant`: Constant arrival rate
- `step`: Step function (base → step rate at time t)
- `sinusoidal`: Sinusoidal variation (hourly cycles)

### Request Length Distributions
- **Prompt lengths**: `lognormal` (μ=3.5, σ=0.8), `uniform`, `powerlaw`, `bimodal`
- **Output lengths**: `truncated_normal` (μ=80, σ=25), `uniform`, `lognormal`

### Scheduling Policies
| Policy | Description | Latency | Fairness |
|--------|-------------|---------|----------|
| `FCFS` | First-Come-First-Serve | Baseline | High (0.87) |
| `SJF` | Shortest Job First | -37% | Low (0.09) |
| `Predicted-SJF` | Estimated processing time | -38% | Low (0.09) |
| `Priority` | Priority with aging | -6% | High (0.84) |

## Metrics

The simulation collects comprehensive metrics:

### Latency Metrics
- Average, median, min, max latency
- P50, P95, P99 percentiles
- Queue wait time / Processing time

### Throughput Metrics
- Requests per second
- Tokens per second

### Fairness Metrics
- **Jain's Fairness Index**: (Σxᵢ)² / (n × Σxᵢ²), values near 1 indicate perfect fairness
- **Starvation Rate**: Percentage of requests with excessive wait times (>10× median)

### System Metrics
- Max queue length
- Average batch size
- Total batches processed

## Report

The full technical report is available in `llm-infer-report/`. To compile:

```bash
cd llm-infer-report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## References

This project implements concepts from:

- Kwon et al. (2023): "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)
- Yu et al. (2022): "Orca: A Distributed Serving System for Transformer-Based Generative Models"
- Agrawal et al. (2024): "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"
- Sheng et al. (2024): "Fairness in Serving Large Language Models"
- Law (2015): "Simulation Modeling and Analysis"

## Authors

**INDENG 174 Group 9**
- Runyuan He
- Jiedong Zhang
- Qingyang Xu

## License

This project is for academic purposes as part of INDENG 174 coursework at UC Berkeley.
