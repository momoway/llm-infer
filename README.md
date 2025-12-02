# LLM Inference Simulation

A discrete-event simulation framework for analyzing LLM inference systems with continuous batching, built with SimPy. This project models the performance characteristics of systems like vLLM and SGLang, focusing on batching strategies and scheduling policies.

## Project Overview

This simulation framework implements a two-phase LLM inference model:

1. **Prefill Phase**: Parallel processing of input prompts (compute-bound)
2. **Decode Phase**: Autoregressive token generation (memory-bound)

### Key Features

- **Multiple Scheduling Policies**: FCFS, SJF, Predicted-SJF, Priority-based
- **Flexible Workload Generation**: LogNormal, TruncatedNormal, PowerLaw, Bimodal distributions
- **Non-stationary Arrival Patterns**: Constant, step, and sinusoidal arrival rates
- **Comprehensive Metrics**: Latency percentiles, throughput, fairness index, starvation rate
- **Statistical Rigor**: Built-in support for 30+ replications with 95% confidence intervals

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or navigate to the repository:
```bash
cd llm-infer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
llm-infer/
├── src/
│   ├── simulation/          # Core simulation components
│   │   ├── request.py       # Request data structure
│   │   ├── request_generator.py  # Workload generation
│   │   ├── batch_processor.py    # Two-phase processing model
│   │   └── llm_server.py    # Main server simulation
│   ├── scheduling/          # Scheduling policies
│   │   └── policies.py      # FCFS, SJF, Predicted-SJF, Priority
│   ├── metrics/             # Metrics collection and analysis
│   │   └── collector.py     # MetricsCollector class
│   └── experiments/         # Experiment configuration and runners
│       ├── config.py        # Configuration dataclasses
│       └── runner.py        # Simulation and experiment runners
├── examples/                # Example scripts
│   ├── simple_example.py    # Basic usage example
│   └── experiment1_batch_size.py  # Batch size sensitivity analysis
├── tex/                     # LaTeX report files
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Quick Start

Run a simple simulation:

```bash
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

The framework supports multi-replication experiments for statistical rigor:

```python
from src.experiments.config import SimulationConfig, ExperimentConfig
from src.experiments.runner import ExperimentRunner

# Create base configuration
base_config = SimulationConfig(
    arrival_rate=10.0,
    batch_size=32,
    scheduling_policy="FCFS",
)

# Create experiment with 30 replications
experiment = ExperimentConfig(
    name="example_experiment",
    description="Example experiment",
    base_config=base_config,
    num_replications=30,
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

## Planned Experiments

The project includes five planned experiments (see [tex/experiments.tex](tex/experiments.tex)):

### Experiment 1: Batch Size Sensitivity Analysis
- **Parameters**: Batch sizes ∈ {1, 2, 4, 8, 16, 32, 64, 128}
- **Fixed conditions**: λ = 10 req/s, FCFS scheduling
- **Script**: `examples/experiment1_batch_size.py`

### Experiment 2: Scheduling Policy Comparison
- **Policies**: FCFS, SJF, Predicted-SJF, Priority
- **Workload**: Bimodal (70% short, 30% long requests)
- **Metrics**: Fairness index, starvation rate

### Experiment 3: Load Stress Testing
- **Scenario**: Gradual load increase (λ = 1 to 50 req/s)
- **Goal**: Identify saturation point and early warning indicators

### Experiment 4: Workload Distribution Sensitivity
- **Distributions**: Uniform, LogNormal (various σ), PowerLaw (various α)
- **Analysis**: Robustness of optimal batch sizes

### Experiment 5: Time-Varying Load Patterns
- **Pattern**: Sinusoidal λ(t) = 10 + 8sin(2πt/3600)
- **Strategies**: Static vs adaptive batch sizing

## Configuration Options

### Arrival Patterns
- `constant`: Constant arrival rate
- `step`: Step function (base → step rate at time t)
- `sinusoidal`: Sinusoidal variation (hourly cycles)

### Request Distributions
- **Prompt lengths**: `lognormal`, `uniform`, `powerlaw`, `bimodal`
- **Output lengths**: `truncated_normal`, `uniform`, `lognormal`

### Scheduling Policies
- `FCFS`: First-Come-First-Serve (arrival time)
- `SJF`: Shortest Job First (prompt length)
- `Predicted-SJF`: Estimated processing time
- `Priority`: Priority-based with aging

### Timing Model Parameters

Based on realistic LLM inference characteristics:
- `alpha = 0.001` s/token (prefill per-token time)
- `beta = 0.05` s (prefill overhead)
- `gamma = 0.0005` s/step (decode step time)

## Metrics

The simulation collects comprehensive metrics:

### Latency Metrics
- Average, median, min, max latency
- P50, P95, P99 percentiles
- Queue wait time
- Processing time

### Throughput Metrics
- Requests per second
- Tokens per second
- GPU utilization (via batch statistics)

### Fairness Metrics
- Jain's fairness index
- Starvation rate (requests with excessive wait times)

### System Metrics
- Max queue length
- Average batch size
- Total batches processed

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ examples/
flake8 src/ examples/
```

## References

This project implements concepts from:

- Kwon et al. (2023): "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)
- Zheng et al. (2024): "SGLang: Efficient Execution of Structured Language Model Programs"
- Jain et al. (1984): "A Quantitative Measure Of Fairness And Discrimination For Resource Allocation In Shared Computer Systems"

## Authors

**INDENG 174 Group 9**
- Runyuan He (3041920716)
- Jiedong Zhang (3041913865)
- Qingyang Xu (3041979645)

## License

This project is for academic purposes as part of INDENG 174 coursework.
