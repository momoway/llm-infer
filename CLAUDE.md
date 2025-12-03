# CLAUDE.md - AI Assistant Guide for LLM Inference Simulation

**Project:** LLM Inference Simulation Framework
**Version:** 0.1.0
**Type:** Academic Research Project (INDENG 174)
**Last Updated:** 2025-12-03

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Architecture](#core-architecture)
4. [Development Workflows](#development-workflows)
5. [Code Conventions](#code-conventions)
6. [Key Components Guide](#key-components-guide)
7. [Configuration System](#configuration-system)
8. [Testing & Quality](#testing--quality)
9. [Common Tasks](#common-tasks)
10. [Important Constraints](#important-constraints)

---

## Project Overview

### Purpose
A discrete-event simulation framework for analyzing LLM inference systems with continuous batching, built with SimPy. Models performance characteristics of systems like vLLM and SGLang.

### Key Characteristics
- **Simulation Type:** Discrete-event simulation (SimPy-based)
- **Focus:** Two-phase LLM inference (prefill + decode)
- **Use Case:** Research on batching strategies and scheduling policies
- **Statistical Rigor:** 30+ replications with 95% confidence intervals
- **Academic Context:** INDENG 174 coursework (Group 9)

### Core Features
1. **Multiple Scheduling Policies:** FCFS, SJF, Predicted-SJF, Priority-based
2. **Flexible Workload Generation:** LogNormal, TruncatedNormal, PowerLaw, Bimodal
3. **Non-stationary Arrivals:** Constant, step, sinusoidal patterns
4. **Comprehensive Metrics:** Latency, throughput, fairness, starvation
5. **Experiment Framework:** Multi-replication with statistical analysis

---

## Repository Structure

```
llm-infer/
├── .git/                      # Git repository metadata
├── .gitignore                 # Python, IDE, results, LaTeX exclusions
├── .gitmodules                # Submodule: llm-infer-report
├── README.md                  # User-facing documentation
├── CLAUDE.md                  # This file - AI assistant guide
├── requirements.txt           # Python dependencies
│
├── src/                       # Core application code (~910 lines)
│   ├── __init__.py           # Package init (version 0.1.0)
│   │
│   ├── simulation/            # Core simulation (522 lines)
│   │   ├── __init__.py
│   │   ├── request.py        # Request dataclass (54 lines)
│   │   ├── request_generator.py  # Workload generation (164 lines)
│   │   ├── batch_processor.py    # Two-phase processing (126 lines)
│   │   └── llm_server.py     # Main server simulation (179 lines)
│   │
│   ├── scheduling/            # Scheduling policies (174 lines)
│   │   ├── __init__.py
│   │   └── policies.py       # FCFS, SJF, Predicted-SJF, Priority
│   │
│   ├── metrics/               # Metrics collection (213 lines)
│   │   ├── __init__.py
│   │   └── collector.py      # MetricsCollector class
│   │
│   └── experiments/           # Experiment infrastructure (267 lines)
│       ├── __init__.py
│       ├── config.py         # Configuration dataclasses (70 lines)
│       └── runner.py         # Simulation/experiment runners (197 lines)
│
├── examples/                  # Example scripts (233 lines)
│   ├── simple_example.py     # Basic usage demo (80 lines)
│   └── experiment1_batch_size.py  # Batch size analysis (153 lines)
│
└── llm-infer-report/          # Git submodule (external docs)
    └── [LaTeX report repository]
```

### Total Statistics
- **Total Lines of Code:** ~1,450 lines
- **Python Files:** 13 files
- **Packages:** 4 (simulation, scheduling, metrics, experiments)
- **Classes:** 13+ classes
- **Functions/Methods:** 50+ methods

---

## Core Architecture

### Layered Architecture

```
┌──────────────────────────────────────────┐
│  Application Layer                       │
│  - examples/simple_example.py            │
│  - examples/experiment1_batch_size.py    │
└──────────────┬───────────────────────────┘
               │ uses
               ▼
┌──────────────────────────────────────────┐
│  Experiment Layer                        │
│  - SimulationRunner                      │
│  - ExperimentRunner                      │
└──────────────┬───────────────────────────┘
               │ configures & orchestrates
               ▼
┌──────────────────────────────────────────┐
│  Service Layer                           │
│  - MetricsCollector                      │
│  - SchedulingPolicy (FCFS/SJF/etc)       │
└──────────────┬───────────────────────────┘
               │ supports
               ▼
┌──────────────────────────────────────────┐
│  Domain Layer                            │
│  - LLMInferenceServer                    │
│  - RequestGenerator                      │
│  - BatchProcessor                        │
└──────────────┬───────────────────────────┘
               │ operates on
               ▼
┌──────────────────────────────────────────┐
│  Data Layer                              │
│  - Request (dataclass)                   │
│  - SimulationConfig (dataclass)          │
│  - SimPy Environment                     │
└──────────────────────────────────────────┘
```

### Key Design Patterns

| Pattern | Usage | Location |
|---------|-------|----------|
| **Factory** | `get_policy(name)` creates scheduling policies | `src/scheduling/policies.py:155` |
| **Abstract Base Class** | `SchedulingPolicy` defines interface | `src/scheduling/policies.py:10` |
| **Dataclass** | Data containers (Request, Config) | `src/simulation/request.py:8`, `src/experiments/config.py:8` |
| **Strategy** | Pluggable scheduling algorithms | `src/scheduling/policies.py` |
| **Process-based** | SimPy generator processes with `yield` | Throughout simulation code |

### Core Data Flow

```
1. SimulationRunner creates:
   ├── SimPy Environment
   ├── RequestGenerator (with config)
   └── LLMInferenceServer (with config)

2. RequestGenerator yields requests:
   ├── Uses Poisson arrivals (exponential inter-arrival)
   ├── Samples prompt/output lengths from distributions
   └── Creates Request objects

3. LLMInferenceServer processes:
   ├── Queues incoming requests
   ├── Collects batches based on batch_size
   ├── Applies scheduling policy to sort requests
   ├── Sends batch to BatchProcessor
   └── Records timestamps in Request objects

4. BatchProcessor simulates GPU:
   ├── Prefill: T = α * Σ(prompt_lengths) + β
   ├── Decode: T = γ * max(output_lengths)
   └── Returns completed requests

5. MetricsCollector analyzes:
   ├── Filters warmup requests
   ├── Computes latency, throughput, fairness metrics
   └── Returns dictionary of results
```

---

## Development Workflows

### Git Workflow

**Current Branch:** `claude/claude-md-mipbh3eb1g560dwp-01LJfTcYnC35hAd2WLo6KLPo`
**Main Branch:** Not explicitly set (likely `main` or `master`)
**Remote:** `origin` → `http://local_proxy@127.0.0.1:45049/git/momoway/llm-infer`

#### Commit History
```
06c68a7 Add submodule              [Latest]
5857742 Bug fix
a250780 first commit                [Initial]
```

#### Branch Naming Convention
- Feature branches: `claude/claude-md-{session_id}`
- Must start with `claude/` and end with matching session ID
- Push failures with 403 indicate branch name mismatch

#### Git Best Practices for This Repo
1. Always push to branch specified in session context
2. Use `git push -u origin <branch-name>` for first push
3. Retry network failures up to 4 times with exponential backoff (2s, 4s, 8s, 16s)
4. For fetches/pulls, prefer specific branches: `git fetch origin <branch-name>`
5. Never push to main/master without explicit permission

### Submodule Management

**Submodule:** `llm-infer-report`
**URL:** `https://github.com/momoway/llm-infer-report.git`
**Status:** Initialized but directory currently empty
**Purpose:** External LaTeX documentation/reporting repository

```bash
# Initialize and update submodule
git submodule init
git submodule update

# Update to latest
git submodule update --remote
```

### Installation & Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd llm-infer

# 2. Initialize submodules
git submodule init
git submodule update

# 3. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python examples/simple_example.py
```

### Dependencies

```
# Core (Required)
simpy>=4.0.0              # Discrete-event simulation engine
numpy>=1.24.0             # Numerical computing, distributions
scipy>=1.10.0             # Statistical functions (t-distribution for CI)

# Visualization (Required for examples)
matplotlib>=3.7.0         # Plotting results

# Data Analysis (Optional)
pandas>=2.0.0             # Data manipulation

# Development (Optional)
pytest>=7.4.0             # Testing framework
black>=23.0.0             # Code formatter
flake8>=6.0.0             # Linter
```

### Running Simulations

#### Basic Simulation
```bash
# Run basic example
python examples/simple_example.py

# Output: Console metrics (avg latency, P99, throughput, etc.)
```

#### Running Experiments
```bash
# Run batch size sensitivity analysis
python examples/experiment1_batch_size.py

# Output:
# - experiment1_results.json (statistical results)
# - experiment1_batch_size.png (4-subplot visualization)
```

---

## Code Conventions

### Python Style

#### Naming Conventions
- **Classes:** PascalCase (`RequestGenerator`, `LLMInferenceServer`, `BatchProcessor`)
- **Functions/Methods:** snake_case (`collect_batch`, `compute_metrics`, `get_policy`)
- **Private Methods:** `_` prefix (`_compute_latency_metrics`, `_compute_fairness_index`)
- **Constants:** Class attributes in lowercase (`alpha = 0.001`, `beta = 0.05`, `gamma = 0.0005`)
- **Type Variables:** PascalCase (`Optional`, `Dict`, `List`, `Callable`, `Any`)

#### Type Hints
- **Required:** All function signatures must have type hints
- **Coverage:** Comprehensive throughout codebase
- **Imports:** `from typing import Optional, Dict, List, Callable, Any`

```python
# Good example from request.py
@property
def queue_wait_time(self) -> Optional[float]:
    """Time spent waiting in queue."""
    if self.queue_entry_time and self.processing_start_time:
        return self.processing_start_time - self.queue_entry_time
    return None
```

#### Docstrings
- **Style:** Google-style docstrings
- **Required Sections:** Description, Args, Returns, Raises (if applicable)
- **Coverage:** All public classes and methods

```python
# Good example from batch_processor.py
def process_batch(
    self,
    batch: List[Request],
    current_time: float
) -> float:
    """
    Simulate processing a batch of requests.

    Args:
        batch: List of Request objects to process
        current_time: Current simulation time

    Returns:
        Total processing time (prefill + decode)
    """
    # Implementation...
```

#### Code Formatting
- **Formatter:** black (line length: default 88)
- **Linter:** flake8
- **Commands:**
  ```bash
  black src/ examples/
  flake8 src/ examples/
  ```

### Dataclass Usage

**Primary Use:** Data containers (Request, SimulationConfig, ExperimentConfig)

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Request:
    """Core fields with defaults."""
    request_id: int
    arrival_time: float
    prompt_length: int
    expected_output_length: int

    # Optional fields with None default
    queue_entry_time: Optional[float] = None
    processing_start_time: Optional[float] = None

    # Computed properties using @property
    @property
    def total_latency(self) -> Optional[float]:
        if self.arrival_time and self.processing_end_time:
            return self.processing_end_time - self.arrival_time
        return None
```

**Key Patterns:**
1. Required fields first, optional fields with defaults after
2. Use `field(default_factory=dict)` for mutable defaults
3. Computed values as `@property` methods
4. Implement `__lt__` for sortable dataclasses

### SimPy Process Patterns

**Generator-based Processes:** Use `yield` for simulation events

```python
def run(self) -> None:
    """Main server loop (SimPy process)."""
    while True:
        # Wait for batch to fill or timeout
        if self.batch_timeout:
            try:
                yield self.env.timeout(self.batch_timeout)
            except simpy.Interrupt:
                pass

        # Collect and process batch
        if len(self.request_queue) > 0:
            batch = self.collect_batch()
            yield self.env.process(self._process_batch(batch))
```

**Key SimPy Operations:**
- `yield self.env.timeout(duration)` - Wait for time
- `yield self.env.process(process)` - Wait for subprocess
- `simpy.Interrupt` - Handle interruptions
- `self.env.now` - Get current simulation time

---

## Key Components Guide

### 1. Request (`src/simulation/request.py:8`)

**Purpose:** Data structure representing a single LLM inference request

**Fields:**
```python
# Required at creation
request_id: int               # Unique identifier
arrival_time: float           # When request arrived (seconds)
prompt_length: int            # Input tokens
expected_output_length: int   # Target output tokens

# Set during processing
queue_entry_time: Optional[float]      # When entered queue
processing_start_time: Optional[float] # When processing began
processing_end_time: Optional[float]   # When completed
actual_output_length: Optional[int]    # May differ from expected
```

**Computed Properties:**
- `queue_wait_time` - Time in queue before processing
- `processing_time` - Time spent in prefill + decode
- `total_latency` - End-to-end latency (arrival to completion)
- `total_tokens` - prompt_length + actual_output_length

**Sorting:** Implements `__lt__` for FCFS (by arrival_time)

### 2. RequestGenerator (`src/simulation/request_generator.py:12`)

**Purpose:** Generates workload with configurable distributions

**Initialization:**
```python
RequestGenerator(
    env: simpy.Environment,
    arrival_rate_fn: Callable[[float], float],  # λ(t) function
    prompt_length_dist: str = "lognormal",
    prompt_dist_params: Dict = {"mu": 4, "sigma": 1.5},
    output_length_dist: str = "truncated_normal",
    output_dist_params: Dict = {"mu": 100, "sigma": 30, "min": 10, "max": 500},
    random_seed: Optional[int] = None
)
```

**Supported Distributions:**

| Distribution | Parameters | Typical Use |
|--------------|------------|-------------|
| `lognormal` | mu, sigma, min (optional), max (optional) | Prompt lengths (skewed) |
| `uniform` | min, max | Simple baseline |
| `truncated_normal` | mu, sigma, min, max | Output lengths (bounded) |
| `powerlaw` | alpha, min, max | Heavy-tailed workloads |
| `bimodal` | short_min, short_max, long_min, long_max, short_prob | Mixed workloads |

**Arrival Patterns:** (defined in `src/simulation/llm_server.py:15-34`)
```python
# Constant rate
constant_rate(rate: float) -> Callable[[float], float]
# Returns: λ(t) = rate

# Step function
step_rate(base_rate: float, step_time: float, step_rate: float) -> Callable
# Returns: λ(t) = base_rate if t < step_time else step_rate

# Sinusoidal
sinusoidal_rate(base_rate: float, amplitude: float, period: float) -> Callable
# Returns: λ(t) = base_rate + amplitude * sin(2π*t/period)
```

### 3. BatchProcessor (`src/simulation/batch_processor.py:9`)

**Purpose:** Simulates GPU processing of batches (prefill + decode)

**Timing Model:** Based on realistic LLM inference characteristics

```python
# Prefill (parallel processing of prompts)
T_prefill = α * Σ(prompt_lengths) + β
# α = 0.001 s/token (per-token compute)
# β = 0.05 s (batch setup overhead)

# Decode (autoregressive generation)
T_decode = γ * max(output_lengths)
# γ = 0.0005 s/step (per-step time)
# Uses longest output (bottleneck)

# Total
T_total = T_prefill + T_decode
```

**Key Methods:**
- `compute_prefill_time(batch)` - Calculate prefill phase time
- `compute_decode_time(batch)` - Calculate decode phase time
- `process_batch(batch, current_time)` - Simulate full batch processing
- `get_statistics()` - Return batch processing stats

### 4. LLMInferenceServer (`src/simulation/llm_server.py:37`)

**Purpose:** Main simulation orchestrator - manages queue, batching, scheduling

**Key Responsibilities:**
1. Accept incoming requests
2. Queue management
3. Batch formation (collect up to batch_size requests)
4. Apply scheduling policy
5. Send batches to BatchProcessor
6. Track completed requests

**Critical Methods:**
```python
def run(self) -> None:
    """Main server loop - SimPy process"""
    # Continuously collect and process batches

def enqueue_request(self, request: Request) -> None:
    """Add request to queue, set queue_entry_time"""

def collect_batch(self) -> List[Request]:
    """
    Collect up to batch_size requests from queue.
    Applies scheduling policy for ordering.
    """

def get_statistics(self) -> Dict[str, Any]:
    """Return comprehensive server statistics"""
```

**Queuing Behavior:**
- FIFO queue with policy-based batch selection
- Requests wait until batch is formed
- No preemption (once batch starts, runs to completion)

### 5. Scheduling Policies (`src/scheduling/policies.py`)

**Base Class:** `SchedulingPolicy` (ABC at line 10)

```python
class SchedulingPolicy(ABC):
    @abstractmethod
    def sort_requests(self, requests: List[Request]) -> List[Request]:
        """Sort requests according to policy."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name."""
        pass
```

**Implementations:**

| Policy | Sorting Key | Line | Use Case |
|--------|-------------|------|----------|
| **FCFS** | `arrival_time` (ascending) | 20 | Fairness baseline |
| **SJF** | `prompt_length` (ascending) | 35 | Minimize avg latency |
| **Predicted-SJF** | Estimated total time (ascending) | 53 | Realistic SJF with prediction |
| **Priority** | Computed priority with aging (descending) | 95 | Prevent starvation |

**Factory Function:**
```python
def get_policy(policy_name: str, **kwargs) -> SchedulingPolicy:
    """Create policy by name."""
    # At line 155
    # Usage: get_policy("FCFS") or get_policy("Priority", base_priority=1.0, ...)
```

**Priority Policy Details:**
- Priority = `base_priority - aging_rate * wait_time + urgency_weight * (1/estimated_time)`
- Aging prevents starvation
- Configurable via `scheduling_policy_params` in config

### 6. MetricsCollector (`src/metrics/collector.py:10`)

**Purpose:** Comprehensive metrics computation from completed requests

**Key Method:**
```python
def compute_metrics(
    requests: List[Request],
    warmup_requests: int = 0,
    simulation_time: float = None,
    max_queue_length: int = None,
    total_batches: int = None,
    avg_batch_size: float = None
) -> Dict[str, float]:
    """Compute all metrics, filtering warmup period."""
```

**Metrics Categories:**

| Category | Metrics | Description |
|----------|---------|-------------|
| **Latency** | avg, median, std, min, max, p50, p95, p99 | End-to-end request latency |
| **Queue** | avg/median/max wait, p95/p99 wait | Time spent in queue |
| **Processing** | avg/median/max processing time | Time in prefill + decode |
| **Throughput** | req/sec, tokens/sec | System throughput |
| **Fairness** | Jain's index (0-1) | Fairness of latency distribution |
| **Starvation** | starvation_rate (%) | % requests exceeding threshold |
| **Tokens** | avg prompt/output length, total tokens | Token statistics |
| **System** | max_queue_length, total_batches, avg_batch_size | System metrics |

**Statistical Methods:**
- `compute_confidence_interval(values, confidence=0.95)` - Uses t-distribution
- Returns: `{"mean": float, "ci_lower": float, "ci_upper": float}`

**Fairness Calculation:**
```python
# Jain's Fairness Index
J = (Σ latencies)² / (n * Σ latencies²)
# Range: [1/n, 1], where 1 = perfect fairness
```

**Starvation Calculation:**
```python
# Default threshold: 3.0 * median_latency
starvation_rate = (count(latency > threshold) / total_requests) * 100
```

### 7. SimulationRunner (`src/experiments/runner.py:15`)

**Purpose:** Execute single simulation run, return metrics

**Usage:**
```python
from src.experiments.config import SimulationConfig
from src.experiments.runner import SimulationRunner

config = SimulationConfig(
    arrival_rate=10.0,
    batch_size=32,
    scheduling_policy="FCFS",
    num_requests=10000,
    warmup_requests=500,
    random_seed=42
)

runner = SimulationRunner(config)
metrics = runner.run()

print(f"Avg latency: {metrics['avg_latency']:.3f}s")
print(f"P99 latency: {metrics['p99_latency']:.3f}s")
print(f"Throughput: {metrics['throughput_req_per_sec']:.2f} req/s")
```

**Internal Flow:**
1. Create SimPy environment
2. Instantiate RequestGenerator with arrival pattern
3. Instantiate LLMInferenceServer with scheduling policy
4. Run simulation until num_requests generated
5. Collect server statistics
6. Compute metrics via MetricsCollector
7. Return dictionary of results

### 8. ExperimentRunner (`src/experiments/runner.py:100`)

**Purpose:** Run multiple replications, aggregate with confidence intervals

**Usage:**
```python
from src.experiments.config import SimulationConfig, ExperimentConfig
from src.experiments.runner import ExperimentRunner

base_config = SimulationConfig(arrival_rate=10.0, batch_size=32)
experiment = ExperimentConfig(
    name="example",
    description="Example experiment",
    base_config=base_config,
    num_replications=30,
    random_seed_start=42
)

# Generate configs with different seeds
configs = experiment.get_replication_configs()

# Run all replications
runner = ExperimentRunner(configs)
results = runner.run_all()  # List of metrics dicts

# Aggregate with 95% CI
aggregated = runner.aggregate_results(results)
print(f"Mean latency: {aggregated['avg_latency']['mean']:.3f}s")
print(f"95% CI: [{aggregated['avg_latency']['ci_lower']:.3f}, "
      f"{aggregated['avg_latency']['ci_upper']:.3f}]")
```

**Aggregation:**
- For each metric, computes: mean, ci_lower, ci_upper
- Uses t-distribution for confidence intervals
- Default: 95% confidence level

---

## Configuration System

### SimulationConfig (`src/experiments/config.py:8`)

**Complete Parameter Reference:**

```python
@dataclass
class SimulationConfig:
    # === Workload Parameters ===
    arrival_rate: float = 10.0                    # λ (requests per second)
    arrival_pattern: str = "constant"             # "constant", "step", "sinusoidal"
    arrival_pattern_params: Dict[str, Any] = {}   # Pattern-specific params

    # === Request Distribution Parameters ===
    # Prompt lengths
    prompt_length_dist: str = "lognormal"
    prompt_dist_params: Dict[str, Any] = {
        "mu": 4,       # Mean of log
        "sigma": 1.5   # Std of log
    }

    # Output lengths
    output_length_dist: str = "truncated_normal"
    output_dist_params: Dict[str, Any] = {
        "mu": 100,     # Mean tokens
        "sigma": 30,   # Std tokens
        "min": 10,     # Min tokens
        "max": 500     # Max tokens
    }

    # === Server Parameters ===
    batch_size: int = 32                          # Max requests per batch
    batch_timeout: Optional[float] = None         # Batch formation timeout (None = wait indefinitely)
    scheduling_policy: str = "FCFS"               # "FCFS", "SJF", "Predicted-SJF", "Priority"
    scheduling_policy_params: Dict[str, Any] = {} # Policy-specific params

    # === Timing Model Parameters ===
    alpha: float = 0.001   # Prefill per-token time (s/token)
    beta: float = 0.05     # Prefill overhead (s)
    gamma: float = 0.0005  # Decode step time (s/step)

    # === Simulation Parameters ===
    num_requests: int = 10000      # Total requests to generate
    warmup_requests: int = 500     # Requests to discard for warmup
    random_seed: Optional[int] = None  # For reproducibility
```

### Configuration Examples

#### Example 1: High-Load FCFS
```python
config = SimulationConfig(
    arrival_rate=50.0,        # High load
    batch_size=64,            # Large batches
    scheduling_policy="FCFS",
    num_requests=20000,
    warmup_requests=1000,
    random_seed=42
)
```

#### Example 2: SJF with Bimodal Workload
```python
config = SimulationConfig(
    arrival_rate=10.0,
    batch_size=32,
    scheduling_policy="SJF",
    prompt_length_dist="bimodal",
    prompt_dist_params={
        "short_min": 10,
        "short_max": 50,
        "long_min": 500,
        "long_max": 2000,
        "short_prob": 0.7  # 70% short, 30% long
    },
    num_requests=10000,
    warmup_requests=500
)
```

#### Example 3: Priority with Sinusoidal Load
```python
config = SimulationConfig(
    arrival_rate=10.0,  # Base rate
    arrival_pattern="sinusoidal",
    arrival_pattern_params={
        "amplitude": 8.0,   # λ varies from 2 to 18 req/s
        "period": 3600.0    # 1-hour cycle
    },
    scheduling_policy="Priority",
    scheduling_policy_params={
        "base_priority": 1.0,
        "aging_rate": 0.1,
        "urgency_weight": 10.0
    },
    num_requests=10000,
    warmup_requests=500
)
```

#### Example 4: Step Load Increase
```python
config = SimulationConfig(
    arrival_rate=5.0,  # Initial rate
    arrival_pattern="step",
    arrival_pattern_params={
        "step_time": 500.0,  # At t=500s
        "step_rate": 20.0    # Jump to 20 req/s
    },
    batch_size=32,
    scheduling_policy="Predicted-SJF",
    num_requests=10000,
    warmup_requests=500
)
```

---

## Testing & Quality

### Current State

**Status:** ⚠️ No tests implemented yet

**Available Tools:**
- `pytest>=7.4.0` - Installed but no test files exist
- `black>=23.0.0` - Code formatter (configured)
- `flake8>=6.0.0` - Linter (configured)

**Intended Structure:** (from README.md:214)
```bash
pytest tests/          # Tests directory doesn't exist yet
black src/ examples/   # Code formatting
flake8 src/ examples/  # Linting
```

### Testing Recommendations for AI Assistants

When implementing tests, follow this structure:

```
tests/
├── __init__.py
├── test_request.py              # Request dataclass tests
├── test_request_generator.py    # Workload generation tests
├── test_batch_processor.py      # Timing model tests
├── test_scheduling_policies.py  # Policy behavior tests
├── test_metrics_collector.py    # Metrics computation tests
├── test_simulation_runner.py    # Integration tests
└── fixtures/                    # Shared test fixtures
    └── sample_configs.py
```

**Key Testing Patterns:**

1. **Deterministic Testing:** Always use `random_seed` for reproducibility
   ```python
   config = SimulationConfig(random_seed=42)
   ```

2. **Small-Scale Tests:** Use small num_requests for fast tests
   ```python
   config = SimulationConfig(num_requests=100, warmup_requests=10)
   ```

3. **Timing Model Validation:**
   ```python
   def test_batch_processor_timing():
       processor = BatchProcessor(alpha=0.001, beta=0.05, gamma=0.0005)
       batch = [
           Request(id=1, arrival_time=0, prompt_length=100, expected_output_length=50)
       ]
       prefill_time = processor.compute_prefill_time(batch)
       expected = 0.001 * 100 + 0.05  # 0.15s
       assert abs(prefill_time - expected) < 1e-6
   ```

4. **Policy Behavior Tests:**
   ```python
   def test_sjf_ordering():
       policy = SJFPolicy()
       requests = [
           Request(id=1, arrival_time=0, prompt_length=200, ...),
           Request(id=2, arrival_time=1, prompt_length=50, ...)
       ]
       sorted_reqs = policy.sort_requests(requests)
       assert sorted_reqs[0].prompt_length < sorted_reqs[1].prompt_length
   ```

### Code Quality Standards

**Before Committing:**
```bash
# 1. Format code
black src/ examples/ tests/

# 2. Check linting
flake8 src/ examples/ tests/

# 3. Run tests (when implemented)
pytest tests/ -v

# 4. Check type hints (optional, if mypy added)
mypy src/
```

**Linting Exceptions:** (if needed, add to `.flake8` or `setup.cfg`)
```ini
[flake8]
max-line-length = 88  # Match black
ignore = E203, W503   # Black-compatible
exclude = .git,__pycache__,venv
```

---

## Common Tasks

### Task 1: Add a New Scheduling Policy

**Steps:**

1. **Create Policy Class** in `src/scheduling/policies.py`:
   ```python
   class MyNewPolicy(SchedulingPolicy):
       """Description of policy."""

       def __init__(self, param1: float = 1.0):
           self.param1 = param1

       def sort_requests(self, requests: List[Request]) -> List[Request]:
           """Sort requests by custom logic."""
           # Example: random shuffle
           import random
           shuffled = requests.copy()
           random.shuffle(shuffled)
           return shuffled

       @property
       def name(self) -> str:
           return "MyNew"
   ```

2. **Update Factory Function** at `src/scheduling/policies.py:155`:
   ```python
   def get_policy(policy_name: str, **kwargs) -> SchedulingPolicy:
       if policy_name == "FCFS":
           return FCFSPolicy()
       # ... existing policies ...
       elif policy_name == "MyNew":
           return MyNewPolicy(**kwargs)
       else:
           raise ValueError(f"Unknown policy: {policy_name}")
   ```

3. **Test the Policy:**
   ```python
   # In examples/ or tests/
   config = SimulationConfig(
       scheduling_policy="MyNew",
       scheduling_policy_params={"param1": 2.0}
   )
   runner = SimulationRunner(config)
   metrics = runner.run()
   ```

### Task 2: Add a New Request Distribution

**Steps:**

1. **Add Distribution Method** in `src/simulation/request_generator.py`:
   ```python
   def _sample_my_dist(self, params: Dict[str, Any]) -> int:
       """Sample from my custom distribution."""
       # Example: exponential
       rate = params.get("rate", 1.0)
       return int(self.rng.exponential(scale=1/rate))
   ```

2. **Update Sampling Logic** in `_sample_prompt_length()` or `_sample_output_length()`:
   ```python
   def _sample_prompt_length(self) -> int:
       if self.prompt_length_dist == "lognormal":
           # ... existing code ...
       elif self.prompt_length_dist == "my_dist":
           return self._sample_my_dist(self.prompt_dist_params)
       # ... rest ...
   ```

3. **Use in Configuration:**
   ```python
   config = SimulationConfig(
       prompt_length_dist="my_dist",
       prompt_dist_params={"rate": 0.01}
   )
   ```

### Task 3: Create a New Experiment

**Template:** Based on `examples/experiment1_batch_size.py`

```python
"""Experiment N: [Description]"""

import json
import numpy as np
import matplotlib.pyplot as plt
from src.experiments.config import SimulationConfig, ExperimentConfig
from src.experiments.runner import SimulationRunner, ExperimentRunner

def run_experiment():
    """Run Experiment N: [Description]"""

    # Define parameter sweep
    param_values = [1, 2, 4, 8, 16, 32, 64, 128]  # Example: batch sizes
    num_replications = 30  # For statistical rigor

    all_results = {}

    for param_val in param_values:
        print(f"\n=== Testing param={param_val} ===")

        # Create base config
        base_config = SimulationConfig(
            arrival_rate=10.0,
            batch_size=param_val,  # Or other parameter
            scheduling_policy="FCFS",
            num_requests=10000,
            warmup_requests=500,
        )

        # Create experiment config
        experiment = ExperimentConfig(
            name=f"experimentN_param_{param_val}",
            description=f"Param={param_val}",
            base_config=base_config,
            num_replications=num_replications,
            random_seed_start=42
        )

        # Run replications
        configs = experiment.get_replication_configs()
        runner = ExperimentRunner(configs)
        results = runner.run_all()

        # Aggregate results
        aggregated = runner.aggregate_results(results)
        all_results[param_val] = aggregated

        # Print summary
        print(f"Mean latency: {aggregated['avg_latency']['mean']:.3f}s "
              f"± {aggregated['avg_latency']['ci_upper'] - aggregated['avg_latency']['mean']:.3f}s")

    # Save results
    with open('experimentN_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot results (4-subplot example)
    plot_results(all_results, param_values)

    return all_results

def plot_results(results, param_values):
    """Create visualization of results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Experiment N: [Title]', fontsize=16)

    # Extract data
    means_latency = [results[p]['avg_latency']['mean'] for p in param_values]
    ci_lower_latency = [results[p]['avg_latency']['ci_lower'] for p in param_values]
    ci_upper_latency = [results[p]['avg_latency']['ci_upper'] for p in param_values]

    # Plot 1: Average Latency
    axes[0, 0].plot(param_values, means_latency, marker='o', linewidth=2)
    axes[0, 0].fill_between(param_values, ci_lower_latency, ci_upper_latency, alpha=0.3)
    axes[0, 0].set_xlabel('Parameter')
    axes[0, 0].set_ylabel('Average Latency (s)')
    axes[0, 0].set_title('Average Latency vs Parameter')
    axes[0, 0].grid(True, alpha=0.3)

    # Additional plots...

    plt.tight_layout()
    plt.savefig('experimentN_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to experimentN_results.png")

if __name__ == "__main__":
    run_experiment()
```

### Task 4: Modify Timing Model Parameters

**Global Change:** Edit `src/experiments/config.py:31-33`:
```python
# Timing model parameters (from report)
alpha: float = 0.002  # Changed from 0.001
beta: float = 0.10    # Changed from 0.05
gamma: float = 0.001  # Changed from 0.0005
```

**Per-Simulation Change:**
```python
config = SimulationConfig(
    alpha=0.002,  # Override default
    beta=0.10,
    gamma=0.001
)
```

**Impact:** Affects all simulations using BatchProcessor timing calculations

### Task 5: Debug Simulation Issues

**Enable Verbose Logging:**

Add logging to key methods:
```python
# In llm_server.py
def collect_batch(self) -> List[Request]:
    """Collect batch from queue."""
    print(f"[{self.env.now:.2f}s] Queue length: {len(self.request_queue)}")

    batch = []
    # ... rest of method ...

    print(f"[{self.env.now:.2f}s] Collected batch of size {len(batch)}")
    return batch
```

**Check Intermediate Metrics:**
```python
runner = SimulationRunner(config)
metrics = runner.run()

# Inspect detailed metrics
print(f"Max queue length: {metrics.get('max_queue_length', 'N/A')}")
print(f"Avg batch size: {metrics.get('avg_batch_size', 'N/A'):.2f}")
print(f"Total batches: {metrics.get('total_batches', 'N/A')}")
```

**Verify Warmup Filtering:**
```python
# Check warmup impact
metrics_no_warmup = runner.run()  # warmup_requests=500
config.warmup_requests = 0
metrics_with_warmup = runner.run()

print(f"Avg latency (no warmup): {metrics_no_warmup['avg_latency']:.3f}s")
print(f"Avg latency (with warmup): {metrics_with_warmup['avg_latency']:.3f}s")
```

### Task 6: Export Results for Analysis

**JSON Export:**
```python
import json

results = runner.run()
with open('simulation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

**CSV Export (for pandas):**
```python
import pandas as pd

# Multiple replications
all_results = []
for i in range(30):
    config.random_seed = 42 + i
    runner = SimulationRunner(config)
    metrics = runner.run()
    metrics['replication'] = i
    all_results.append(metrics)

df = pd.DataFrame(all_results)
df.to_csv('experiment_results.csv', index=False)
```

---

## Important Constraints

### 1. Academic Context

**This is a coursework project** for INDENG 174.

- **Authors:** Group 9 (Runyuan He, Jiedong Zhang, Qingyang Xu)
- **Purpose:** Educational research on LLM inference systems
- **License:** Academic use only

**Implications for AI Assistants:**
- Maintain academic integrity (cite sources, avoid plagiarism)
- Focus on learning objectives (explain concepts clearly)
- Don't over-engineer (keep solutions appropriate for coursework)

### 2. No Test Infrastructure

**Current Status:** No tests exist despite pytest being installed.

**When Adding Features:**
- Manually verify with `examples/simple_example.py`
- Run small-scale simulations (num_requests=100) for quick validation
- Check metrics make logical sense (latency > 0, throughput reasonable)

**If Implementing Tests:**
- Follow structure in [Testing & Quality](#testing--quality)
- Use small num_requests for fast execution
- Always set random_seed for determinism

### 3. Simulation-Specific Constraints

**SimPy Process Limitations:**
- Cannot use `async/await` (SimPy uses generators)
- Must use `yield` for all timing operations
- Avoid blocking operations (file I/O during simulation)

**Timing Model Assumptions:**
- Prefill is compute-bound (batched efficiently)
- Decode is memory-bound (limited by longest sequence)
- No KV cache eviction (simplified model)
- No preemption (batches run to completion)

**Statistical Validity:**
- Warmup period required (default 500 requests)
- Multiple replications needed for CI (default 30)
- Random seed critical for reproducibility

### 4. Performance Considerations

**Simulation Speed:**
- Discrete-event simulation is fast (no real waiting)
- Bottleneck: Python overhead for large num_requests
- For experiments: 10,000 requests typical, 30 replications = 300,000 requests total

**Memory Usage:**
- All completed requests stored in memory
- For 10,000 requests: ~1-2 MB per replication
- Be cautious with num_requests > 100,000

**Plotting:**
- matplotlib blocking (plt.show() pauses execution)
- Use `plt.savefig()` instead for automated experiments
- DPI=300 recommended for publication-quality plots

### 5. Configuration Gotchas

**Mutable Defaults:**
```python
# BAD - shared dict across instances
prompt_dist_params: Dict[str, Any] = {}

# GOOD - new dict per instance
prompt_dist_params: Dict[str, Any] = field(default_factory=dict)
```

**Optional vs Required:**
- `random_seed: Optional[int] = None` - None means use system randomness
- `batch_timeout: Optional[float] = None` - None means wait indefinitely

**Policy Parameters:**
```python
# Must match policy's __init__ parameters
scheduling_policy="Priority",
scheduling_policy_params={
    "base_priority": 1.0,      # Matches PriorityPolicy.__init__
    "aging_rate": 0.1,
    "urgency_weight": 10.0
}
```

### 6. Git & Collaboration

**Submodule Warning:**
- `llm-infer-report/` is a git submodule (external repo)
- Don't commit changes inside submodule directory
- Update via `git submodule update --remote`

**Branch Naming:**
- Feature branches must start with `claude/`
- Must end with session ID (or push will fail with 403)
- Example: `claude/claude-md-mipbh3eb1g560dwp-01LJfTcYnC35hAd2WLo6KLPo`

**Large Files:**
- `.gitignore` excludes `*.json`, `*.csv`, `*.png`, `*.pdf`
- Experiment results won't be committed
- This is intentional (results are reproducible via random_seed)

### 7. Planned vs Implemented

**Implemented (2/5 experiments):**
1. ✅ Experiment 1: Batch Size Sensitivity (`examples/experiment1_batch_size.py`)
2. ✅ Simple Example (`examples/simple_example.py`)

**Planned (3/5 experiments):**
3. ⏳ Experiment 2: Scheduling Policy Comparison
4. ⏳ Experiment 3: Load Stress Testing
5. ⏳ Experiment 4: Workload Distribution Sensitivity
6. ⏳ Experiment 5: Time-Varying Load Patterns

**When Implementing Planned Experiments:**
- Follow template in [Task 3: Create a New Experiment](#task-3-create-a-new-experiment)
- Refer to README.md:137-159 for experiment specifications
- Use 30 replications for statistical rigor
- Create visualization (4-subplot pattern from experiment1)

### 8. External Dependencies

**No Internet Required:**
- All dependencies installable offline via `requirements.txt`
- No API calls, no cloud services

**Version Pins:**
- Using `>=` for flexibility (e.g., `simpy>=4.0.0`)
- May need to pin versions if compatibility issues arise
- Current versions tested: Python 3.8+

### 9. Documentation Synchronization

**Two Documentation Files:**
1. **README.md** - User-facing (installation, usage, examples)
2. **CLAUDE.md** - AI assistant guide (architecture, conventions, internals)

**When Modifying Code:**
- Update README.md if user-facing behavior changes
- Update CLAUDE.md if architecture/conventions change
- Keep both in sync for consistency

---

## Quick Reference

### File Locations

| Component | File | Line |
|-----------|------|------|
| Request dataclass | `src/simulation/request.py` | 8 |
| RequestGenerator | `src/simulation/request_generator.py` | 12 |
| BatchProcessor | `src/simulation/batch_processor.py` | 9 |
| LLMInferenceServer | `src/simulation/llm_server.py` | 37 |
| Arrival patterns | `src/simulation/llm_server.py` | 15-34 |
| SchedulingPolicy ABC | `src/scheduling/policies.py` | 10 |
| FCFS/SJF/Predicted-SJF/Priority | `src/scheduling/policies.py` | 20-150 |
| get_policy factory | `src/scheduling/policies.py` | 155 |
| MetricsCollector | `src/metrics/collector.py` | 10 |
| SimulationConfig | `src/experiments/config.py` | 8 |
| ExperimentConfig | `src/experiments/config.py` | 54 |
| SimulationRunner | `src/experiments/runner.py` | 15 |
| ExperimentRunner | `src/experiments/runner.py` | 100 |

### Default Values

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| arrival_rate | 10.0 req/s | Moderate load |
| batch_size | 32 | Typical GPU batch size |
| scheduling_policy | "FCFS" | Fairness baseline |
| num_requests | 10000 | Statistical sufficiency |
| warmup_requests | 500 | Transient removal |
| num_replications | 30 | 95% CI accuracy |
| alpha | 0.001 s/token | Realistic prefill |
| beta | 0.05 s | Batch overhead |
| gamma | 0.0005 s/step | Realistic decode |
| prompt_length_dist | "lognormal" | Skewed distribution |
| output_length_dist | "truncated_normal" | Bounded distribution |

### Metrics Keys

```python
metrics = {
    # Latency
    'avg_latency': float,
    'median_latency': float,
    'std_latency': float,
    'min_latency': float,
    'max_latency': float,
    'p50_latency': float,
    'p95_latency': float,
    'p99_latency': float,

    # Queue
    'avg_queue_wait': float,
    'median_queue_wait': float,
    'max_queue_wait': float,
    'p95_queue_wait': float,
    'p99_queue_wait': float,

    # Processing
    'avg_processing_time': float,
    'median_processing_time': float,
    'max_processing_time': float,

    # Throughput
    'throughput_req_per_sec': float,
    'throughput_tokens_per_sec': float,
    'total_requests': int,
    'simulation_time': float,

    # Fairness
    'fairness_index': float,        # Jain's index [0, 1]
    'starvation_rate': float,       # Percentage [0, 100]

    # Tokens
    'avg_prompt_length': float,
    'avg_output_length': float,
    'total_tokens': int,

    # System
    'max_queue_length': int,
    'total_batches': int,
    'avg_batch_size': float,
}
```

### Common Commands

```bash
# Installation
pip install -r requirements.txt

# Run examples
python examples/simple_example.py
python examples/experiment1_batch_size.py

# Code quality
black src/ examples/
flake8 src/ examples/

# Git operations
git status
git add CLAUDE.md
git commit -m "Add comprehensive CLAUDE.md documentation"
git push -u origin claude/claude-md-mipbh3eb1g560dwp-01LJfTcYnC35hAd2WLo6KLPo

# Submodule management
git submodule init
git submodule update
git submodule update --remote
```

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-03 | 1.0.0 | Initial comprehensive CLAUDE.md created |

---

## Contact & Support

**Project Authors:**
- Runyuan He (3041920716)
- Jiedong Zhang (3041913865)
- Qingyang Xu (3041979645)

**Course:** INDENG 174
**Institution:** UC Berkeley (implied from context)

**Related Repositories:**
- Main: `momoway/llm-infer`
- Report: `momoway/llm-infer-report` (submodule)

---

**End of CLAUDE.md**
