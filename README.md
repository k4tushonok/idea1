# HTGO — Hierarchical Textual Gradient Optimization

Automatic prompt optimization framework that combines local textual-gradient search with a global history-aware exploration step.

## How it works

Each generation runs two phases:

1. **Local phase** — the current best prompt is iteratively improved via textual gradients (LLM-generated error analysis) and beam search with Monte Carlo paraphrases.
2. **Global phase** — triggered on stagnation or every `GLOBAL_TRIGGER_INTERVAL` generations. The LLM receives a meta-prompt built from the last `GLOBAL_HISTORY_WINDOW` history nodes and generates structurally new candidates.

The best candidate from both phases becomes the starting prompt for the next generation. Optimization stops when no improvement exceeds `MIN_IMPROVEMENT` for `PATIENCE` consecutive generations, or after `MAX_GENERATIONS` total.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

```python
from hierarchical_optimizer import HierarchicalOptimizer
from data_structures import Example
from config import SQUAD_METRICS   # or XSUM_METRICS / GENERATION_METRICS / GSM8K_METRICS

# 1. Prepare examples
train_examples = [
    Example(
        input_text="Context: ... Question: What is X?",
        expected_output="X",
        metadata={"all_answers": ["X", "x"]}
    ),
    # ...
]
validation_examples = [...]   # held-out set used for scoring

# 2. Create optimizer
optimizer = HierarchicalOptimizer(
    metrics_config=SQUAD_METRICS,
    task_description="Extract the exact answer to the question from the context."
)

# 3. Run optimization
initial_prompt = "Answer the question based on the context."
best_node = optimizer.optimize(
    initial_prompt=initial_prompt,
    train_examples=train_examples,
    validation_examples=validation_examples,
    save_dir="optimization_results/my_run",   # optional: saves reports to this directory
)

print("Best prompt:", best_node.prompt_text)
print("Best score: ", best_node.selection_score())
```

---

## Supported tasks and metrics

Configure `metrics_config` with the preset that matches your task, or build a custom list.

| Task | Preset | Metrics |
|------|--------|---------|
| SQuAD v2 (QA) | `SQUAD_METRICS` | Exact Match ×0.8 + Token F1 ×0.2 |
| XSum (summarisation) | `XSUM_METRICS` | ROUGE-L ×0.5 + BERTScore ×0.35 + METEOR ×0.15 |
| CommonGen (constrained gen.) | `GENERATION_METRICS` | BERTScore ×0.5 + ROUGE-L ×0.3 + METEOR ×0.2 |
| GSM8K (math reasoning) | `GSM8K_METRICS` | Numeric Exact Match ×1.0 |

All presets are defined in [config.py](config.py).

---

## Key configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_GENERATIONS` | 6 | Maximum number of optimization generations |
| `PATIENCE` | 3 | Early stopping patience (generations without improvement) |
| `MIN_IMPROVEMENT` | 0.001 | Minimum score gain to count as improvement |
| `LOCAL_ITERATIONS_PER_GENERATION` | 2 | Local optimization iterations per generation |
| `N_GRADIENTS` | 4 | Error-sampling iterations for gradient generation |
| `ERRORS_PER_GRADIENT` | 4 | Failure examples per gradient |
| `MC_SAMPLES_PER_STEP` | 2 | Monte Carlo paraphrase samples per candidate |
| `PRE_SCREEN_TOP_K` | 4 | Candidates fully evaluated after pre-screening |
| `GLOBAL_CANDIDATES` | 16 | Candidates generated per global step |
| `GLOBAL_HISTORY_WINDOW` | 10 | History nodes included in the meta-prompt |
| `GLOBAL_TRIGGER_INTERVAL` | 2 | Run global step every N generations |
| `GLOBAL_TEMPERATURE` | 0.75 | LLM temperature for global optimization |
| `LOCAL_TEMPERATURE` | 0.1 | LLM temperature for local optimization |
| `EXEMPLAR_SELECTION_STRATEGY` | `current_most_frequent` | Strategy for selecting wrong-exemplars (see below) |
| `PROVIDER` | `openai` | LLM provider: `openai`, `gemini`, or `local` |
| `MODEL` | `gpt-4o-mini` | Model name |

### Wrong-exemplar selection strategies

| Strategy | Description |
|----------|-------------|
| `current_most_frequent` | Examples most often failed by the **current** best prompt (default) |
| `accumulative_most_frequent` | Examples most often failed **across all** history nodes |
| `random` | Random sample per generation |
| `constant` | Fixed random sample (seed 0), never changes |

---

## Saving results

Pass `save_dir` to `optimizer.optimize()` to automatically produce:

```
save_dir/
  best_prompt.txt          # final optimized prompt text
  optimization_history.json
  optimization_report.json # scores, timing, generation log
  trajectory.txt           # per-generation score trace
```

---
