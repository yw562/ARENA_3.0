# Measuring Alignment Under Self-Improvement

This project studies how **alignment properties (refusal / safety)** evolve under
iterated self-improvement loops (STaR / SPIN-style), alongside capability gains.

The core question:
> As models bootstrap their own reasoning ability, does alignment hold up?

---

## Overview

We implement a minimal, controlled self-improvement pipeline:

1. **Generate** model outputs on a reasoning task (GSM8K)
2. **Filter** for correctness (exact match)
3. **Self-train** on model-generated reasoning traces
4. **Evaluate** both:
   - Capability (GSM8K accuracy)
   - Safety (AdvBench refusal & leakage rates)
5. Repeat for multiple iterations

Crucially, capability and safety are tracked **at every iteration**.

---

## Design Principles

- **Separation of concerns**
  - Research loop & evaluation run locally
  - Parameter updates (LoRA / SFT) are executed platform-side (Fireworks)
- **Auditability**
  - All eval outputs and safety labels are saved per iteration
- **Controlled comparisons**
  - Supports frozen-policy vs LoRA-updated variants

---

## Metrics

### Capability
- GSM8K exact match accuracy

### Safety
- AdvBench hard refusal rate
- Leakage rate (non-refusal harmful continuations)

---

## Repository Structure

```text
alignment_self_improve/
├── configs/
│   └── instruct.yaml          # experiment configuration
├── src/
│   └── asi/
│       ├── cli.py             # main entrypoint
│       ├── loop.py            # generate → filter → train loop
│       ├── data.py            # dataset loaders & parsers
│       ├── train.py           # Tinker / Fireworks wrappers (sampling + LoRA)
│       ├── eval_capability.py # GSM8K exact-match metric
│       ├── eval_safety.py     # refusal & leakage metrics
│       └── tracking.py        # metrics & artifact logging
├── scripts/
│   └── run_experiment.py      # thin wrapper around cli
├── results/
│   └── run_YYYYMMDD_HHMMSS/   # auto-generated experiment outputs
└── README.md


---

## Running the Pipeline

```bash
python scripts/run_experiment.py --config configs/instruct.yaml



Contact

This project is developed as part of ARENA and in collaboration with
Simon Lermen. For questions, see commit history and configuration files.
