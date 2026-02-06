# Results Index (alignment_self_improve)

This folder stores run artifacts for the **Alignment-under-Self-Improvement** experiment:
- Capability: GSM8K (exact-match on final answer)
- Safety: AdvBench (refusal vs leakage classification)

Each run directory is self-contained:
- `config.yaml`: resolved config used for the run
- `metrics.csv`: summary metrics by iteration
- `iter_*/`: per-iteration artifacts (prompts, outputs, scored evals, model refs)

---

## Baselines

### 1) Failed baseline (negative control): Llama-3.2-1B (base)
**Folder:** `run_20260204_002153/`  
**Model:** `meta-llama/Llama-3.2-1B` (base, non-instruct)

**Why it failed / what we learned**
- Base (non-instruct) checkpoint is **not safety-aligned**, so AdvBench refusal metrics are not meaningful as an “alignment baseline”.
- Kept as a negative control for traceability and comparison.

**Status:** not used for self-improvement iterations.

---

### 2) Valid safety baseline (starting point): Qwen3-4B-Instruct
**Folder:** `run_20260204_100058/`  
**Model:** `Qwen/Qwen3-4B-Instruct-2507`

**What worked**
- Stable instruction-following behavior.
- AdvBench responses are consistently refusal-style (hard refusals dominate).

**Known issue (must-fix before reporting capability numbers)**
- GSM8K exact-match accuracy is currently unreliable because generated outputs contain special tokens
  such as `<|endoftext|>` / `<|im_end|>` and sometimes include extra trailing text, which breaks exact-match.
- Action item: **strip special tokens + normalize final answer** before computing GSM8K exact-match.

**Status:** use this as the aligned baseline checkpoint for subsequent self-improvement iterations,
*after* fixing GSM8K evaluation hygiene.

---

## Notes / TODOs

- [ ] GSM8K evaluator: strip special tokens (`<|endoftext|>`, `<|im_end|>`) and extract `#### <number>` robustly.
- [ ] Safety classifier: ensure “leakage” triggers only on actionable harmful content (avoid flagging benign refusal phrasing).

---

## Run 2026-02-04 — Iter-0 Baseline

This run establishes a clean iter-0 baseline prior to self-improvement.

**Model**
- Qwen/Qwen3-4B-Instruct-2507

**Folder:** run_20260204_133909/

**Capability (GSM8K, n=50)**
- Exact match accuracy: **0.72**
- 0-shot evaluation
- Full chain-of-thought generation preserved
- Minor truncation observed on a small number of longer problems, likely due to generation length limits

**Safety (AdvBench-mini, n=5)**
- Hard refusal rate: **1.0**
- No post-refusal leakage detected
- Refusals follow standard instruct-style patterns with benign redirection

**Evaluation hygiene updates**
- Removed stop tokens for GSM8K capability evaluation to avoid premature truncation
- Hardened leakage detection logic to avoid false negatives caused by sentence splitting

This confirms the starting checkpoint is aligned and suitable as a baseline
for measuring alignment drift under self-improvement.

**Planned next steps**
- Increase GSM8K `max_new_tokens`
- Expand AdvBench sample size
- Run iter 1–3 to observe capability vs safety trends

# Run 2026-02-04 — Iter-0 + Iter-1 (Strict RS-SFT, Stall Case)

**Folder:** run_20260204_221037/

This run probes the behavior of a strict rejection-sampling self-improvement loop under limited signal and backend capacity constraints.

---

## Model
* **Base checkpoint:** meta-llama/Llama-3.1-8B
* **Starting point:** non-instruct model
* **Note:** chosen due to temporary unavailability / rate-limit of Qwen-3-4B-Instruct on Tinker backend

---

## Iter-0: Baseline Evaluation

### Safety (AdvBench-mini, n=5)
* **Hard refusal rate:** 0.0
* **Leakage:** extensive
* Model produces directly actionable harmful instructions (e.g. bomb-making, hacking, weapon construction).
* Responses are classified as label="other" with is_refusal=false, indicating absence of safety alignment.

**Interpretation:**
As expected, a base (non-instruct) model does not provide a meaningful safety baseline. This run is kept as a negative control for comparison and traceability.

### Capability (GSM8K, n=50)
* **Exact-match accuracy:** 0.0 (unreliable)
* **Outputs frequently:**
    * contain verbose or repetitive reasoning
    * violate final-answer formatting (#### <number>)
    * include extraneous text or incorrect arithmetic
* Parsing failures dominate exact-match scoring.

**Interpretation:**
GSM8K exact-match is not meaningful for this checkpoint without additional normalization; capability numbers are not used for conclusions in this run.

---

## Iter-1: Self-Improvement Attempt (Strict Rejection Sampling)

### Generation
* 20 GSM8K problems sampled
* Model consistently produced incorrect final answers despite valid reasoning structure
* **Example failure pattern:**
    * Ground truth: #### 72
    * Model output: #### 96

### Filtering
* **Filtering method:** exact-match on final answer
* **Kept samples:** 0 / 20
* No examples satisfied the strict rejection-sampling criterion.

### Training
* Skipped (no training data after filtering)

---

## Key Result:
Under strict rejection-sampling SFT, self-improvement stalls at iter-1 due to zero retained signal.

## Takeaway
This run demonstrates a signal-sparsity failure mode of strict self-improvement:
When starting from a non-aligned (base) checkpoint, exact-match rejection sampling can eliminate all training signal, preventing any learning update.

## Takeaway
This run isolates a clear failure mode:
Strict RS-SFT requires an already-aligned starting distribution.
Without it, filtering eliminates all learning signal, preventing any update.

---

## Planned Fixes for Next Run
The following changes will be applied in the next iteration to restore meaningful signal:

* **Use an aligned starting checkpoint**
    * Switch from meta-llama/Llama-3.1-8B → meta-llama/Llama-3.1-8B-Instruct
* **Increase generation signal**
    * Set num_samples: 100
    * Purpose: reduce variance and avoid accidental zero-kept batches.
* **Graceful handling of zero-kept iterations**
    * If num_kept == 0, skip training and proceed to evaluation.
    * Purpose:
        * avoid runtime failure
        * explicitly measure self-improvement stall as an outcome
        * preserve comparability across iterations
    * This allows the experiment to distinguish “no learning signal” from “learning that degrades alignment”, which is critical for phase-boundary analysis.

# Llama-3.1-8B-Instruct Self-Improvement Baseline and iter1(fail)

**Run:** `Llama-3.1-8B-Instruct`  
**Folder:** `results/run_20260204_002153/`

## Purpose
Test whether a small aligned instruct model can serve as a baseline for measuring alignment under self-improvement using a LoRA-based self-training loop.

## Setup (key parameters)
* **Base model:** `meta-llama/Llama-3.1-8B-Instruct`
* **Loop:** 1 iteration (baseline → attempted update)
* **Self-training task:** GSM8K
    * 100 samples, temperature 0.0
    * Strict exact-match + final-answer-format filtering
* **Training:** LoRA, 50 steps

## Evaluation
* **GSM8K** (capability)
* **AdvBench** (refusal rate)

## Results
| iter | GSM8K acc | AdvBench refusal |
| :--- | :--- | :--- |
| 0 | ~0.04 | ~0.40 |

* After filtering, almost no usable GSM8K samples remain.
* The self-improvement loop collapses at iter-1.

## Conclusion
Despite running end-to-end, Llama-3.1-8B-Instruct fails due to a capability floor: GSM8K accuracy is too low to sustain correctness-based self-training.

**Conclusion:** Llama-3.1-8B-Instruct is not a viable baseline for alignment-under-self-improvement experiments.

## run_20260206_162023 — Fireworks baseline (Qwen3-8B-Base)

**Path**
/root/ARENA_3.0/alignment_self_improve/results/run_20260206_162023

pgsql
Copy code


**Setup**
- Backend: Fireworks
- Model: Qwen3-8B-Base
- Iterations: 0

**Metrics (iter0)**
| iteration | gsm8k_acc | advbench_refusal_rate | n_gsm8k_eval | n_advbench_eval |
|-----------|-----------|-----------------------|---------------|------------------|
| 0 | 0.74 | 0.40 | 50 | 5 |

**Notes**
- Capability metric recorded as baseline reference.
- Safety metrics logged for completeness; base model is not aligned.

**Next**
- Switch to **Qwen3-VL-30B (aligned)**.

## Qwen3-30B-Thinking — iter0 (Fireworks)

**Setup**
- Backend: Fireworks
- Model: Qwen3-30B-Thinking
- Iteration: 0

**Metrics**
| iteration | gsm8k_acc | advbench_refusal_rate | n_gsm8k_eval | n_advbench_eval |
|-----------|-----------|-----------------------|---------------|------------------|
| 0 | 0.80 | 0.60 | 50 | 5 |

**Notes**
- Capability improves relative to 8B baseline.
- Refusal rate increase likely driven by extensive thinking traces.
- Safety metrics are potentially biased due to truncation and evaluation mismatch.

**Next**
- Switch to non-thinking instruct model for alignment evaluation.


# Phase 1 Baseline — Frozen-Policy Self-Improvement

**Run directory**  
`alignment_self_improve/results/run_20260206_165451`

## Setup
- **Model**: `accounts/fireworks/models/qwen3-vl-30b-a3b-instruct`
- **Policy**: Frozen (no parameter updates)
- **Iterations**: 0
- **Backend**: Fireworks
- **Purpose**: Establish a clean anchor baseline for capability and safety
  before enabling parameter updates in Phase 2.

## Metrics (iter 0)

| iteration | gsm8k_acc | advbench_refusal_rate | n_gsm8k_eval | n_advbench_eval |
|----------|-----------|-----------------------|--------------|-----------------|
| 0        | 0.96      | 1.00                  | 50           | 5               |

## Interpretation
- The model demonstrates **near-saturated mathematical capability** on GSM8K.
- The model exhibits **perfect refusal behavior** on AdvBench under a
  frozen-policy setting.
- This run serves as the **anchor baseline** for all subsequent alignment
  drift measurements.

## Status
- Phase 1 (frozen-policy self-improvement) **completed**.
- Phase 2 will enable **real parameter updates (LoRA)** to measure alignment
  dynamics under iterative self-improvement.


# Run: 2026-02-06 — Iter-0 Baseline (kimi-k2-instruct-0905)

## Setup
- Backend: Fireworks
- Model: kimi-k2-instruct-0905
- Iterations: 0 (baseline only)
- Training: disabled (baseline evaluation)

## Metrics (iter-0)

| iteration | gsm8k_acc | advbench_refusal_rate | n_gsm8k_eval | n_advbench_eval |
|----------|-----------|-----------------------|--------------|-----------------|
| 0 | 0.98 | 0.80 | 50 | 5 |

## Interpretation
- The model shows near-saturated mathematical capability on GSM8K.
- Safety behavior is strong but not perfect, leaving measurable headroom for alignment drift analysis.
- This run serves as the **iter-0 anchor** for subsequent self-improvement experiments.

## Next Step
- Enable supervised fine-tuning (LoRA) and run iter-1 → iter-k to measure alignment dynamics under self-improvement.


# Run 2026-02-06 — Frozen Self-Improvement Baseline

This run establishes a **frozen-policy anchor baseline** for measuring alignment drift under self-improvement loops.

## Experimental Setup

- **Model**: `accounts/fireworks/models/kimi-k2-instruct-0905`
- **Loop**: STaR/SPIN-style
  - Generate → Filter (exact-match on GSM8K) → Replay
- **Training mode**: `frozen` (no weight updates)
- **Iterations**: 3 self-improvement steps
- **Sampling**:
  - GSM8K: 50 examples (exact match)
  - AdvBench-mini: 5 prompts
- **Evaluation metrics**:
  - Capability: GSM8K exact-match accuracy
  - Safety: hard refusal rate, leakage rate

## Results

| iteration | gsm8k_acc | refusal_rate | leakage_rate |
|----------:|----------:|-------------:|-------------:|
| 0 | 1.00 | 0.80 | 0.00 |
| 1 | 0.98 | 0.80 | 0.00 |
| 2 | 1.00 | 0.80 | 0.00 |
| 3 | 1.00 | 0.80 | 0.00 |

## Key Observations

- **No alignment drift under frozen policy**:
  - Safety metrics (refusal, leakage) remain invariant across iterations.
- **Capability remains stable**:
  - Minor GSM8K fluctuation (0.98 → 1.00) within sampling noise.
- **Conclusion**:
  - The self-improvement *loop structure alone* does not induce alignment drift.
  - Parameter updates are a necessary condition for safety change.

## Purpose of This Run

This run serves as a **control group** and **anchor baseline** for Phase 2 (LoRA fine-tuning),
ensuring that any future safety drift can be causally attributed to learning rather than
iteration mechanics or evaluation artifacts.
