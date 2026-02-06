from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .data import GSM8KExample, load_gsm8k, parse_gsm8k_final_answer, parse_model_final_answer
from .train import (
    TinkerModelRef,
    create_initial_model_ref,
    finetune_sft_lora,
    load_model_ref,
    sample_text,
)
from .tracking import ensure_dir, write_json


@dataclass
class IterationResult:
    iteration: int
    model_dir: Path
    generated_data_path: Path
    filtered_data_path: Path
    num_generated: int
    num_kept: int

def run_self_improvement_iteration(
    *,
    iteration: int,
    model_dir: Path,
    config: Dict,
    work_dir: Path,
) -> IterationResult:
    iter_dir = work_dir / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    generated_path = iter_dir / "generated.jsonl"
    filtered_path = iter_dir / "filtered.jsonl"
    new_model_dir = iter_dir / "model"

    # --- new add：generation ---
    if generated_path.exists():
        print(f"[iter {iteration}] Found existing generated data.")
        with generated_path.open("r") as f:
            num_generated = sum(1 for _ in f)
    else:
        print(f"[iter {iteration}] No generated data found. Generating...")
        num_generated = generate_candidates(
            model_dir=model_dir,
            config=config,
            output_path=generated_path,
        )

    # --- new add：filtering ---
    if filtered_path.exists():
        print(f"[iter {iteration}] Found existing filtered data.")
        with filtered_path.open("r") as f:
            num_kept = sum(1 for _ in f)
    else:
        num_kept = filter_candidates(
            generated_path=generated_path,
            config=config,
            output_path=filtered_path,
        )

    # --- previous logics ---
    if num_kept == 0:
        print(f"[iter {iteration}] No training data kept; skipping finetune.")
        return IterationResult(
            iteration=iteration,
            model_dir=model_dir,
            generated_data_path=generated_path,
            filtered_data_path=filtered_path,
            num_generated=num_generated,
            num_kept=0,
        )

    # --- train, safety optional ---
    if (new_model_dir / "tinker_model_ref.json").exists():
        print(f"[iter {iteration}] Model already trained. Skipping training.")
    else:
        train_on_filtered_data(
            base_model_dir=model_dir,
            training_data_path=filtered_path,
            config=config,
            output_model_dir=new_model_dir,
            iteration=iteration,
        )

    return IterationResult(
        iteration=iteration,
        model_dir=new_model_dir,
        generated_data_path=generated_path,
        filtered_data_path=filtered_path,
        num_generated=num_generated,
        num_kept=num_kept,
    )


def _get_prompt_template(cfg: Dict) -> str:
    return cfg["generation"]["prompt_template"]


def generate_candidates(
    *,
    model_dir: Path,
    config: Dict,
    output_path: Path,
) -> int:
    """
    Generate candidate GSM8K solutions using current model (Tinker sampling).
    Writes JSONL rows containing question, ground_truth_final, model_output, parsed_final.
    """
    ensure_dir(output_path.parent)

    # Load training examples for self-generation (small subset)
    n = int(config["generation"]["num_samples"])
    train_split = config.get("generation", {}).get("train_split", "train")
    gsm_train = load_gsm8k(train_split, limit=n)

    model_ref = load_model_ref(model_dir)

    template = _get_prompt_template(config)
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    temperature = float(config["generation"]["temperature"])
    stop = None  # keep it simple; you can tune later

    rows = []
    for ex in gsm_train:
        prompt = template.format(question=ex.question)
        outs = sample_text(
            model_ref=model_ref,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=1,
        )
        out = outs[0] if outs else ""
        rows.append(
            {
                "question": ex.question,
                "answer": ex.answer,
                "ground_truth_final": parse_gsm8k_final_answer(ex.answer),
                "model_output": out,
                "parsed_final": parse_model_final_answer(out),
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(rows)


def filter_candidates(
    *,
    generated_path: Path,
    config: Dict,
    output_path: Path,
) -> int:
    """
    Minimal exact-match filtering: keep examples whose parsed_final == ground_truth_final.
    Writes JSONL rows with prompt/completion for SFT.
    """
    ensure_dir(output_path.parent)

    method = config["filtering"]["method"]
    if method != "exact_match":
        raise ValueError(f"Unsupported filtering.method={method} in MVP")

    prompt_template = _get_prompt_template(config)

    kept = []
    with generated_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("parsed_final", "") == r.get("ground_truth_final", ""):
                # completion format: encourage a stable final answer marker
                completion = r["model_output"]
                # If model didn't include ####, append stable marker to reduce eval noise later
                if "####" not in completion:
                    completion = completion.rstrip() + f"\n#### {r['ground_truth_final']}\n"
                kept.append(
                    {
                        "prompt": prompt_template.format(question=r["question"]),
                        "completion": completion,
                    }
                )

    with output_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(kept)
def train_on_filtered_data(
    *,
    base_model_dir: Path,
    training_data_path: Path,
    config: Dict,
    output_model_dir: Path,
    iteration: int,
) -> None:
    """
    Phase-1 (frozen-policy): do NOT update weights.
    We still materialize iter_{k}/model/tinker_model_ref.json for bookkeeping,
    so the loop structure (generate→filter→"train") stays intact.
    """
    mode = config.get("training", {}).get("mode", "train")
    # base_ref = load_model_ref(base_model_dir)
    base_ref = load_model_ref(base_model_dir)

    current_model_id = (
        base_ref.sampling_model_path
        if base_ref.sampling_model_path is not None
        else base_ref.base_model
    )


    # Always load pairs (we keep artifacts for audit), but frozen ignores them
    pairs: List[Tuple[str, str]] = []
    with training_data_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pairs.append((r["prompt"], r["completion"]))

    if mode == "frozen":
        # Key: call finetune_sft_lora but pass zero steps + batch_size=1 so it's a no-op
        # (works even if later you swap train.py implementation again).
        finetune_sft_lora(
            base_model=base_ref.base_model,
            train_pairs=[],
            output_model_dir=output_model_dir,
            learning_rate=0.0,
            max_steps=0,
            batch_size=1,
            lora_rank=1,
            save_name=f"asi_iter_{iteration}_frozen",
            mode="frozen",   # NEW
        )

        return

    # Phase-2 (real training) — keep your original path
    lr = float(config["training"]["learning_rate"])
    bs = int(config["training"]["batch_size"])
    steps = int(config["training"]["max_steps"])
    rank = int(config["training"].get("lora_rank", 32))
    save_name = f"asi_iter_{iteration}"

    finetune_sft_lora(
        base_model=current_model_id, 
        train_pairs=pairs,
        output_model_dir=output_model_dir,
        learning_rate=lr,
        max_steps=steps,
        batch_size=bs,
        lora_rank=rank,
        save_name=save_name,
        mode="train",
    )

