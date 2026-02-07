

# train.py for Fireworks (FINAL, WORKING VERSION)

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from .tracking import ensure_dir, write_json

MODEL_REF_FILENAME = "tinker_model_ref.json"

# ============================================================
# Model ref
# ============================================================

@dataclass
class TinkerModelRef:
    base_model: str
    sampling_model_path: Optional[str] = None
    lora_job_id: Optional[str] = None


def save_model_ref(model_dir: Path, ref: TinkerModelRef) -> None:
    ensure_dir(model_dir)
    write_json(model_dir / MODEL_REF_FILENAME, ref.__dict__)


def load_model_ref(model_dir: Path) -> TinkerModelRef:
    p = model_dir / MODEL_REF_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"Missing {MODEL_REF_FILENAME} in {model_dir}")
    return TinkerModelRef(**json.loads(p.read_text()))


def create_initial_model_ref(model_dir: Path, base_model: str) -> None:
    save_model_ref(
        model_dir,
        TinkerModelRef(base_model=base_model, sampling_model_path=None),
    )

# ============================================================
# Fireworks client (OpenAI-compatible)
# ============================================================

_FW_CLIENT: Optional[OpenAI] = None


def _get_fw_client() -> OpenAI:
    global _FW_CLIENT
    if _FW_CLIENT is None:
        _FW_CLIENT = OpenAI(
            # key?
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FIREWORKS_API_KEY"],
        )
    return _FW_CLIENT

# ============================================================
# Sampling (THIS is what was breaking before)
# ============================================================

def _fw_chat(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: Optional[List[str]] = None,
) -> str:
    client = _get_fw_client()
    last_err: Exception | None = None

    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(2)

    raise RuntimeError(f"Fireworks request failed after retries: {last_err}")


def sample_text(
    *,
    model_ref: TinkerModelRef,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: Optional[List[str]] = None,
    num_samples: int = 1,
) -> List[str]:
    model_id = model_ref.sampling_model_path or model_ref.base_model
    return [
        _fw_chat(model_id, prompt, max_tokens, temperature, stop)
        for _ in range(num_samples)
    ]

# # ============================================================
# # Fine-tuning (Phase-1 frozen / Phase-2 train)
# # ============================================================

# def finetune_sft_lora(
#     *,
#     base_model: str,
#     train_pairs: List[Tuple[str, str]],
#     output_model_dir: Path,
#     learning_rate: float,
#     max_steps: int,
#     batch_size: int,
#     lora_rank: int = 32,
#     save_name: str = "asi_model",
#     mode: str = "train",
# ) -> TinkerModelRef:
#     ensure_dir(output_model_dir)

#     # -------------------------------
#     # Phase 1: frozen policy
#     # -------------------------------
#     if mode == "frozen":
#         ref = TinkerModelRef(
#             base_model=base_model,
#             sampling_model_path=base_model,
#         )
#         save_model_ref(output_model_dir, ref)
#         return ref

#     # -------------------------------
#     # Phase 2: real Fireworks LoRA
#     # -------------------------------
#     assert max_steps > 0 and learning_rate > 0.0

#     train_file = output_model_dir / "train.jsonl"
#     with train_file.open("w", encoding="utf-8") as f:
#         for p, c in train_pairs:
#             f.write(json.dumps({
#                 "messages": [
#                     {"role": "user", "content": p},
#                     {"role": "assistant", "content": c},
#                 ]
#             }) + "\n")

def finetune_sft_lora(
    *,
    base_model: str,
    train_pairs: List[Tuple[str, str]],
    output_model_dir: Path,
    learning_rate: float,
    max_steps: int,
    batch_size: int,
    lora_rank: int = 32,
    save_name: str = "asi_model",
    mode: str = "train",
) -> TinkerModelRef:
    # """
    # Fireworks-compatible LoRA hook.

    # This function DOES NOT run LoRA training.
    # It only:
    #   1) writes train.jsonl
    #   2) creates a placeholder model_ref

    # Actual LoRA training is done via Fireworks UI / firectl.
    # """
    """
    This function prepares LoRA training artifacts.
    Actual LoRA training is triggered by scripts/submit_fireworks_sft.py.
    """
    ensure_dir(output_model_dir)

    # -------------------------------
    # Phase 1: frozen policy
    # -------------------------------
    if mode == "frozen":
        ref = TinkerModelRef(
            base_model=base_model,
            sampling_model_path=base_model, ## TEMP: use base model until LoRA job completes
        )
        save_model_ref(output_model_dir, ref)
        return ref

    # -------------------------------
    # Phase 2: write training data ONLY
    # -------------------------------
    train_dir = output_model_dir.parent / "train"
    ensure_dir(train_dir)

    train_file = train_dir / "train.jsonl"
    with train_file.open("w", encoding="utf-8") as f:
        for p, c in train_pairs:
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": c},
                ]
            }) + "\n")

    # write a TODO file so you don't forget what to do next
    write_json(
        train_dir / "FIREWORKS_TODO.json",
        {
            "base_model": base_model,
            "dataset_path": str(train_file),
            "lora_rank": lora_rank,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "instruction": (
                "Upload train.jsonl to Fireworks → run SFT (LoRA) → "
                "copy the resulting model_id into "
                f"{output_model_dir / MODEL_REF_FILENAME}"
            ),
        },
    )

    # placeholder model ref (NO UPDATE YET)
    ref = TinkerModelRef(
        base_model=base_model,
        sampling_model_path=base_model,  # will be overwritten manually
    )
    save_model_ref(output_model_dir, ref)

    print(f"[LoRA] Training data written to {train_file}")
    print("[LoRA] Run Fireworks SFT manually, then update sampling_model_path")

    return ref



# #train.py for fireworks

# from __future__ import annotations

# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import os




# from .tracking import ensure_dir, write_json


# MODEL_REF_FILENAME = "tinker_model_ref.json"

# _TOKENIZER_CACHE: Dict[str, object] = {}

# @dataclass
# class TinkerModelRef: # just name... too lazy to revise it:)

#     base_model: str
#     # after saving weights, Tinker returns a sampling client with a model_path/name;
#     # we store the identifier we can reuse.
#     sampling_model_path: Optional[str] = None


# def save_model_ref(model_dir: Path, ref: TinkerModelRef) -> None:
#     ensure_dir(model_dir)
#     write_json(model_dir / MODEL_REF_FILENAME, ref.__dict__)


# def load_model_ref(model_dir: Path) -> TinkerModelRef:
#     p = model_dir / MODEL_REF_FILENAME
#     if not p.exists():
#         raise FileNotFoundError(f"Missing {MODEL_REF_FILENAME} in {model_dir}")
#     d = json.loads(p.read_text())
#     return TinkerModelRef(**d)


# def create_initial_model_ref(model_dir: Path, base_model: str) -> None:
#     """
#     For iteration 0, we only have base_model; sampling_model_path will be created after first finetune.
#     """
#     save_model_ref(model_dir, TinkerModelRef(base_model=base_model, sampling_model_path=None))


# def finetune_sft_lora(
#     *,
#     base_model: str,
#     train_pairs: List[Tuple[str, str]],
#     output_model_dir: Path,
#     learning_rate: float,
#     max_steps: int,
#     batch_size: int,
#     lora_rank: int = 32,
#     save_name: str = "asi_model",
#     mode: str = "train",   # NEW
# ) -> TinkerModelRef:
#     """
#     Fireworks LoRA fine-tuning with explicit frozen/train switch.
#     """

#     ensure_dir(output_model_dir)

#     # ==========================================================
#     # Phase 1: frozen-policy (no parameter updates)
#     # ==========================================================
#     if mode == "frozen":
#         model_ref = TinkerModelRef(
#             base_model=base_model,
#             sampling_model_path=base_model,
#         )
#         save_model_ref(output_model_dir, model_ref)
#         return model_ref

#     # ==========================================================
#     # Phase 2: real LoRA fine-tuning
#     # ==========================================================
#     assert max_steps > 0 and learning_rate > 0.0, \
#         "training.mode=train but max_steps / learning_rate invalid"

#     # write training data
#     train_file = output_model_dir / "train.jsonl"
#     with train_file.open("w", encoding="utf-8") as f:
#         for prompt, completion in train_pairs:
#             f.write(json.dumps({
#                 "messages": [
#                     {"role": "user", "content": prompt},
#                     {"role": "assistant", "content": completion},
#                 ]
#             }) + "\n")

#     client = _get_fw_client()

#     # 1.upload training file
#     with open(train_file, "rb") as f:
#         upload = client.files.create(
#             file=f,
#             purpose="fine-tune",
#         )

#     # 2.launch finetune job
#     job = client.fine_tuning.jobs.create(
#         model=base_model,
#         training_file=upload.id,   # use file_id
#         hyperparameters={
#             "learning_rate": learning_rate,
#             "batch_size": batch_size,
#             "max_steps": max_steps,
#             "lora_rank": lora_rank,
#         },
#         suffix=save_name,
#     )


#     import time
#     while True:
#         job = client.fine_tuning.jobs.retrieve(job.id)
#         if job.status == "succeeded":
#             break
#         if job.status == "failed":
#             raise RuntimeError(f"Fireworks finetune failed: {job}")
#         time.sleep(30)
        
#     assert job.fine_tuned_model is not None, \
#         "Fine-tune succeeded but no fine_tuned_model returned"

#     model_ref = TinkerModelRef(
#         base_model=base_model,
#         sampling_model_path=job.fine_tuned_model,
#     )
#     save_model_ref(output_model_dir, model_ref)
#     return model_ref

# # def finetune_sft_lora(
# #     *,
# #     base_model: str,
# #     train_pairs: List[Tuple[str, str]],
# #     output_model_dir: Path,
# #     learning_rate: float,
# #     max_steps: int,
# #     batch_size: int,
# #     lora_rank: int = 32,
# #     save_name: str = "asi_model",
# # ) -> TinkerModelRef:
# #     """
# #     No-op fine-tuning for Fireworks fallback.
# #     We intentionally keep the model frozen to obtain
# #     a stable self-improvement signal under fixed policy.
# #     """
# #     # assert max_steps == 0 or learning_rate == 0.0, "finetune_sft_lora called in non-frozen mode"

# #     ensure_dir(output_model_dir)

# #     # IMPORTANT: do NOT change the model
# #     model_ref = TinkerModelRef(
# #         base_model=base_model,
# #         sampling_model_path=base_model,  # reuse base model for next iter
# #     )

# #     save_model_ref(output_model_dir, model_ref)
# #     return model_ref


# # --- add near imports ---
# import os
# import time
# from typing import List, Optional
# from openai import OpenAI

# _FW_CLIENT: Optional[OpenAI] = None

# def _get_fw_client() -> OpenAI:
#     global _FW_CLIENT
#     if _FW_CLIENT is None:
#         _FW_CLIENT = OpenAI(
#             base_url="https://api.fireworks.ai/inference",  
#             api_key=os.environ["FIREWORKS_API_KEY"],
#         )
#     return _FW_CLIENT

# def _fw_chat(
#     model: str,
#     prompt: str,
#     max_tokens: int,
#     temperature: float,
#     stop: Optional[List[str]] = None,
# ) -> str:
#     client = _get_fw_client()
#     last_err: Exception | None = None

#     for _ in range(3):
#         try:
#             resp = client.responses.create(
#                 model=model,
#                 input=prompt,
#                 temperature=temperature,
#                 max_output_tokens=max_tokens,
#                 stop=stop,
#             )
#             return resp.output_text.strip()
#         except Exception as e:
#             last_err = e
#             time.sleep(2)

#     raise RuntimeError(f"Fireworks request failed after retries: {last_err}")


# def _fw_chat(
#     model: str,
#     prompt: str,
#     max_tokens: int,
#     temperature: float,
#     stop: Optional[List[str]] = None,
# ) -> str:
#     client = _get_fw_client()
#     last_err: Exception | None = None

#     for _ in range(3):
#         try:
#             resp = client.chat.completions.create(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=temperature,
#                 # openai sdk standard param:
#                 max_tokens=max_tokens,
#                 stop=stop,
#             )
#             return (resp.choices[0].message.content or "").strip()
#         except Exception as e:
#             last_err = e
#             time.sleep(2)

#     raise RuntimeError(f"Fireworks request failed after retries: {last_err}")



'''
#train.py for tinker api

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tracking import ensure_dir, write_json


MODEL_REF_FILENAME = "tinker_model_ref.json"

_TOKENIZER_CACHE: Dict[str, object] = {}

@dataclass
class TinkerModelRef:
    """
    Local handle to a remote Tinker model.
    We store enough metadata to recreate clients.
    """
    base_model: str
    # after saving weights, Tinker returns a sampling client with a model_path/name;
    # we store the identifier we can reuse.
    sampling_model_path: Optional[str] = None


def save_model_ref(model_dir: Path, ref: TinkerModelRef) -> None:
    ensure_dir(model_dir)
    write_json(model_dir / MODEL_REF_FILENAME, ref.__dict__)


def load_model_ref(model_dir: Path) -> TinkerModelRef:
    p = model_dir / MODEL_REF_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"Missing {MODEL_REF_FILENAME} in {model_dir}")
    d = json.loads(p.read_text())
    return TinkerModelRef(**d)


def create_initial_model_ref(model_dir: Path, base_model: str) -> None:
    """
    For iteration 0, we only have base_model; sampling_model_path will be created after first finetune.
    """
    save_model_ref(model_dir, TinkerModelRef(base_model=base_model, sampling_model_path=None))


# def get_service_client():
#     import tinker  # type: ignore
#     return tinker.ServiceClient()

_SERVICE_CLIENT = None

def get_service_client():
    global _SERVICE_CLIENT
    if _SERVICE_CLIENT is None:
        import tinker  # type: ignore
        _SERVICE_CLIENT = tinker.ServiceClient()
    return _SERVICE_CLIENT


def get_training_client(base_model: str, rank: int = 32):
    """
    Create a LoRA training client on top of base_model.
    """
    import tinker  # type: ignore
    service_client = get_service_client()
    return service_client.create_lora_training_client(base_model=base_model, rank=rank)


def get_sampling_client_from_training(training_client, name: str):
    """
    Save weights and get a sampling client.
    """
    return training_client.save_weights_and_get_sampling_client(name=name)


def build_datum_sft(tokenizer, prompt: str, completion: str):
    """
    Build a single supervised learning datum: loss only on completion tokens.
    Mirrors Tinker docs pattern. :contentReference[oaicite:2]{index=2}
    """
    from tinker import types  # type: ignore

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


def finetune_sft_lora(
    *,
    base_model: str,
    train_pairs: List[Tuple[str, str]],  # (prompt, completion)
    output_model_dir: Path,
    learning_rate: float,
    max_steps: int,
    batch_size: int,
    lora_rank: int = 32,
    save_name: str = "asi_model",
) -> TinkerModelRef:
    """
    Minimal supervised fine-tuning with Tinker primitives.
    """
    import tinker  # type: ignore
    from tinker import types  # type: ignore

    ensure_dir(output_model_dir)

    training_client = get_training_client(base_model=base_model, rank=lora_rank)
    tokenizer = training_client.get_tokenizer()

    # Preprocess all examples into Datum objects (small-scale, OK for MVP).
    data = [build_datum_sft(tokenizer, p, c) for (p, c) in train_pairs]
    if not data:
        raise ValueError("No training data after filtering.")

    # Simple minibatching loop
    step = 0
    idx = 0
    while step < max_steps:
        batch = []
        for _ in range(batch_size):
            batch.append(data[idx % len(data)])
            idx += 1

        # Queue fw/bw and optim step (docs recommend submitting before waiting). :contentReference[oaicite:3]{index=3}
        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=learning_rate))

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # Optional: compute a quick weighted loss for logging/debug
        logprobs = np.concatenate([out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([ex.loss_fn_inputs["weights"].tolist() for ex in batch])
        loss_per_token = -float(np.dot(logprobs, weights) / max(weights.sum(), 1.0))

        step += 1

    # Save weights and get a sampling client for generation
    sampling_client = get_sampling_client_from_training(training_client, name=save_name)
    model_ref = TinkerModelRef(base_model=base_model, sampling_model_path=sampling_client.model_path)

    save_model_ref(output_model_dir, model_ref)
    return model_ref



def sample_text(
    *,
    model_ref: TinkerModelRef,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: Optional[List[str]] = None,
    num_samples: int = 1,
) -> List[str]:
    """
    Sampling:
    - Iter-0 (no sampling_model_path yet): use a training client to obtain a valid sampling client + tokenizer.
    - Iter-1+ (sampling_model_path exists): use ServiceClient sampling client (no training client).
    """
    from tinker import types  # type: ignore

    service_client = get_service_client()

    if model_ref.sampling_model_path is None:
        # Iter-0: base model name is NOT a valid service sampling path.
        # Use training client to get a sampling client (this returns a valid tinker model_path).
        training_client = get_training_client(base_model=model_ref.base_model, rank=32)
        tokenizer = training_client.get_tokenizer()
        sampling_client = training_client.save_weights_and_get_sampling_client(name="iter0_base_sampler")
    else:
        # Iter-1+: sampling_model_path should be a valid tinker path
        sampling_client = service_client.create_sampling_client(model_path=model_ref.sampling_model_path)
        tokenizer = get_tokenizer(model_ref.base_model)

    mi = types.ModelInput.from_ints(tokenizer.encode(prompt))
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or [],
    )
    fut = sampling_client.sample(prompt=mi, sampling_params=params, num_samples=num_samples)
    res = fut.result()
    return [tokenizer.decode(seq.tokens) for seq in res.sequences]

def get_tokenizer(base_model: str):
    """
    Get tokenizer in a safe way.
    We deliberately use a training client ONLY to fetch the tokenizer,
    because create_sampling_client requires a valid tinker model_path.
    """
    if base_model not in _TOKENIZER_CACHE:
        training_client = get_training_client(
            base_model=base_model,
            rank=1,  # minimal rank, tokenizer-only usage
        )
        _TOKENIZER_CACHE[base_model] = training_client.get_tokenizer()
    return _TOKENIZER_CACHE[base_model]
'''