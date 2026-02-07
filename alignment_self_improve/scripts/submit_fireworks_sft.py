import os
import time
import json
import requests
from pathlib import Path

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
BASE_URL = "https://api.fireworks.ai/v1"
HEADERS = {
    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    "Content-Type": "application/json",
}

def submit_sft(iter_dir: Path):
    todo = json.loads((iter_dir / "train" / "FIREWORKS_TODO.json").read_text())

    # ---------- 1. create dataset ----------
    ds_id = f"{iter_dir.name}-{int(time.time())}"
    r = requests.post(
        f"{BASE_URL}/datasets",
        headers=HEADERS,
        json={"datasetId": ds_id, "dataset": {"userUploaded": {}}},
    )
    r.raise_for_status()

    # ---------- 2. upload file ----------
    with open(todo["dataset_path"], "rb") as f:
        r = requests.post(
            f"{BASE_URL}/datasets/{ds_id}:upload",
            headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
            files={"file": f},
        )
        r.raise_for_status()

    dataset_name = r.json()["name"]  # IMPORTANT

    # ---------- 3. submit job ----------
    r = requests.post(
        f"{BASE_URL}/fineTuningJobs",
        headers=HEADERS,
        json={
            "base_model": todo["base_model"],
            "training_dataset": dataset_name,
            "hyperparameters": {
                "learning_rate": todo["learning_rate"],
                "batch_size": todo["batch_size"],
                "max_steps": todo["max_steps"],
                "lora_rank": todo["lora_rank"],
            },
        },
    )
    r.raise_for_status()
    job_name = r.json()["name"]

    # ---------- 4. poll ----------
    while True:
        r = requests.get(f"{BASE_URL}/{job_name}", headers=HEADERS)
        r.raise_for_status()
        state = r.json()["state"]
        print("[SFT]", state)

        if state == "SUCCEEDED":
            model_id = r.json()["model"]
            break
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(r.text)

        time.sleep(30)

    # ---------- 5. update model_ref ----------
    ref_path = iter_dir / "model" / "tinker_model_ref.json"
    ref = json.loads(ref_path.read_text())
    ref["sampling_model_path"] = model_id
    ref["lora_job_id"] = job_name
    ref_path.write_text(json.dumps(ref, indent=2))

    print("DONE:", model_id)
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python submit_fireworks_sft.py <iter_dir>")
        sys.exit(1)

    iter_dir = Path(sys.argv[1])
    submit_sft(iter_dir)
