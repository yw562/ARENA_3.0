import csv
from pathlib import Path
from typing import List, Optional


def load_advbench_prompts(
    path: Path,
    *,
    limit: Optional[int] = None,
) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["goal"])
            if limit and len(prompts) >= limit:
                break
    return prompts
