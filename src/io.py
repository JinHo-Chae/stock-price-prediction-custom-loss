import json
from typing import Dict, Tuple

Stats = Dict[str, Tuple[float, float]]  # {"open": (m,s), ...}

def save_stats(path: str, stats_dict: dict):
    """
    stats_dict 예:
    {
      "019170.KS": {"open":[m,s], "high":[m,s], ...},
      ...
    }
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)

def load_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
