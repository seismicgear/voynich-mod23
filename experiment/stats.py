"""
stats.py

Statistical helper functions.
"""

import statistics

def calculate_stats(observed: float, null_dist: list[float], smaller_is_better: bool = False) -> dict:
    """
    Calculate statistical metrics comparing an observed value to a null distribution.

    Returns:
        dict: {
            "mean": float,
            "std": float,
            "z_score": float,
            "p_value": float
        }
    """
    if not null_dist:
        return {
            "mean": 0.0, "std": 0.0, "z_score": 0.0, "p_value": 1.0
        }

    mean = statistics.mean(null_dist)
    std = statistics.pstdev(null_dist)
    z_score = (observed - mean) / std if std > 0 else 0.0

    # P-value with add-one smoothing
    n = len(null_dist)
    if smaller_is_better:
        k = sum(1 for x in null_dist if x <= observed)
    else:
        k = sum(1 for x in null_dist if x >= observed)

    p_value = (k + 1) / (n + 1)

    return {
        "mean": mean,
        "std": std,
        "z_score": z_score,
        "p_value": p_value
    }
