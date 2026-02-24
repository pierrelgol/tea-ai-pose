from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class BinarySearchResult:
    best_value: int | None
    attempts: list[int]


# Finds the maximum feasible integer in [low, high] using a monotonic feasibility predicate.
def binary_search_max_feasible(
    *,
    low: int,
    high: int,
    is_feasible: Callable[[int], bool],
    max_attempts: int | None = None,
) -> BinarySearchResult:
    if low > high:
        return BinarySearchResult(best_value=None, attempts=[])

    lo = int(low)
    hi = int(high)
    best: int | None = None
    attempts: list[int] = []

    while lo <= hi:
        if max_attempts is not None and len(attempts) >= int(max_attempts):
            break
        mid = (lo + hi) // 2
        attempts.append(mid)
        if is_feasible(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return BinarySearchResult(best_value=best, attempts=attempts)
