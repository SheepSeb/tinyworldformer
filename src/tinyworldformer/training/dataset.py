"""Batch iterator over the transitions written by `tinyworldformer.training.data` (numpy only).

Convert with e.g. `torch.from_numpy(batch["grid"]).long()` on the training side.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

REWARD_OFFSET = 1  # every game's rewards start at -1: reward -> class index


class Transitions:
    def __init__(self, path: str | Path):
        with np.load(path) as f:
            self.d = {k: f[k] for k in f.files if k != "game"}
            self.game = str(f["game"]) if "game" in f.files else "breakout"  # older files: Breakout
        self.d["reward_cls"] = (self.d["reward"] + REWARD_OFFSET).astype(np.uint8)
        if "random" not in self.d:  # recorded before the per-step flag existed
            from tinyworldformer.games import get

            self.d["random"] = np.isin(self.d["reward"], get(self.game).random_rewards)

    def __len__(self) -> int:
        return len(self.d["action"])

    def batches(
        self, batch_size: int, rng: np.random.Generator | None = None, drop_last: bool = True
    ) -> Iterator[dict[str, np.ndarray]]:
        """One pass over the data; shuffled when `rng` is given, in order otherwise.

        Each batch holds the stored fields plus:
          reward_cls (B,)       reward + 1, a class index
          changed    (B, H, W)  cells where next_grid differs from grid (the informative ones)
          event      (B,)       reward != 0 or lives changed (hits, misses, serves)
        """
        order = rng.permutation(len(self)) if rng is not None else np.arange(len(self))
        stop = len(self) - batch_size + 1 if drop_last else len(self)
        for i in range(0, stop, batch_size):
            idx = np.sort(order[i : i + batch_size])  # sorted: cheaper gather, order is irrelevant
            batch = {k: v[idx] for k, v in self.d.items()}
            batch["changed"] = batch["grid"] != batch["next_grid"]
            batch["event"] = (batch["reward"] != 0) | (batch["lives"] != batch["next_lives"])
            yield batch
