import numpy as np

from tinyworldformer.games import breakout as b
from tinyworldformer.training import data
from tinyworldformer.training.dataset import Transitions


def test_batches(tmp_path):
    np.savez(tmp_path / "t.npz", **data.generate(transitions=600, max_steps=100, seed=0))
    ds = Transitions(tmp_path / "t.npz")
    batches = list(ds.batches(64, np.random.default_rng(0)))
    assert len(batches) == len(ds) // 64
    x = batches[0]
    assert x["grid"].shape == (64, b.H, b.W) and x["changed"].shape == (64, b.H, b.W)
    assert x["reward_cls"].max() <= 3 and x["next_lives"].max() <= 3
    assert x["changed"].any(axis=(1, 2)).all()  # the ball moves every step
    assert x["event"].dtype == bool
