import numpy as np

from tinyworldformer.games import breakout as b
from tinyworldformer.training import data


def test_generate_matches_engine():
    d = data.generate(transitions=500, max_steps=100, seed=0)
    n = len(d["action"])
    assert d["grid"].shape == (n, b.H, b.W) and d["next_grid"].shape == (n, b.H, b.W)
    # within an episode, next_grid of step t is grid of step t+1
    same_ep = d["episode"][1:] == d["episode"][:-1]
    assert (d["next_grid"][:-1][same_ep] == d["grid"][1:][same_ep]).all()
    # replaying a stored transition in the engine reproduces it (no serve involved)
    i = int(np.flatnonzero(d["reward"] >= 0)[10])
    s = b.State(d["grid"][i], int(d["lives"][i]))
    nxt, r, _ = b.step(s, int(d["action"][i]))
    assert (nxt.grid == d["next_grid"][i]).all() and r == d["reward"][i]
