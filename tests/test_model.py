import numpy as np
import pytest
import torch

from tinyworldformer import games
from tinyworldformer.training import export
from tinyworldformer.rustexport import intref
from tinyworldformer.training.model import ActQuant, Config, WorldModel, decode, patchify, unpatchify

GAMES = list(games.GAMES)


@pytest.mark.parametrize("name", GAMES)
def test_patchify_roundtrip(name):
    g = games.get(name)
    grid = torch.from_numpy(g.module.reset(np.random.default_rng(0)).grid)[None].long()
    p = patchify(grid, g)
    assert p.shape == (1, g.T, g.SUB)
    assert (unpatchify(p, g) == grid).all()


@pytest.mark.parametrize("name", GAMES)
def test_decode_places_each_singleton_exactly_once(name):
    g = games.get(name)
    torch.manual_seed(0)
    cells = torch.randn(16, g.T, g.SUB, g.V)
    for gen in (None, torch.Generator().manual_seed(0)):
        grid = decode(cells, g, gen)
        for lo, hi, _ in g.singletons:
            assert (((grid >= lo) & (grid < hi)).flatten(1).sum(1) == 1).all()


@pytest.mark.parametrize("name", GAMES)
def test_integer_spec_tracks_qat_model(name):
    torch.manual_seed(0)
    model = WorldModel(Config(loops=2, game=name))
    g = model.g
    rng = np.random.default_rng(0)
    grids = np.stack([g.module.reset(rng).grid for _ in range(8)])
    action = rng.integers(g.A, size=8)
    ActQuant.enabled = True
    model.train()
    with torch.no_grad():  # calibrate the activation scales
        for _ in range(20):
            model(torch.from_numpy(grids), torch.from_numpy(action))
    model.eval()
    m = export.to_int(model)
    patches = patchify(torch.from_numpy(grids).long(), g).numpy()
    cells_int, _ = intref.forward(m, patches, action, 2)
    with torch.no_grad():
        cells, _ = model(torch.from_numpy(grids), torch.from_numpy(action), 2)[-1]
    # untrained weights, so compare logits loosely: same scale, high correlation
    assert np.corrcoef(cells_int.reshape(-1), cells.reshape(-1).numpy())[0, 1] > 0.98
