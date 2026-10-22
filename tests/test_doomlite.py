import numpy as np

from tinyworldformer.games import doomlite as d
from tinyworldformer.training.streamx import DOOM_OBS, observe_doom


def test_render_shape_and_hud():
    s = d.reset(np.random.default_rng(0))
    g = s.grid
    assert g.shape == (d.H, d.W) and g.max() < d.NUM_TOKENS
    mini = g[d.VIEW_H :, d.W - d.MAP_N :]
    assert (mini >= d.MAP_PLAYER).sum() == 1 and (mini == d.MAP_ENEMY).sum() == 1
    assert (g[27] == d.HEALTH_BAR).sum() == 3 * d.HEALTH


def test_enemy_sprite_is_on_the_correct_side():
    """Every visible enemy shows up left or right of centre exactly as its sideways offset says."""
    checked = 0
    for px, py in d.FREE:
        for facing in range(4):
            dx, dy = d.DIRS[facing]
            for ex, ey in d.FREE:
                if (ex, ey) == (px, py):
                    continue
                cols = np.flatnonzero(np.isin(d.render(d.State(px, py, facing, ex, ey, 5))[: d.VIEW_H], (d.ENEMY, d.ENEMY_EYE)).any(0))
                right = (ex - px) * -dy + (ey - py) * dx
                if len(cols) and right != 0:
                    assert np.sign(cols.mean() + 0.5 - d.W / 2) == np.sign(right), (px, py, facing, ex, ey)
                    checked += 1
    assert checked > 20  # narrow maze: only a few dozen off-centre sightlines exist


def test_fire_kills_in_sight_and_respawns_far():
    rng = np.random.default_rng(0)
    s = d.State(1, 5, 0, 1, 3, 5)  # enemy two cells straight ahead
    n, r, _ = d.step(s, d.FIRE, rng)
    assert r == 1.0 and n.flash and abs(n.ex - n.px) + abs(n.ey - n.py) >= 5


def test_enemy_chases_and_hits_every_second_step():
    s = d.State(1, 5, 0, 1, 3, 5, steps=1)  # enemy's turn
    assert (s.grid[29:31, 20:22] == d.ENEMY).all()  # the HUD light says so
    n, r, _ = d.step(s, d.NOOP)  # enemy steps next to the player
    assert (n.ex, n.ey) == (1, 4) and r == 0
    n, r, _ = d.step(n, d.NOOP)  # its off step: nothing happens
    assert r == 0 and n.health == 5
    n, r, _ = d.step(n, d.NOOP)  # then it hits
    assert r == -1.0 and n.health == 4


def test_observe_reads_the_frame():
    s = d.State(1, 5, 0, 1, 3, 5)
    obs = observe_doom(s.grid)
    assert obs.shape == (DOOM_OBS,)
    assert obs[5 * 8 + 1] == 1 and obs[64 + 0] == 1  # player cell, facing north
    assert obs[-1] == 1  # enemy in the crosshair
