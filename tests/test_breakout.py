import numpy as np

from tinyworldformer.games import breakout as b


def run(policy, rng, max_steps=5000):
    s, total = b.reset(rng), 0.0
    for t in range(max_steps):
        s, r, done = b.step(s, policy(s), rng)
        total += r
        if done:
            break
    return s, total, t


def tracker(s):
    _, bc = b._ball_pos(s.grid)
    pc = b._paddle_center(s.grid)
    return b.RIGHT if bc > pc else b.LEFT if bc < pc else b.STAY


def test_reset_has_one_ball_and_full_bricks():
    s = b.reset(np.random.default_rng(0))
    assert (s.grid >= b.BALL_UL).sum() == 1
    assert (s.grid == b.BRICK).sum() == len(b.BRICK_ROWS) * (b.W - 2)
    assert (s.grid == b.PADDLE).sum() == b.PADDLE_W


def test_step_is_pure_and_deterministic():
    s = b.reset(np.random.default_rng(0))
    before = s.grid.copy()
    a, _, _ = b.step(s, b.LEFT, np.random.default_rng(1))
    c, _, _ = b.step(s, b.LEFT, np.random.default_rng(1))
    assert (s.grid == before).all() and (a.grid == c.grid).all()


def test_invariants_under_random_play():
    rng = np.random.default_rng(0)
    s = b.reset(rng)
    bricks = (s.grid == b.BRICK).sum()
    for _ in range(3000):
        s, _, done = b.step(s, int(rng.integers(3)), rng)
        assert (s.grid >= b.BALL_UL).sum() == 1
        assert (s.grid == b.PADDLE).sum() == b.PADDLE_W
        n = (s.grid == b.BRICK).sum()
        assert n <= bricks
        bricks = n
        assert (s.grid[0] == b.WALL).all()
        if done:
            s = b.reset(rng)
            bricks = (s.grid == b.BRICK).sum()


def test_idle_agent_loses_all_lives():
    s, total, _ = run(lambda s: b.STAY, np.random.default_rng(0))
    assert s.lives <= 0


def test_tracker_clears_bricks():
    s, total, _ = run(tracker, np.random.default_rng(0), max_steps=20000)
    assert (s.grid == b.BRICK).sum() == 0 and s.lives > 0


def test_render_shapes():
    s = b.reset(np.random.default_rng(0))
    assert b.render(s, cell=4).shape == (b.H * 4, b.W * 4, 3)
    assert len(b.render_ascii(s).splitlines()) == b.H


def test_colorize_and_ansi():
    s = b.reset(np.random.default_rng(0))
    img = b.colorize(s.grid)
    assert img.shape == (b.H, b.W, 3)
    # bricks are coloured by row
    first, second = (img[r, 3] for r in list(b.BRICK_ROWS)[:2])
    assert not (first == second).all()
    assert b.render_ansi(s.grid).count("\n") == b.H - 1
