import numpy as np

from tinyworldformer.games import flappy as f


def test_frame_has_one_bird_and_one_marker():
    s = f.reset(np.random.default_rng(0))
    g = s.grid
    assert g.shape == (f.H, f.W) and g.max() < f.NUM_TOKENS
    assert (g >= f.BIRD).sum() == 1 and (g == f.MARKER).sum() == 1
    assert g[s.y, f.BIRD_X] == f.BIRD + s.v + 2


def test_gravity_and_flap():
    s = f.State(10, 0, (), 5, 0)
    n, _, _ = f.step(s, f.NOOP)
    assert (n.y, n.v) == (11, 1)
    n, _, _ = f.step(n, f.FLAP)
    assert (n.y, n.v) == (9, -2)


def test_world_scrolls_left():
    rng = np.random.default_rng(0)
    s = f.State(10, 0, ((15, 5),), 5, 0)
    n, _, _ = f.step(s, f.FLAP, rng)
    assert n.pipes[0][0] == 14
    a, b = s.grid, n.grid
    assert (a[f.GROUND + 1, 1 : f.PLAY_W] == b[f.GROUND + 1, : f.PLAY_W - 1]).all()  # ground stripes shift


def test_pipes_spawn_from_the_marker_with_spacing():
    rng = np.random.default_rng(0)
    s = f.State(10, 0, ((f.PLAY_W - f.PIPE_W - f.SPACING + 1, 5),), 8, 0)
    n, _, _ = f.step(s, f.NOOP, rng)
    assert n.random_event and n.pipes[-1] == (f.PLAY_W - f.PIPE_W, 8)
    assert n.pipes[-1][0] - n.pipes[0][0] == f.SPACING


def test_passing_scores_and_crashing_ends():
    s = f.State(10, 0, ((f.BIRD_X - f.PIPE_W + 1, 7),), 5, 0)  # gap rows 7..13, bird inside
    n, r, done = f.step(s, f.FLAP)
    assert r == 1.0 and not done
    s = f.State(10, 0, ((f.BIRD_X, 15),), 5, 0)  # bird above the gap: crash
    n, r, done = f.step(s, f.NOOP)
    assert r == -1.0 and done and not n.alive


def test_observe_reads_bird_and_next_pipe():
    from tinyworldformer.training.streamx import FB_DIST, FB_ROWS, FLAPPY_OBS, observe_flappy

    s = f.State(10, 1, ((12, 4),), 6, 0)
    obs = observe_flappy(s.grid)
    assert obs.shape == (FLAPPY_OBS,) and obs[10] == 1 and obs[FB_ROWS + 3] == 1  # row 10, speed +1
    k = FB_ROWS + 5
    assert obs[k + 12 - (f.BIRD_X - 1)] == 1  # next pipe 6 columns ahead
    assert obs[k + FB_DIST + 4 + f.GAP // 2 - 10 + FB_ROWS - 1] == 1  # its gap centre, relative


def test_frames_read_back_into_the_state():
    from tinyworldformer.games.agents import pilot_agent

    rng = np.random.default_rng(0)
    pol, s = pilot_agent(rng, 0.2), f.reset(rng)
    for _ in range(2000):
        got = f.read(s.grid)
        visible = tuple(p for p in s.pipes if p[0] + f.PIPE_W > 0)
        assert (got.y, got.v, got.pipes, got.next_gap, got.phase) == (s.y, s.v, visible, s.next_gap, s.phase)
        s, _, done = f.step(s, pol(s), rng)
        if done:
            s = f.reset(rng)
