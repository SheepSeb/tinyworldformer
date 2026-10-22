import numpy as np

from tinyworldformer.games import pong as g


def test_frame_has_one_ball_and_two_paddles():
    s = g.reset(np.random.default_rng(0))
    f = s.grid
    assert f.shape == (g.H, g.W) and f.max() < g.NUM_TOKENS
    assert (f >= g.BALL).sum() == 1
    assert (f == g.PADDLE).sum() == g.PADDLE_LEN and (f == g.CPU).sum() == g.PADDLE_LEN
    assert (f == g.SLOT_P).sum() == g.POINTS and (f == g.SLOT_C).sum() == g.POINTS


def test_ball_moves_and_bounces_off_walls():
    s = g.State(10, g.TOP, -1, -1, 2, 2)
    n, _, _ = g.step(s, g.STAY)
    assert (n.bx, n.by, n.dx, n.dy) == (9, g.TOP + 1, -1, 1)


def test_paddle_hit_angles_the_ball():
    s = g.State(g.PLAYER_X + 1, 8, -1, 0, 8, 8)  # ball meets the paddle's top cell
    n, r, _ = g.step(s, g.STAY)
    assert r == 0 and (n.bx, n.by, n.dx, n.dy) == (g.PLAYER_X + 1, 8, 1, -1)
    s = g.State(g.PLAYER_X + 1, 9, -1, 0, 8, 8)  # middle: the slope stays
    n, _, _ = g.step(s, g.STAY)
    assert (n.dx, n.dy) == (1, 0)


def test_points_serve_toward_the_loser():
    rng = np.random.default_rng(0)
    s = g.State(g.PLAYER_X + 1, 3, -1, 0, 12, 8)
    n, r, done = g.step(s, g.STAY, rng)
    assert r == -1 and n.lost == 1 and n.random_event and n.dx == -1 and n.bx == g.NET_X and not done
    s = g.State(g.CPU_X - 1, 3, 1, 0, 8, 12)
    n, r, _ = g.step(s, g.STAY, rng)
    assert r == 1 and n.score == 1 and n.dx == 1
    s = g.State(g.CPU_X - 1, 3, 1, 0, 8, 12, score=g.POINTS - 1)
    assert g.step(s, g.STAY, rng)[2]


def test_cpu_waits_then_tracks():
    s = g.State(10, 3, 1, 0, 8, 12)
    assert g.step(s, g.STAY)[0].cpu == 12  # ball far away
    s = g.State(g.REACT_X + 2, 3, 1, 0, 8, 12)  # (the CPU pauses on every third column)
    assert g.step(s, g.STAY)[0].cpu == 11


def test_paddle_stops_at_the_walls():
    s = g.State(10, 5, 1, 0, g.TOP, 8)
    assert g.step(s, g.UP)[0].player == g.TOP


def test_frames_read_back_into_the_state():
    from tinyworldformer.games.agents import returner_agent

    rng = np.random.default_rng(0)
    pol, s = returner_agent(rng, 0.3, 0.5), g.reset(rng)
    for _ in range(3000):
        got = g.read(s.grid)
        assert (got.bx, got.by, got.dx, got.dy, got.player, got.cpu, got.score, got.lost) == \
            (s.bx, s.by, s.dx, s.dy, s.player, s.cpu, s.score, s.lost)
        s, _, done = g.step(s, pol(s), rng)
        if done:
            s = g.reset(rng, varied=rng.random() < 0.5)


def test_observe():
    from tinyworldformer.training.streamx import PONG_OBS, observe_pong

    s = g.State(10, 5, 1, -1, 8, 3)
    obs = observe_pong(s.grid)
    assert obs.shape == (PONG_OBS,) and obs.sum() == 6
    assert obs[5] == 1 and obs[g.H + 10] == 1 and obs[g.H + g.W + 3] == 1
