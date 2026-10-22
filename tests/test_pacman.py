import numpy as np

from tinyworldformer.games import pacman as p


def state(pac=(11, 19), heading=0, a=(1, 1), b=(21, 1), steps=0, pellets=None, lives=p.LIVES):
    pellets = ~p.WALLS if pellets is None else pellets
    pellets = pellets.copy()
    pellets[pac[1], pac[0]] = False
    return p.State(pellets, pac, heading, a, b, lives, steps)


def test_maze_is_symmetric_connected_without_dead_ends():
    assert (p.WALLS == p.WALLS[:, ::-1]).all()
    assert (p.DIST != 255).all()
    for x, y in p.FREE:
        assert sum((x + dx, y + dy) in p.INDEX for dx, dy in p.DIRS) >= 2
    assert len(p.FREE) - 1 >= 200  # hundreds of pellets


def test_frame_has_one_of_each_actor():
    s = p.reset(np.random.default_rng(0))
    g = s.grid
    assert g.shape == (p.H, p.W) and g.max() < p.NUM_TOKENS
    assert (g >= p.PAC).sum() == 1
    assert ((g == p.GHOST_A) | (g == p.GHOST_A + 1)).sum() == 1
    assert ((g == p.GHOST_B) | (g == p.GHOST_B + 1)).sum() == 1
    assert (g[p.HUD_ROW, p.LIFE_COLS] == p.LIFE).sum() == p.LIVES
    assert (g == p.PELLET).sum() == s.pellets.sum() - s.pellets[s.a[1], s.a[0]] - s.pellets[s.b[1], s.b[0]]


def test_sprites():
    s = state(pac=(5, 1), heading=1, a=(3, 1), b=(21, 19))
    g = s.grid
    assert (g[2:4, 10:12] == [[p.PAC_BODY, p.PAC_BODY], [p.PAC_BODY, p.PAC + 1]]).all()  # mouth east: lower right
    assert (g[2:4, 6:8] == [[p.EYE, p.EYE], [p.GHOST_A + 1, p.BODY_A]]).all()  # on a pellet
    s = state(a=(3, 1), pellets=np.zeros_like(p.WALLS))
    assert s.grid[3, 6] == p.GHOST_A


def test_turn_keep_going_and_buffered_turn():
    s = state(pac=(5, 1), heading=1, a=(1, 19), b=(21, 19))
    n, _, _ = p.step(s, p.KEEP)
    assert n.pac == (6, 1) and n.heading == 1
    n, _, _ = p.step(s, p.DOWN)  # (5, 2) is open
    assert n.pac == (5, 2) and n.heading == 2
    n, _, _ = p.step(s, p.UP)  # wall above: keeps going east
    assert n.pac == (6, 1) and n.heading == 1
    s = state(pac=(21, 1), heading=1, a=(1, 19), b=(11, 19))
    n, _, _ = p.step(s, p.KEEP)  # wall ahead: stays
    assert n.pac == (21, 1)


def test_eating_a_pellet_pays_and_removes_it():
    s = state(pac=(5, 1), heading=1, a=(1, 19), b=(21, 19))
    n, r, _ = p.step(s, p.KEEP)
    assert r == 1.0 and not n.pellets[1, 6] and n.pellets.sum() == s.pellets.sum() - 1
    n2, r, _ = p.step(n, p.LEFT)  # back onto the cell Pac-Man started on: nothing there
    assert r == 0.0 and (n2.pellets == n.pellets).all()


def test_ghosts_take_turns_and_chase_along_shortest_paths():
    s = state(pac=(1, 19), heading=1, a=(11, 1), b=(11, 5), steps=0)
    n, _, _ = p.step(s, p.KEEP)
    assert n.b == s.b and n.a != s.a
    assert p.dist(n.a, n.pac) == p.dist(s.a, n.pac) - 1
    n2, _, _ = p.step(n, p.KEEP)
    assert n2.a == n.a and n2.b != n.b
    assert p.dist(n2.b, n2.pac) == p.dist(n.b, n2.pac) - 1


def test_a_ghost_does_not_step_onto_the_other_ghost():
    s = state(pac=(1, 1), a=(1, 5), b=(1, 4), steps=0, heading=2)
    n, _, _ = p.step(s, p.UP)  # pac wall-blocked north of (1,1)... stays heading
    assert n.a != n.b


def test_caught_costs_a_life_and_respawns_far_away():
    rng = np.random.default_rng(0)
    s = state(pac=(5, 1), heading=1, a=(7, 1), b=(21, 19), steps=0)
    n, r, done = p.step(s, p.KEEP, rng)  # pac to (6,1), ghost A steps to (6,1)
    assert r == -1.0 and n.lives == p.LIVES - 1 and not done and n.random_event
    assert n.pac == p.START and n.heading == 0
    assert n.a[0] < p.MAP_W // 2 < n.b[0] and p.dist(p.START, n.a) >= 10 and p.dist(p.START, n.b) >= 10
    s = state(pac=(5, 1), heading=1, a=(6, 1))  # walking into a ghost
    n, r, _ = p.step(s, p.KEEP, rng)
    assert r == -1.0 and n.random_event
    s = state(pac=(5, 1), heading=1, a=(7, 1), lives=1)
    assert p.step(s, p.KEEP, rng)[2]  # last life


def test_clearing_the_last_pellet_ends_the_game():
    pellets = np.zeros_like(p.WALLS)
    pellets[1, 6] = True
    s = state(pac=(5, 1), heading=1, a=(1, 19), b=(21, 19), pellets=pellets)
    n, r, done = p.step(s, p.KEEP)
    assert r == 1.0 and done and n.left() == 0


def test_frames_read_back_into_the_state():
    from tinyworldformer.games.agents import forager_agent

    rng = np.random.default_rng(0)
    pol, s = forager_agent(rng, 0.2), p.reset(rng)
    for _ in range(1500):
        got = p.read(s.grid)
        assert got is not None
        assert (got.pellets == s.pellets).all()
        assert (got.pac, got.heading, got.a, got.b, got.lives, got.steps) == (s.pac, s.heading, s.a, s.b, s.lives, s.steps % 2)
        s, _, done = p.step(s, pol(s), rng)
        if done:
            s = p.reset(rng, varied=rng.random() < 0.5)


def test_observe_features():
    from tinyworldformer.training.streamx import PACMAN_OBS, PM_PER_DIR, observe_pacman

    s = state(pac=(5, 1), heading=1, a=(5, 5), b=(21, 19), steps=1)
    obs = observe_pacman(s.grid)
    assert obs.shape == (PACMAN_OBS,)
    assert obs[0] == 0 and obs[PM_PER_DIR] == 1 and obs[2 * PM_PER_DIR] == 1  # north is a wall; east and south are open
    assert obs[4 * PM_PER_DIR + 1] == 1  # heading east
    assert observe_pacman(np.full((p.H, p.W), p.HUD, np.uint8)).sum() == 0
    assert obs[4 * PM_PER_DIR + 4] == 0  # ghost B moves next
    assert obs[4 * PM_PER_DIR + 5 : 4 * PM_PER_DIR + 9].tolist() == [0, 1, 1, 1]  # ghost A: 4 steps away
