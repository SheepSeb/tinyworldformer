"""Scripted policies used to collect transitions. Each factory returns a fresh policy."""

from collections.abc import Callable

import numpy as np

from tinyworldformer.games import breakout as b, doomlite as dl, flappy as fb, pacman as pm, pong as pg

Policy = Callable[[b.State], int]


def random_agent(rng: np.random.Generator) -> Policy:
    return lambda s: int(rng.integers(b.NUM_ACTIONS))


def sticky_agent(rng: np.random.Generator, repeat: float = 0.9) -> Policy:
    """Holds its last action with probability `repeat`, like a player holding a key."""
    last = b.STAY

    def policy(s: b.State) -> int:
        nonlocal last
        if rng.random() > repeat:
            last = int(rng.integers(b.NUM_ACTIONS))
        return last

    return policy


def tracker_agent(rng: np.random.Generator, eps: float = 0.0) -> Policy:
    """Moves under the ball; with probability `eps` takes a random action instead."""

    def policy(s: b.State) -> int:
        if rng.random() < eps:
            return int(rng.integers(b.NUM_ACTIONS))
        ball_col = b._ball_pos(s.grid)[1]
        paddle_col = b._paddle_center(s.grid)
        return b.RIGHT if ball_col > paddle_col else b.LEFT if ball_col < paddle_col else b.STAY

    return policy


# name -> (factory, share of the dataset's transitions)
AGENTS: dict[str, tuple[Callable[[np.random.Generator], Policy], float]] = {
    "random": (random_agent, 0.25),
    "sticky": (sticky_agent, 0.25),
    "tracker": (lambda rng: tracker_agent(rng, 0.0), 0.1),
    "noisy_tracker": (lambda rng: tracker_agent(rng, rng.uniform(0.0, 0.5)), 0.4),
}


def hunter_agent(rng: np.random.Generator, eps: float = 0.0) -> Policy:
    """Doom-lite: fire when the enemy is in sight, otherwise walk the shortest path toward it."""

    def policy(s: dl.State) -> int:
        if rng.random() < eps:
            return int(rng.integers(dl.NUM_ACTIONS))
        if dl._in_sight(s):
            return dl.FIRE
        nx, ny = dl._bfs_step(s.px, s.py, s.ex, s.ey)  # first step from the player toward the enemy
        want = dl.DIRS.index((nx - s.px, ny - s.py))
        if want == s.facing:
            return dl.FORWARD
        return dl.RIGHT if want == (s.facing + 1) % 4 else dl.LEFT

    return policy


def _random(n: int) -> Callable[[np.random.Generator], Policy]:
    return lambda rng: (lambda s: int(rng.integers(n)))


def _sticky(n: int, repeat: float = 0.8) -> Callable[[np.random.Generator], Policy]:
    def make(rng: np.random.Generator) -> Policy:
        last = 0

        def policy(s) -> int:
            nonlocal last
            if rng.random() > repeat:
                last = int(rng.integers(n))
            return last

        return policy

    return make


DOOM_AGENTS: dict[str, tuple[Callable[[np.random.Generator], Policy], float]] = {
    "random": (_random(dl.NUM_ACTIONS), 0.25),
    "sticky": (_sticky(dl.NUM_ACTIONS), 0.2),
    "hunter": (lambda rng: hunter_agent(rng, 0.0), 0.15),
    "noisy_hunter": (lambda rng: hunter_agent(rng, rng.uniform(0.0, 0.5)), 0.4),
}


def pilot_agent(rng: np.random.Generator, eps: float = 0.0) -> Policy:
    """Flappy Bird: flap when gravity would take the bird below the next gap's centre."""

    def policy(s: fb.State) -> int:
        if rng.random() < eps:
            return int(rng.random() < 0.3)
        ahead = [(x, top) for x, top in s.pipes if x + fb.PIPE_W > fb.BIRD_X]
        target = (ahead[0][1] if ahead else s.next_gap) + fb.GAP // 2 + 1
        return fb.FLAP if s.y + min(s.v + 1, 2) > target else fb.NOOP

    return policy


FLAPPY_AGENTS: dict[str, tuple[Callable[[np.random.Generator], Policy], float]] = {
    "random": (lambda rng: (lambda s: int(rng.random() < 0.25)), 0.2),  # flapping half the time hugs the ceiling
    "sticky": (_sticky(fb.NUM_ACTIONS, 0.6), 0.1),
    "pilot": (lambda rng: pilot_agent(rng, 0.0), 0.2),
    "noisy_pilot": (lambda rng: pilot_agent(rng, rng.uniform(0.0, 0.3)), 0.5),
}


_XS, _YS = np.array([c[0] for c in pm.FREE]), np.array([c[1] for c in pm.FREE])


def forager_agent(rng: np.random.Generator, eps: float = 0.0, fear: float = 5.0) -> Policy:
    """Pac-Man lite: step toward the nearest pellet, away from ghosts that are within 3 steps of the
    next cell (`fear` is how much that matters); with probability `eps` a random action."""

    def policy(s: pm.State) -> int:
        if rng.random() < eps:
            return int(rng.integers(pm.NUM_ACTIONS))
        pellets = s.pellets[_YS, _XS]
        best, best_cost = pm.KEEP, 1e9
        for d, (dx, dy) in enumerate(pm.DIRS):
            n = (s.pac[0] + dx, s.pac[1] + dy)
            if n not in pm.INDEX:
                continue
            cost = float(pm.DIST[pm.INDEX[n]][pellets].min()) if pellets.any() else 0.0
            for g in (s.a, s.b):
                cost += fear * max(0, 4 - pm.dist(g, n))
            cost += 0.1 * (d == (s.heading + 2) % 4)  # reversing is the last resort
            if cost < best_cost:
                best, best_cost = d + 1, cost
        return best

    return policy


PACMAN_AGENTS: dict[str, tuple[Callable[[np.random.Generator], Policy], float]] = {
    "random": (_random(pm.NUM_ACTIONS), 0.15),
    "sticky": (_sticky(pm.NUM_ACTIONS, 0.8), 0.15),
    "forager": (lambda rng: forager_agent(rng, 0.0), 0.15),
    "noisy_forager": (lambda rng: forager_agent(rng, rng.uniform(0.0, 0.4), rng.uniform(0.0, 8.0)), 0.55),
}


def returner_agent(rng: np.random.Generator, eps: float = 0.0, edge: float = 0.0) -> Policy:
    """Pong: keep the paddle on the ball's row. With probability `edge` per rally it aims to take
    the ball on an end of the paddle (angled returns), otherwise on its middle."""
    aim = 1

    def policy(s: pg.State) -> int:
        nonlocal aim
        if s.dx > 0 and rng.random() < 0.1:
            aim = int(rng.choice((0, 3))) if rng.random() < edge else int(rng.integers(1, 3))
        if rng.random() < eps:
            return int(rng.integers(pg.NUM_ACTIONS))
        nxt = s.by + s.dy if pg.TOP <= s.by + s.dy <= pg.BOTTOM else s.by - s.dy  # where the ball will be
        target = nxt - aim
        return pg.UP if s.player > target else pg.DOWN if s.player < target else pg.STAY

    return policy


PONG_AGENTS: dict[str, tuple[Callable[[np.random.Generator], Policy], float]] = {
    "random": (_random(pg.NUM_ACTIONS), 0.15),
    "sticky": (_sticky(pg.NUM_ACTIONS, 0.9), 0.15),
    "returner": (lambda rng: returner_agent(rng, 0.0, 0.5), 0.15),
    "noisy_returner": (lambda rng: returner_agent(rng, rng.uniform(0.0, 0.5), rng.uniform(0.0, 1.0)), 0.55),
}


def agents_for(game: str) -> dict[str, tuple[Callable[[np.random.Generator], Policy], float]]:
    return {"breakout": AGENTS, "doomlite": DOOM_AGENTS, "flappy": FLAPPY_AGENTS, "pacman": PACMAN_AGENTS, "pong": PONG_AGENTS}[game]
