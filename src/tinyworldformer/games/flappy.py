"""Flappy Bird on a 24x24 token frame.

The world scrolls left one column per step: pipes, and the stripes on the ground. The bird stays
in column 6; its token encodes its vertical speed (-2..2), so one frame is enough to know the
next. Flapping sets the speed to -2, otherwise gravity adds 1 per step (up to 2).

New pipes come every 9 columns with a random gap height. So that the dream can sample that, the
next gap is shown ahead of time by a single marker cell in the rightmost column (a HUD strip): a
pipe entering takes its gap from the marker, and the marker jumps to a new random height. That
jump is the game's only random event, and decodes like Breakout's ball: one global choice.

+1 for every pipe passed; hitting a pipe or the ground is -1 and ends the game.
"""

from dataclasses import dataclass

import numpy as np

H, W = 24, 24
PLAY_W = W - 1  # the last column is the HUD strip with the next-gap marker
GROUND = 22  # rows 22-23: grass, then ground; the bird dies on reaching row 22
BUSH_ROW = 21
BIRD_X = 6
PIPE_W = 2
GAP = 7
SPACING = 9  # columns from one pipe to the next
GAP_TOPS = range(2, GROUND - GAP - 1)  # possible top rows of a gap
MAX_STEPS = 3000

# tokens
SKY, BUSH, GRASS, GROUND_LIGHT, GROUND_DARK, PIPE, PIPE_EDGE, PIPE_CAP, HUD, BEAK = range(10)
MARKER = 10  # the next gap's centre, in the HUD strip
BIRD = 11  # + (speed + 2): speeds -2..2
NUM_TOKENS = 16

# actions
NOOP, FLAP = 0, 1
NUM_ACTIONS = 2


@dataclass(frozen=True)
class State:
    y: int
    v: int
    pipes: tuple[tuple[int, int], ...]  # (left column, gap top), oldest first
    next_gap: int
    phase: int  # ground stripe scroll offset
    alive: bool = True
    random_event: bool = False  # this step drew a new gap height

    @property
    def lives(self) -> int:
        return int(self.alive)

    @property
    def grid(self) -> np.ndarray:
        return render(self)


def render(s: State) -> np.ndarray:
    grid = np.full((H, W), SKY, np.uint8)
    grid[BUSH_ROW, :PLAY_W] = [BUSH if (c * 7) % 5 < 3 else SKY for c in range(PLAY_W)]  # static bush line
    grid[GROUND, :PLAY_W] = GRASS
    grid[GROUND + 1, :PLAY_W] = [GROUND_DARK if (c + s.phase) % 4 < 2 else GROUND_LIGHT for c in range(PLAY_W)]
    for x, top in s.pipes:
        for dx, tok in ((0, PIPE), (1, PIPE_EDGE)):
            c = x + dx
            if 0 <= c < PLAY_W:
                grid[:top, c] = tok
                grid[top + GAP : GROUND, c] = tok
                if top > 0:
                    grid[top - 1, c] = PIPE_CAP
                grid[top + GAP, c] = PIPE_CAP
    grid[:, PLAY_W] = HUD
    grid[s.next_gap + GAP // 2, PLAY_W] = MARKER
    grid[s.y, BIRD_X + 1] = BEAK
    grid[s.y, BIRD_X] = BIRD + s.v + 2
    return grid


def _gap(rng: np.random.Generator) -> int:
    return int(rng.integers(GAP_TOPS.start, GAP_TOPS.stop))


def reset(rng: np.random.Generator | None = None, varied: bool = False) -> State:
    """New game. With `varied` the bird starts at a random height and speed (data coverage)."""
    rng = rng or np.random.default_rng()
    y, v = (int(rng.integers(3, GROUND - 3)), int(rng.integers(-2, 3))) if varied else (10, 0)
    return State(y, v, ((PLAY_W - PIPE_W - 4, _gap(rng)),), _gap(rng), 0)


def _hits(y0: int, y1: int, pipes: tuple[tuple[int, int], ...]) -> bool:
    """Does the bird's path from row y0 to y1 in its column cross a pipe or the ground?"""
    if y1 >= GROUND:
        return True
    for x, top in pipes:
        if x <= BIRD_X < x + PIPE_W:
            lo, hi = min(y0, y1), max(y0, y1)
            if any(r < top or r >= top + GAP for r in range(lo, hi + 1)):
                return True
    return False


def step(s: State, action: int, rng: np.random.Generator | None = None) -> tuple[State, float, bool]:
    """Advance one tick. `rng` is only used to draw the next gap height when a pipe enters."""
    pipes = tuple((x - 1, top) for x, top in s.pipes if x - 1 + PIPE_W > 0)
    next_gap, spawned = s.next_gap, False
    if not pipes or pipes[-1][0] <= PLAY_W - PIPE_W - SPACING:
        pipes += ((PLAY_W - PIPE_W, next_gap),)
        next_gap, spawned = _gap(rng or np.random.default_rng()), True
    v = -2 if action == FLAP else min(s.v + 1, 2)
    y = max(s.y + v, 0)  # the top of the screen stops the bird, it does not kill it
    reward = 0.0
    if _hits(s.y, y, pipes):
        n = State(min(y, GROUND - 1), v, pipes, next_gap, (s.phase + 1) % 4, False, spawned)
        return n, -1.0, True
    if any(x + PIPE_W == BIRD_X for x, _ in pipes):  # a pipe's right edge just passed the bird
        reward = 1.0
    n = State(y, v, pipes, next_gap, (s.phase + 1) % 4, True, spawned)
    return n, reward, False


def read(grid: np.ndarray):
    """Flappy Bird frame -> State (the inverse of `render`), or None if unreadable."""
    pipe = (PIPE, PIPE_EDGE, PIPE_CAP)
    bird = np.argwhere(grid[:, :PLAY_W] >= BIRD)
    marker = np.flatnonzero(grid[:, PLAY_W] == MARKER)
    if len(bird) != 1 or len(marker) != 1:
        return None
    y, _ = bird[0]
    cols = grid[:GROUND, :PLAY_W]
    body = (cols == PIPE).any(0)  # a pipe's left column
    edge = (cols == PIPE_EDGE).any(0)  # its shaded right column
    pipes = []
    if edge[0] and not body[0]:  # only the edge is still on screen
        pipes.append(-1)
    pipes += [c for c in range(PLAY_W) if body[c]]

    def top(c: int) -> int:
        """Gap top from the caps (rows top-1 and top+GAP), read over both of the pipe's columns,
        so the bird or its beak covering one cell does not matter."""
        caps = np.zeros(GROUND, bool)
        for cc in (c, c + 1):
            if 0 <= cc < PLAY_W:
                caps |= grid[:GROUND, cc] == PIPE_CAP
        rows = np.flatnonzero(caps)
        lower = [r for r in rows if r >= GAP]
        return int(lower[-1]) - GAP if lower else int(rows[0]) + 1 if len(rows) else 0

    pipes = [(c, top(c)) for c in pipes]
    phase = next((p for p in range(4) if all(
        grid[GROUND + 1, c] == (GROUND_DARK if (c + p) % 4 < 2 else GROUND_LIGHT) for c in range(PLAY_W))), -1)
    return State(int(y), int(grid[y, BIRD_X]) - BIRD - 2, tuple(pipes), int(marker[0]) - GAP // 2, phase)


PALETTE = np.array(
    [
        (110, 195, 235),  # sky (shaded by row in colorize)
        (85, 180, 70),  # bush
        (120, 210, 80),  # grass
        (220, 205, 145),  # ground, light stripe
        (200, 180, 120),  # ground, dark stripe
        (95, 190, 60),  # pipe
        (60, 135, 40),  # pipe shaded edge
        (140, 225, 90),  # pipe cap
        (30, 30, 36),  # hud strip
        (250, 120, 40),  # beak
        (255, 255, 255),  # next-gap marker
        *[(255, 215, 50)] * 5,  # bird (speed is in the token, not the colour)
    ],
    np.uint8,
)


def colorize(grid: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB; the sky gets lighter toward the horizon."""
    img = PALETTE[grid].astype(np.float32)
    rows = np.arange(grid.shape[0])[:, None, None]
    shade = 0.8 + 0.3 * rows / H
    return np.where((grid == SKY)[..., None], np.clip(img * shade, 0, 255), img).astype(np.uint8)
