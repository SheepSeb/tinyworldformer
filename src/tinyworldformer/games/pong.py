"""Pong on a 20x32 token frame: your paddle on the left, a CPU paddle on the right.

The ball is one cell; its token encodes its heading (left or right, and up, level or down), so one
frame is enough to know the next. It moves one cell per step and bounces off the top and bottom
walls. A paddle hit sends it back; hitting the paddle's top cell angles it up, the bottom cell
down, the middle two keep its slope.

The CPU paddle only reacts once the ball is coming its way and past column 20, then moves toward
the ball's row on two of every three ball columns, so angled shots can beat it. That rule reads
everything off the frame (the ball's column and heading), so the frame stays the whole state.

A ball past the CPU is +1, a ball past you is -1. Either way the next serve comes from the centre
line at a random row and slope, toward whoever lost the point: that serve is the game's only
random event. The HUD shows both scores as 5 slots each that fill up; the first to 5 wins.
"""

from dataclasses import dataclass

import numpy as np

H, W = 20, 32
HUD_ROW = 0
TOP, BOTTOM = 2, H - 2  # rows the ball can be in (row 1 and the last row are walls)
PADDLE_LEN = 4
PLAYER_X, CPU_X = 1, W - 2
NET_X = W // 2
REACT_X = W - 12  # the CPU starts tracking once the ball reaches this column
POINTS = 5
SLOTS_P = [2 + 2 * i for i in range(POINTS)]  # HUD columns of your score
SLOTS_C = [W - 3 - 2 * i for i in range(POINTS)]  # and the CPU's

# tokens
EMPTY, WALL, NET, PADDLE, CPU, HUD, SLOT_P, POINT_P, SLOT_C, POINT_C = range(10)
BALL = 10  # + (dx > 0) * 3 + (dy + 1)
NUM_TOKENS = 16

# actions
STAY, UP, DOWN = range(3)
NUM_ACTIONS = 3


@dataclass(frozen=True)
class State:
    bx: int
    by: int
    dx: int
    dy: int
    player: int  # top row of each paddle
    cpu: int
    score: int = 0
    lost: int = 0
    random_event: bool = False  # this step served a new ball

    @property
    def lives(self) -> int:
        return POINTS - self.lost

    @property
    def grid(self) -> np.ndarray:
        return render(self)


def render(s: State) -> np.ndarray:
    grid = np.full((H, W), EMPTY, np.uint8)
    grid[HUD_ROW] = HUD
    grid[TOP - 1] = grid[BOTTOM + 1] = WALL
    grid[TOP : BOTTOM + 1 : 2, NET_X] = NET
    for i, c in enumerate(SLOTS_P):
        grid[HUD_ROW, c] = POINT_P if i < s.score else SLOT_P
    for i, c in enumerate(SLOTS_C):
        grid[HUD_ROW, c] = POINT_C if i < s.lost else SLOT_C
    grid[s.player : s.player + PADDLE_LEN, PLAYER_X] = PADDLE
    grid[s.cpu : s.cpu + PADDLE_LEN, CPU_X] = CPU
    grid[s.by, s.bx] = BALL + (s.dx > 0) * 3 + s.dy + 1
    return grid


def _serve(rng: np.random.Generator, toward: int) -> tuple[int, int, int, int]:
    return NET_X, int(rng.integers(TOP + 2, BOTTOM - 1)), toward, int(rng.integers(-1, 2))


def reset(rng: np.random.Generator | None = None, varied: bool = False) -> State:
    """New game. With `varied`, the paddles, the ball and the score start anywhere (data coverage)."""
    rng = rng or np.random.default_rng()
    mid = (TOP + BOTTOM + 1 - PADDLE_LEN) // 2
    if not varied:
        return State(*_serve(rng, -1), mid, mid)
    paddles = rng.integers(TOP, BOTTOM + 2 - PADDLE_LEN, 2)
    bx, by = int(rng.integers(PLAYER_X + 1, CPU_X)), int(rng.integers(TOP, BOTTOM + 1))
    dx, dy = int(rng.choice((-1, 1))), int(rng.integers(-1, 2))
    return State(bx, by, dx, dy, int(paddles[0]), int(paddles[1]), int(rng.integers(POINTS)), int(rng.integers(POINTS)))


def _move(top: int, d: int) -> int:
    return min(max(top + d, TOP), BOTTOM + 1 - PADDLE_LEN)


def _cpu_move(s: State) -> int:
    if s.dx < 0 or s.bx < REACT_X or s.bx % 3 == 0:  # two moves for every three ball columns
        return 0
    return -1 if s.by < s.cpu + 1 else 1 if s.by > s.cpu + 2 else 0


def step(s: State, action: int, rng: np.random.Generator | None = None) -> tuple[State, float, bool]:
    """Advance one tick. `rng` is only used for the serve after a point."""
    player = _move(s.player, {STAY: 0, UP: -1, DOWN: 1}[action])
    cpu = _move(s.cpu, _cpu_move(s))
    dx, dy = s.dx, s.dy
    by = s.by + dy
    if not TOP <= by <= BOTTOM:  # bounce off the wall
        dy = -dy
        by = s.by + dy
    bx = s.bx + dx
    reward = 0.0
    paddle_x, top = (PLAYER_X, player) if dx < 0 else (CPU_X, cpu)
    if bx == paddle_x:
        if top <= by < top + PADDLE_LEN:  # hit: back the other way, angled by where it hit
            part = by - top
            dy = -1 if part == 0 else 1 if part == PADDLE_LEN - 1 else dy
            dx = -dx
            bx = s.bx
        else:
            reward = -1.0 if dx < 0 else 1.0
    score, lost = s.score + (reward > 0), s.lost + (reward < 0)
    if reward:
        bx, by, dx, dy = _serve(rng or np.random.default_rng(), -1 if reward < 0 else 1)
    n = State(bx, by, dx, dy, player, cpu, score, lost, bool(reward))
    return n, reward, score >= POINTS or lost >= POINTS


def read(grid: np.ndarray):
    """Pong frame -> State (the inverse of `render`), or None if unreadable."""
    ball = np.argwhere(grid >= BALL)
    paddle = np.flatnonzero(grid[:, PLAYER_X] == PADDLE)
    cpu = np.flatnonzero(grid[:, CPU_X] == CPU)
    if len(ball) != 1 or len(paddle) != PADDLE_LEN or len(cpu) != PADDLE_LEN:
        return None
    r, c = ball[0]
    k = int(grid[r, c]) - BALL
    hud = grid[HUD_ROW]
    return State(int(c), int(r), 1 if k >= 3 else -1, k % 3 - 1, int(paddle[0]), int(cpu[0]),
                 int((hud == POINT_P).sum()), int((hud == POINT_C).sum()))


PALETTE = np.array(
    [
        (10, 12, 18),  # court (shaded by row in colorize)
        (200, 200, 210),  # wall
        (70, 75, 90),  # net
        (80, 170, 255),  # your paddle
        (255, 90, 80),  # cpu paddle
        (22, 22, 28),  # hud
        (40, 60, 90),  # your empty score slot
        (80, 170, 255),  # your point
        (90, 40, 40),  # cpu empty score slot
        (255, 90, 80),  # cpu point
        *[(255, 255, 255)] * 6,  # ball (heading is in the token, not the colour)
    ],
    np.uint8,
)


def colorize(grid: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB; the court gets a faint glow toward the middle row."""
    img = PALETTE[grid].astype(np.float32)
    rows = np.arange(grid.shape[0])[:, None, None]
    shade = 1.6 - 0.6 * np.abs(rows - (TOP + BOTTOM) / 2) / ((BOTTOM - TOP) / 2)
    return np.where((grid == EMPTY)[..., None], np.clip(img * shade, 0, 255), img).astype(np.uint8)
