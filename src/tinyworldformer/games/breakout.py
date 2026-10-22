"""Discrete Breakout on a small token grid.

The state is a (H, W) grid of tokens plus the remaining lives. The ball's
direction is part of its token, so a single frame fully determines the future
(apart from the random serve). The ball moves one cell per step, diagonally.
"""

from typing import NamedTuple

import numpy as np

H, W = 20, 16
BRICK_ROWS = range(2, 6)
PADDLE_ROW = H - 2
PADDLE_W = 3
LIVES = 3

# tokens
EMPTY, WALL, BRICK, PADDLE = 0, 1, 2, 3
BALL_UL, BALL_UR, BALL_DL, BALL_DR = 4, 5, 6, 7
NUM_TOKENS = 8

# token <-> (dr, dc)
_DIRS = {BALL_UL: (-1, -1), BALL_UR: (-1, 1), BALL_DL: (1, -1), BALL_DR: (1, 1)}
_BALL_TOKEN = {d: t for t, d in _DIRS.items()}

# actions
STAY, LEFT, RIGHT = 0, 1, 2
NUM_ACTIONS = 3


class State(NamedTuple):
    grid: np.ndarray  # (H, W) uint8 tokens
    lives: int


def _paddle_center(grid: np.ndarray) -> int:
    return int(np.flatnonzero(grid[PADDLE_ROW] == PADDLE).mean().round())


def _ball_pos(grid: np.ndarray) -> tuple[int, int]:
    rows, cols = np.nonzero(grid >= BALL_UL)
    return int(rows[0]), int(cols[0])


def _place_serve(grid: np.ndarray, rng: np.random.Generator) -> None:
    """Put the ball above the paddle centre, moving up in a random diagonal."""
    grid[grid >= BALL_UL] = EMPTY
    dc = int(rng.choice((-1, 1)))
    grid[PADDLE_ROW - 1, _paddle_center(grid)] = _BALL_TOKEN[(-1, dc)]


def reset(rng: np.random.Generator | None = None, holes: bool = False) -> State:
    """New game. With `holes`, a random share of the bricks is already gone and the paddle starts
    anywhere: same rules, but late-game states are reached right away (for data coverage)."""
    rng = rng or np.random.default_rng()
    grid = np.zeros((H, W), np.uint8)
    grid[0, :] = grid[:, 0] = grid[:, -1] = WALL
    bricks = grid[BRICK_ROWS.start : BRICK_ROWS.stop, 1 : W - 1]
    bricks[:] = BRICK
    if holes:
        bricks[rng.random(bricks.shape) > rng.uniform(0.1, 0.9)] = EMPTY
    mid = int(rng.integers(2, W - 2)) if holes else W // 2
    grid[PADDLE_ROW, mid - PADDLE_W // 2 : mid + PADDLE_W // 2 + 1] = PADDLE
    _place_serve(grid, rng)
    return State(grid, LIVES)


def step(
    state: State, action: int, rng: np.random.Generator | None = None
) -> tuple[State, float, bool]:
    """Advance one tick. `rng` is only used to serve the ball after a miss."""
    grid, lives = state.grid.copy(), state.lives
    reward = 0.0

    # paddle
    center = _paddle_center(grid)
    new_center = int(np.clip(center + (action == RIGHT) - (action == LEFT), 2, W - 3))
    if (grid[PADDLE_ROW, new_center - 1 : new_center + 2] >= BALL_UL).any():
        new_center = center  # the ball is beside the paddle: blocked
    grid[grid == PADDLE] = EMPTY
    grid[PADDLE_ROW, new_center - 1 : new_center + 2] = PADDLE

    # ball: reflect against whatever it would run into, then move one cell
    r, c = _ball_pos(grid)
    dr, dc = _DIRS[int(grid[r, c])]
    solid = lambda rr, cc: grid[rr, cc] in (WALL, BRICK, PADDLE)  # noqa: E731
    hit_v, hit_h = solid(r + dr, c), solid(r, c + dc)
    hit_d = solid(r + dr, c + dc) and not (hit_v or hit_h)

    for cell, hit in (((r + dr, c), hit_v), ((r, c + dc), hit_h), ((r + dr, c + dc), hit_d)):
        if hit and grid[cell] == BRICK:
            grid[cell] = EMPTY
            reward += 1.0
    paddle_hit = (hit_v and grid[r + dr, c] == PADDLE) or (
        hit_d and grid[r + dr, c + dc] == PADDLE
    )
    if hit_v or hit_d:
        dr = -dr
    if hit_h or hit_d:
        dc = -dc
    if paddle_hit:  # aim with the paddle: outer cells steer the ball outward
        dc = int(np.sign(c - new_center)) or dc

    grid[r, c] = EMPTY
    if not solid(r + dr, c + dc):
        r, c = r + dr, c + dc
    grid[r, c] = _BALL_TOKEN[(dr, dc)]

    # miss
    if r == H - 1:
        lives -= 1
        reward -= 1.0
        _place_serve(grid, rng or np.random.default_rng())

    done = lives <= 0 or not (grid == BRICK).any()
    return State(grid, lives), reward, bool(done)


_PALETTE = np.array(
    [
        (0, 0, 0),  # empty
        (90, 90, 90),  # wall
        (200, 80, 60),  # brick (overridden per row, see BRICK_COLORS)
        (60, 140, 220),  # paddle
        *[(255, 255, 255)] * 4,  # ball
    ],
    np.uint8,
)
# colour by row, so the model only has to predict "brick" and not its colour
BRICK_COLORS = np.array(
    [(220, 60, 60), (230, 140, 50), (220, 200, 60), (70, 190, 90)], np.uint8
)


def colorize(grid: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB, one pixel per cell. Works on any token grid, e.g. a generated one."""
    img = _PALETTE[grid]
    for row, color in zip(BRICK_ROWS, BRICK_COLORS):
        img[row][grid[row] == BRICK] = color
    return img


def render(state: State, cell: int = 8) -> np.ndarray:
    """(H*cell, W*cell, 3) uint8 RGB image."""
    return colorize(state.grid).repeat(cell, axis=0).repeat(cell, axis=1)


def render_ansi(grid: np.ndarray) -> str:
    """Truecolor terminal frame: two spaces per cell so cells look square."""
    rows = []
    for row in colorize(grid):
        rows.append("".join(f"\x1b[48;2;{r};{g};{b}m  " for r, g, b in row) + "\x1b[0m")
    return "\n".join(rows)


def render_ascii(state: State) -> str:
    chars = " #=_\\/,."  # ball glyph shows heading: \ UL, / UR, , DL, . DR
    return "\n".join("".join(chars[t] for t in row) for row in state.grid)
