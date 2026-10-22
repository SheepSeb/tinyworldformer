"""Pac-Man lite: a fixed 23x21 maze with 250 cells, a pellet in almost every one, and two ghosts.

The frame is 48x48 tokens and draws every maze cell as 2x2 tokens: walls shaded where they face a
path, a pellet as a dot, Pac-Man as a yellow block with a dark mouth on the side it is heading,
ghosts with white eyes over a coloured body. Below the maze is a HUD with the lives (small Pac-Men)
and a light in the colour of the ghost that moves next. One cell of each sprite carries the state:
Pac-Man's mouth token encodes its heading, and each ghost's lower-left token says whether a pellet
lies under the ghost (so a ghost standing on a pellet does not hide it). The other sprite cells
follow from that cell (`STAMPS`), so the dream draws them instead of predicting them. One frame
is the whole state.

Rules, one step per action. Actions: keep going, or N / E / S / W. Pac-Man moves one cell: a
direction turns and steps if that way is open, and otherwise Pac-Man keeps going the way it was
heading (a buffered turn); "keep going" only steps ahead. Stepping onto a pellet eats it (+1).
The ghosts are half as fast and take turns: ghost A moves on even steps, ghost B on odd steps. A
moving ghost steps along a shortest path to Pac-Man (A breaks ties N, E, S, W; B the other way
round), and stays put rather than step onto the other ghost. A ghost that reaches Pac-Man, or
Pac-Man stepping into a ghost, costs a life (-1): Pac-Man goes back to the start and the ghosts
respawn at random cells far away, A on the left half of the maze and B on the right. That respawn
is the game's only random event. The episode ends with the last life or the last visible pellet.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

MAP = [
    "#######################",
    "#.....................#",
    "#.###.#.#######.#.###.#",
    "#...#.............#...#",
    "#.#.#.#####.#####.#.#.#",
    "#.#.#.....#.#.....#.#.#",
    "#.#.###.#.#.#.#.###.#.#",
    "#.#.................#.#",
    "#.###.#.###.###.#.###.#",
    "#.....#.........#.....#",
    "#.#####.#.###.#.#####.#",
    "#.......#.....#.......#",
    "#.###.#.#.#.#.#.#.###.#",
    "#.#.......#.#.......#.#",
    "#.#.#.###.#.#.###.#.#.#",
    "#...#.#.........#.#...#",
    "#####.#.#######.#.#####",
    "#.....................#",
    "#.#####.###.###.#####.#",
    "#.....................#",
    "#######################",
]
WALLS = np.array([[c == "#" for c in row] for row in MAP])
MAP_H, MAP_W = WALLS.shape
FREE = [(x, y) for y in range(MAP_H) for x in range(MAP_W) if not WALLS[y, x]]
INDEX = {c: i for i, c in enumerate(FREE)}
START = (11, 19)

H, W = 48, 48
S = 2  # tokens per maze cell, each way
LIVES = 3
MAX_STEPS = 1500

# tokens
EMPTY, WALL, WALL_FILL, PELLET, HUD, PAC_BODY, EYE, BODY_A, BODY_B, LIFE, LIGHT_A, LIGHT_B = range(12)
GHOST_A = 12  # the ghost's lower-left cell; + 1 when a pellet lies underneath
GHOST_B = 14
PAC = 16  # Pac-Man's mouth; + heading: N, E, S, W
NUM_TOKENS = 20
MOUTH = [(1, 0), (1, 1), (0, 1), (0, 0)]  # (dx, dy) of the mouth in the 2x2 sprite, per heading


def _stamps() -> tuple[tuple[int, int, int, int], ...]:
    """(singleton token, row offset, col offset, token): the sprite cells that follow from where a
    singleton is. The dream draws them around the placed singleton instead of predicting them."""
    out = []
    for tok, body in ((GHOST_A, BODY_A), (GHOST_B, BODY_B)):
        for t in (tok, tok + 1):
            out += [(t, -1, 0, EYE), (t, -1, 1, EYE), (t, 0, 1, body)]
    for h, (dx, dy) in enumerate(MOUTH):
        out += [(PAC + h, y - dy, x - dx, PAC_BODY) for y in range(S) for x in range(S) if (x, y) != (dx, dy)]
    return tuple(out)


STAMPS = _stamps()
HUD_ROW = 2 * MAP_H + 1  # top row of the 2x2 HUD icons
LIFE_COLS = [1 + 3 * i for i in range(LIVES)]
LIGHT_COL = 2 * MAP_W - 3

# actions
KEEP, UP, RIGHT, DOWN, LEFT = range(5)
NUM_ACTIONS = 5
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W as (dx, dy)
ORDER_A = (0, 1, 2, 3)
ORDER_B = (3, 2, 1, 0)


def _distances() -> np.ndarray:
    """Shortest-path length between every pair of free cells (BFS from each)."""
    n = len(FREE)
    dist = np.full((n, n), 255, np.uint8)
    for s in range(n):
        dist[s, s] = 0
        q = deque([FREE[s]])
        while q:
            x, y = q.popleft()
            for dx, dy in DIRS:
                c = (x + dx, y + dy)
                if c in INDEX and dist[s, INDEX[c]] == 255:
                    dist[s, INDEX[c]] = dist[s, INDEX[(x, y)]] + 1
                    q.append(c)
    return dist


DIST = _distances()


def dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return int(DIST[INDEX[a], INDEX[b]])


@dataclass(frozen=True)
class State:
    pellets: np.ndarray  # (MAP_H, MAP_W) bool
    pac: tuple[int, int]
    heading: int
    a: tuple[int, int]
    b: tuple[int, int]
    lives: int = LIVES
    steps: int = 0
    random_event: bool = False  # this step respawned the ghosts

    @property
    def grid(self) -> np.ndarray:
        return render(self)

    def left(self) -> int:
        """Pellets that show in the frame (one under a ghost does not count as visible)."""
        n = int(self.pellets.sum())
        return n - sum(bool(self.pellets[g[1], g[0]]) for g in (self.a, self.b))


def _background() -> np.ndarray:
    """The maze without pellets or actors: wall tokens next to a path are lit, the rest are fill."""
    walls = np.kron(WALLS, np.ones((S, S), bool))
    padded = np.pad(walls, 1, constant_values=True)
    hh, ww = walls.shape
    near_path = np.zeros_like(walls)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near_path |= ~padded[1 + dy : 1 + dy + hh, 1 + dx : 1 + dx + ww]
    grid = np.full((H, W), HUD, np.uint8)
    grid[:hh, :ww] = np.where(walls, np.where(near_path, WALL, WALL_FILL), EMPTY)
    return grid


BACKGROUND = _background()


def _pac(grid: np.ndarray, row: int, col: int, heading: int, mouth: int, body: int = PAC_BODY) -> None:
    grid[row : row + S, col : col + S] = body
    dx, dy = MOUTH[heading]
    grid[row + dy, col + dx] = mouth


def render(s: State) -> np.ndarray:
    grid = BACKGROUND.copy()
    ys, xs = np.nonzero(s.pellets)
    grid[S * ys, S * xs] = PELLET  # the dot sits in the top-left of its cell
    for (x, y), anchor, body in ((s.a, GHOST_A, BODY_A), (s.b, GHOST_B, BODY_B)):
        grid[S * y, S * x : S * x + S] = EYE
        grid[S * y + 1, S * x : S * x + S] = body
        grid[S * y + 1, S * x] = anchor + int(s.pellets[y, x])
    _pac(grid, S * s.pac[1], S * s.pac[0], s.heading, PAC + s.heading)
    for i in range(s.lives):
        _pac(grid, HUD_ROW, LIFE_COLS[i], 1, HUD, LIFE)
    grid[HUD_ROW : HUD_ROW + S, LIGHT_COL : LIGHT_COL + S] = LIGHT_A if s.steps % 2 == 0 else LIGHT_B
    return grid


def locate(grid: np.ndarray):
    """Read a frame back into maze coordinates: (Pac-Man's cell and heading or None, ghost A's
    cell or None, ghost B's cell or None, pellets (MAP_H, MAP_W) bool). Sprites that are missing,
    doubled or off the cell lattice read as None."""
    top = grid[: S * MAP_H, : S * MAP_W]
    pac = np.argwhere(top >= PAC)
    found = None
    if len(pac) == 1:
        r, c = pac[0]
        heading = int(top[r, c]) - PAC
        dx, dy = MOUTH[heading]
        if (r - dy) % S == 0 and (c - dx) % S == 0:
            found = ((int(c) - dx) // S, (int(r) - dy) // S), heading
    pellets = top[::S, ::S] == PELLET
    ghosts = []
    for tok in (GHOST_A, GHOST_B):
        g = np.argwhere((top == tok) | (top == tok + 1))
        if len(g) == 1 and g[0][0] % S == 1 and g[0][1] % S == 0:
            x, y = int(g[0][1]) // S, int(g[0][0]) // S
            pellets[y, x] = top[g[0][0], g[0][1]] == tok + 1
            ghosts.append((x, y))
        else:
            ghosts.append(None)
    return found, ghosts[0], ghosts[1], pellets


def read(grid: np.ndarray) -> State | None:
    """Frame -> State (the inverse of `render`), or None if no real state draws this frame."""
    pac, a, b, pellets = locate(grid)
    if pac is None or a is None or b is None or any(WALLS[y, x] for x, y in (pac[0], a, b)):
        return None
    lives = int((grid[HUD_ROW, LIFE_COLS] == LIFE).sum())
    steps = int(grid[HUD_ROW, LIGHT_COL] == LIGHT_B)
    return State(pellets & ~WALLS, pac[0], pac[1], a, b, lives, steps)


def _spawn(rng: np.random.Generator, left: bool) -> tuple[int, int]:
    """A cell far from the start, on the left or the right half of the maze."""
    far = [c for c in FREE if (c[0] < MAP_W // 2) == left and dist(START, c) >= 10]
    return far[rng.integers(len(far))]


def reset(rng: np.random.Generator | None = None, varied: bool = False) -> State:
    """New game. With `varied`, pellets are missing at random and Pac-Man starts anywhere,
    heading anywhere, ghosts too (data coverage)."""
    rng = rng or np.random.default_rng()
    pellets = ~WALLS
    if varied:
        pellets = pellets & (rng.random(WALLS.shape) < rng.uniform(0.05, 1.0))
        cells = [FREE[i] for i in rng.choice(len(FREE), 3, replace=False)]
        pac, a, b = cells
        heading = int(rng.integers(4))
    else:
        pac, heading = START, 0
        a, b = _spawn(rng, True), _spawn(rng, False)
    pellets = pellets.copy()
    pellets[pac[1], pac[0]] = False
    if not pellets.any():
        pellets[a[1], a[0]] = True
        pellets[b[1], b[0]] = True
        pellets[pac[1], pac[0]] = False
    return State(pellets, pac, heading, a, b)


def chase(ghost: tuple[int, int], target: tuple[int, int], order: tuple[int, ...]) -> tuple[int, int]:
    """The neighbour of `ghost` that is closest to `target` (ties: first in `order`)."""
    best, best_d = ghost, 255
    for d in order:
        c = (ghost[0] + DIRS[d][0], ghost[1] + DIRS[d][1])
        if c in INDEX and dist(c, target) < best_d:
            best, best_d = c, dist(c, target)
    return best


def step(s: State, action: int, rng: np.random.Generator | None = None) -> tuple[State, float, bool]:
    """Advance one tick. `rng` is only used to respawn the ghosts after a catch."""
    heading = s.heading
    px, py = s.pac
    pellets, reward, caught = s.pellets, 0.0, False
    turn = action - 1 if action != KEEP else None
    for d in ([turn, heading] if turn is not None else [heading]):
        nx, ny = px + DIRS[d][0], py + DIRS[d][1]
        if not WALLS[ny, nx]:
            heading, px, py = d, nx, ny
            break
    if (px, py) in (s.a, s.b):
        caught = True
    elif pellets[py, px]:
        pellets = pellets.copy()
        pellets[py, px] = False
        reward = 1.0
    a, b = s.a, s.b
    if not caught:
        if s.steps % 2 == 0:
            a = chase(a, (px, py), ORDER_A)
            a = a if a != b else s.a
            caught = a == (px, py)
        else:
            b = chase(b, (px, py), ORDER_B)
            b = b if b != a else s.b
            caught = b == (px, py)
    lives = s.lives
    if caught:
        rng = rng or np.random.default_rng()
        lives -= 1
        reward = -1.0
        px, py, heading = START[0], START[1], 0
        a, b = _spawn(rng, True), _spawn(rng, False)
    n = State(pellets, (px, py), heading, a, b, lives, s.steps + 1, caught)
    return n, reward, lives <= 0 or n.left() == 0 or n.steps >= MAX_STEPS


PALETTE = np.array(
    [
        (6, 6, 16),  # empty path
        (50, 80, 235),  # wall facing a path (shaded by row in colorize)
        (16, 22, 90),  # wall fill
        (255, 200, 160),  # pellet
        (18, 18, 22),  # hud
        (255, 222, 30),  # pac-man body
        (245, 245, 255),  # ghost eyes
        (240, 30, 30),  # ghost A body (Blinky red)
        (255, 140, 210),  # ghost B body (Pinky pink)
        (255, 222, 30),  # a life in the HUD
        (240, 30, 30),  # HUD light: ghost A moves next
        (255, 140, 210),  # HUD light: ghost B moves next
        (240, 30, 30),  # ghost A, lower left
        (240, 30, 30),  # ghost A, lower left, on a pellet
        (255, 140, 210),  # ghost B, lower left
        (255, 140, 210),  # ghost B, lower left, on a pellet
        *[(6, 6, 16)] * 4,  # pac-man's mouth (heading is in the token, not the colour)
    ],
    np.uint8,
)


def colorize(grid: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB; the lit walls fade a little from top to bottom, like a CRT glow."""
    img = PALETTE[grid].astype(np.float32)
    rows = np.arange(grid.shape[0])[:, None, None]
    shade = 1.1 - 0.35 * rows / (S * MAP_H)
    return np.where((grid == WALL)[..., None], np.clip(img * shade, 0, 255), img).astype(np.uint8)
