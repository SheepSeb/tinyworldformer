"""Doom-lite: a tiny grid raycaster shooter on a 32x32 token frame.

The top 32x24 is a first-person view (raycast walls, enemy sprite, gun, muzzle flash), the bottom
8 rows are a HUD: health bar and an 8x8 minimap with the player (facing as part of the token) and
the enemy. The minimap makes one frame enough to know the whole state.

Rules, one step per action: the player turns 90 degrees or steps one cell; firing kills the enemy
if it is straight ahead with no wall between. The enemy then respawns at a random cell far away
(the only random event). Otherwise, every second step, the enemy steps toward the player along a
shortest path, or hits for one health when adjacent. A HUD light shows when it is the enemy's
turn, so the frame still tells everything.
"""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

MAP = [
    "########",
    "#..#...#",
    "#.##.#.#",
    "#....#.#",
    "#.#.##.#",
    "#.#....#",
    "#...#..#",
    "########",
]
WALLS = np.array([[c == "#" for c in row] for row in MAP])
MAP_N = len(MAP)
FREE = [(x, y) for y in range(MAP_N) for x in range(MAP_N) if not WALLS[y, x]]

H, W = 32, 32
VIEW_H = 24
HEALTH = 5
MAX_STEPS = 500

# tokens (colours come from `colorize`, which also shades floor and ceiling by screen row)
(CEIL, FLOOR, WALL_LIGHT, WALL_DARK, WALL_LIGHT_FAR, WALL_DARK_FAR, ENEMY, ENEMY_EYE, GUN, GUN_LIGHT,
 FLASH, HURT, HUD, HEALTH_BAR, MAP_WALL, MAP_FLOOR) = range(16)
MAP_ENEMY = 16
MAP_PLAYER = 17  # + facing: N, E, S, W
NUM_TOKENS = 21
FAR = 2.5  # walls further than this get the darker shade

# actions
NOOP, FORWARD, LEFT, RIGHT, FIRE = range(5)
NUM_ACTIONS = 5

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W as (dx, dy)
FOV = 0.66  # camera plane half-width: ~66 degree field of view, like Wolfenstein


@dataclass(frozen=True)
class State:
    px: int
    py: int
    facing: int
    ex: int
    ey: int
    health: int
    flash: bool = False
    steps: int = 0
    hurt: bool = False  # took a hit this step: the view gets a red border

    @property
    def lives(self) -> int:  # what the data pipeline records
        return self.health

    @property
    def grid(self) -> np.ndarray:
        return render(self)


# --- the 3D view ------------------------------------------------------------------------------


def _cast(px: float, py: float, dx: float, dy: float) -> tuple[float, int, tuple[int, int]]:
    """DDA through the grid: (perpendicular wall distance, side hit: 0 = x-side, 1 = y-side, wall cell)."""
    mx, my = int(px), int(py)
    ddx = abs(1 / dx) if dx else 1e30
    ddy = abs(1 / dy) if dy else 1e30
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    side_x = ((mx + 1 - px) if dx > 0 else (px - mx)) * ddx
    side_y = ((my + 1 - py) if dy > 0 else (py - my)) * ddy
    while True:
        if side_x < side_y:
            side_x += ddx
            mx += sx
            side = 0
        else:
            side_y += ddy
            my += sy
            side = 1
        if WALLS[my, mx]:
            return (side_x - ddx if side == 0 else side_y - ddy), side, (mx, my)


def _walls(px: int, py: int, facing: int) -> tuple[np.ndarray, np.ndarray]:
    """View with walls, floor and ceiling only, plus the per-column depth buffer."""
    view = np.zeros((VIEW_H, W), np.uint8)
    depth = np.zeros(W)
    dx, dy = DIRS[facing]
    plane = (-dy * FOV, dx * FOV)
    hits = [_cast(px + 0.5, py + 0.5, dx + plane[0] * c, dy + plane[1] * c)
            for c in (2 * (col + 0.5) / W - 1 for col in range(W))]
    for col, (dist, side, cell) in enumerate(hits):
        depth[col] = dist
        h = min(VIEW_H, int(VIEW_H / max(dist, 1e-6)))
        top = (VIEW_H - h) // 2
        view[:top, col] = CEIL
        near = (WALL_LIGHT, WALL_DARK)[side]
        tok = near if dist < FAR else near + 2  # +2: the far shades
        # a dark line where the wall moves on to another block or face: corners become readable
        if col > 0 and hits[col - 1][1:] != (side, cell):
            tok = WALL_DARK_FAR
        view[top : top + h, col] = tok
        view[top + h :, col] = FLOOR
    return view, depth


_VIEWS = {(x, y, f): _walls(x, y, f) for x, y in FREE for f in range(4)}


# the enemy: an imp-like figure, scaled to its distance ("E" eye, "#" body)
SPRITE = [
    "..####..",
    ".######.",
    ".#E##E#.",
    ".######.",
    "..####..",
    ".######.",
    "########",
    "#.####.#",
    "..#..#..",
    ".##..##.",
]
_SPRITE = np.array([[{"#": ENEMY, "E": ENEMY_EYE}.get(c, 0) for c in row] for row in SPRITE])

GUN_ROWS = [  # pistol at the bottom centre, "L" lit edges
    "....LL....",
    "...LGGL...",
    "..GGGGGG..",
    ".GGLGGLGG.",
]
FLASH_ROWS = [  # muzzle flash above the barrel
    ".F..FF..F.",
    "..F.FF.F..",
    "...FFFF...",
]


def _stamp(view: np.ndarray, rows: list[str], top: int, tokens: dict[str, int]) -> None:
    left = (W - len(rows[0])) // 2
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch in tokens:
                view[top + r, left + c] = tokens[ch]


def _sprite(view: np.ndarray, depth: np.ndarray, s: State) -> None:
    """Draw the enemy scaled by distance, hidden behind nearer walls."""
    dx, dy = DIRS[s.facing]
    plane = (-dy * FOV, dx * FOV)
    rel_x, rel_y = s.ex - s.px, s.ey - s.py
    inv = 1 / (plane[0] * dy - dx * plane[1])
    tx = inv * (dy * rel_x - dx * rel_y)  # sideways in camera space
    tz = inv * (-plane[1] * rel_x + plane[0] * rel_y)  # depth
    if tz <= 0.1:
        return
    center = int(W / 2 * (1 + tx / tz))
    h = min(VIEW_H, int(VIEW_H / tz * 0.9))
    w = max(2, int(h * 0.75))
    top = (VIEW_H - h) // 2 + h // 8
    sh, sw = _SPRITE.shape
    for col in range(center - w // 2, center - w // 2 + w):
        if not (0 <= col < W and tz < depth[col]):
            continue
        u = (col - (center - w // 2)) * sw // w
        for row in range(max(top, 0), min(top + h, VIEW_H)):
            tok = _SPRITE[(row - top) * sh // h, u]
            if tok:
                view[row, col] = tok


def render(s: State) -> np.ndarray:
    grid = np.full((H, W), HUD, np.uint8)
    view, depth = _VIEWS[(s.px, s.py, s.facing)]
    view = view.copy()
    _sprite(view, depth, s)
    _stamp(view, GUN_ROWS, VIEW_H - len(GUN_ROWS), {"G": GUN, "L": GUN_LIGHT})
    if s.flash:
        _stamp(view, FLASH_ROWS, VIEW_H - len(GUN_ROWS) - len(FLASH_ROWS), {"F": FLASH})
    if s.hurt:
        view[0, :] = view[-1, :] = view[:, 0] = view[:, -1] = HURT
    grid[:VIEW_H] = view
    if s.steps % 2 == 1:  # the enemy acts on the next step
        grid[29:31, 20:22] = ENEMY
    for i in range(HEALTH):  # health bar: 5 segments of 3 px, rows 27-28
        if i < s.health:
            grid[27:29, 2 + 4 * i : 5 + 4 * i] = HEALTH_BAR
    mini = np.where(WALLS, MAP_WALL, MAP_FLOOR).astype(np.uint8)
    mini[s.ey, s.ex] = MAP_ENEMY
    mini[s.py, s.px] = MAP_PLAYER + s.facing
    grid[VIEW_H:, W - MAP_N :] = mini
    return grid


# --- rules ------------------------------------------------------------------------------------


def _bfs_step(ex: int, ey: int, px: int, py: int) -> tuple[int, int]:
    """Next cell on a shortest path from the enemy to the player (ties: N, E, S, W order)."""
    dist = {(px, py): 0}
    q = deque([(px, py)])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS:
            n = (x + dx, y + dy)
            if not WALLS[n[1], n[0]] and n not in dist:
                dist[n] = dist[(x, y)] + 1
                q.append(n)
    best = min(((ex + dx, ey + dy) for dx, dy in DIRS if (ex + dx, ey + dy) in dist), key=lambda c: dist[c])
    return best


def _in_sight(s: State) -> bool:
    dx, dy = DIRS[s.facing]
    x, y = s.px + dx, s.py + dy
    while not WALLS[y, x]:
        if (x, y) == (s.ex, s.ey):
            return True
        x, y = x + dx, y + dy
    return False


def _spawn(rng: np.random.Generator, px: int, py: int) -> tuple[int, int]:
    far = [c for c in FREE if abs(c[0] - px) + abs(c[1] - py) >= 5]
    return far[rng.integers(len(far))]


def reset(rng: np.random.Generator | None = None, random_start: bool = False) -> State:
    """New game. With `random_start` the player starts anywhere, facing anywhere (data coverage)."""
    rng = rng or np.random.default_rng()
    px, py, facing = (FREE[rng.integers(len(FREE))] + (int(rng.integers(4)),)) if random_start else (1, 1, 2)
    ex, ey = _spawn(rng, px, py)
    return State(px, py, facing, ex, ey, HEALTH)


def step(s: State, action: int, rng: np.random.Generator | None = None) -> tuple[State, float, bool]:
    """Advance one tick. `rng` is only used to respawn the enemy after a kill."""
    reward = 0.0
    px, py, facing, flash = s.px, s.py, s.facing, False
    ex, ey, health = s.ex, s.ey, s.health
    if action == FORWARD:
        nx, ny = px + DIRS[facing][0], py + DIRS[facing][1]
        if not WALLS[ny, nx] and (nx, ny) != (ex, ey):
            px, py = nx, ny
    elif action == LEFT:
        facing = (facing + 3) % 4
    elif action == RIGHT:
        facing = (facing + 1) % 4
    elif action == FIRE:
        flash = True
        if _in_sight(replace(s, px=px, py=py, facing=facing)):
            reward += 1.0
            ex, ey = _spawn(rng or np.random.default_rng(), px, py)
            return State(px, py, facing, ex, ey, health, flash, s.steps + 1), reward, s.steps + 1 >= MAX_STEPS
    hurt = False
    if s.steps % 2 == 0:  # not the enemy's turn
        pass
    elif abs(ex - px) + abs(ey - py) == 1:
        health -= 1
        reward -= 1.0
        hurt = True
    else:
        ex, ey = _bfs_step(ex, ey, px, py)
        if (ex, ey) == (px, py):  # never walk into the player
            ex, ey = s.ex, s.ey
    n = State(px, py, facing, ex, ey, health, flash, s.steps + 1, hurt)
    return n, reward, health <= 0 or n.steps >= MAX_STEPS


def read(grid: np.ndarray) -> State | None:
    """Frame -> State, from the minimap, the health bar and the turn light (the inverse of `render`
    for everything the rules use), or None when the minimap does not hold exactly one player and
    one enemy, or puts one inside a wall."""
    mini = grid[VIEW_H:, W - MAP_N :]
    p, e = np.argwhere(mini >= MAP_PLAYER), np.argwhere(mini == MAP_ENEMY)
    if len(p) != 1 or len(e) != 1 or WALLS[p[0][0], p[0][1]] or WALLS[e[0][0], e[0][1]]:
        return None
    (py, px), (ey, ex) = p[0], e[0]
    health = int((grid[27] == HEALTH_BAR).sum()) // 3
    steps = int((grid[29:31, 20:22] == ENEMY).all())
    return State(int(px), int(py), int(mini[py, px] - MAP_PLAYER), int(ex), int(ey), health, steps=steps)


PALETTE = np.array(
    [
        (70, 70, 85),  # ceiling (shaded by row in colorize)
        (95, 75, 55),  # floor (shaded by row)
        (190, 175, 150),  # wall, lit side, near
        (135, 122, 105),  # wall, shaded side, near
        (120, 110, 95),  # wall, lit side, far
        (85, 78, 68),  # wall, shaded side, far
        (190, 45, 35),  # enemy
        (255, 230, 60),  # enemy eyes
        (55, 55, 60),  # gun
        (150, 150, 160),  # gun highlight
        (255, 210, 70),  # muzzle flash
        (230, 20, 20),  # hurt border
        (20, 20, 24),  # hud
        (60, 210, 90),  # health
        (140, 140, 140),  # minimap wall
        (40, 40, 48),  # minimap floor
        (240, 50, 50),  # minimap enemy
        *[(90, 170, 255)] * 4,  # minimap player
    ],
    np.uint8,
)


def colorize(grid: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB. Ceiling and floor fade toward the horizon, which reads as depth."""
    img = PALETTE[grid].astype(np.float32)
    rows = np.arange(grid.shape[0])[:, None]
    fade = np.clip(np.abs(rows - (VIEW_H - 1) / 2) / (VIEW_H / 2), 0.0, 1.0)  # 0 at the horizon, 1 at the edges
    shade = (0.45 + 0.75 * fade)[..., None]
    sky_or_floor = ((grid == CEIL) | (grid == FLOOR))[..., None]
    return np.where(sky_or_floor, np.clip(img * shade, 0, 255), img).astype(np.uint8)


def render_ascii(grid: np.ndarray) -> str:
    chars = " .#=:-Eo%^*!  +X x^>v<"
    return "\n".join("".join(chars[t] for t in row) for row in grid)
