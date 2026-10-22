"""Play the real games in the terminal.

    python -m tinyworldformer.games.play                    # Breakout: arrows or a/d move, q quits
    python -m tinyworldformer.games.play --game doomlite    # Doom-lite, one step per key press:
                                                      # w/up forward, a/d or left/right turn,
                                                      # space fire, s wait, q quit
    python -m tinyworldformer.games.play --game flappy      # Flappy Bird: space, w or up flaps, q quits
    python -m tinyworldformer.games.play --game pacman      # Pac-Man lite: wasd or arrows turn, q quits
    python -m tinyworldformer.games.play --game pong        # Pong: w/s or up/down move (held), q quits
"""

import argparse
import curses
import time

import numpy as np

from tinyworldformer.games import breakout as b, doomlite as dl, flappy as fb, pacman as pm, pong as pg

TICK = 0.08  # seconds per step


def _xterm256(r: int, g: int, b: int) -> int:
    """Nearest colour in the 6x6x6 cube of the xterm-256 palette."""
    return 16 + 36 * round(r / 51) + 6 * round(g / 51) + round(b / 51)


def _draw(stdscr: "curses.window", rgb: np.ndarray, pairs: dict[int, int]) -> None:
    for y, row in enumerate(rgb):
        for x, rgb in enumerate(row):
            color = _xterm256(*map(int, rgb))
            if color not in pairs:
                pairs[color] = len(pairs) + 1
                curses.init_pair(pairs[color], color, color)
            stdscr.addstr(y, 2 * x, "  ", curses.color_pair(pairs[color]))


def _play(stdscr: "curses.window") -> str:
    curses.curs_set(0)
    curses.start_color()
    stdscr.nodelay(True)
    rng = np.random.default_rng()
    state, score, done = b.reset(rng), 0, False
    action, pairs = b.STAY, {}
    while True:
        deadline = time.monotonic() + TICK
        while time.monotonic() < deadline:
            key = stdscr.getch()
            if key in (ord("q"), 27):
                return f"quit, score {score}"
            if key in (curses.KEY_LEFT, ord("a")):
                action = b.LEFT
            elif key in (curses.KEY_RIGHT, ord("d")):
                action = b.RIGHT
            elif key in (curses.KEY_DOWN, ord("s"), ord(" ")):
                action = b.STAY
            time.sleep(0.005)

        state, reward, done = b.step(state, action, rng)
        score += reward > 0
        stdscr.erase()
        _draw(stdscr, b.colorize(state.grid), pairs)
        stdscr.addstr(b.H, 0, f"score {score}  lives {state.lives}  (a/d or arrows, q quits)")
        stdscr.refresh()
        if done:
            won = not (state.grid == b.BRICK).any()
            return f"{'you win' if won else 'game over'}, score {score}"


def _play_doom(stdscr: "curses.window") -> str:
    curses.curs_set(0)
    curses.start_color()
    rng = np.random.default_rng()
    state, kills, pairs = dl.reset(rng), 0, {}
    keys = {curses.KEY_UP: dl.FORWARD, ord("w"): dl.FORWARD, curses.KEY_LEFT: dl.LEFT, ord("a"): dl.LEFT,
            curses.KEY_RIGHT: dl.RIGHT, ord("d"): dl.RIGHT, ord(" "): dl.FIRE, ord("s"): dl.NOOP}
    while True:
        stdscr.erase()
        _draw(stdscr, dl.colorize(state.grid), pairs)
        stdscr.addstr(dl.H, 0, f"kills {kills}  health {state.health}  step {state.steps}")
        stdscr.addstr(dl.H + 1, 0, "w forward  a/d turn  space fire  s wait  q quit")
        stdscr.addstr(dl.H + 2, 0, "minimap (bottom right): blue = you, red = enemy.  "
                                   "red light in the HUD = the enemy moves on your next step")
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return f"quit, {kills} kills"
        if key not in keys:
            continue
        state, reward, done = dl.step(state, keys[key], rng)
        kills += reward > 0
        if done:
            return f"{'you died' if state.health <= 0 else 'time up'}, {kills} kills"


def _play_flappy(stdscr: "curses.window") -> str:
    curses.curs_set(0)
    curses.start_color()
    stdscr.nodelay(True)
    rng = np.random.default_rng()
    state, pipes, pairs = fb.reset(rng), 0, {}
    while True:
        deadline, action = time.monotonic() + 0.1, fb.NOOP
        while time.monotonic() < deadline:
            key = stdscr.getch()
            if key in (ord("q"), 27):
                return f"quit, {pipes} pipes"
            if key in (ord(" "), ord("w"), curses.KEY_UP):
                action = fb.FLAP
            time.sleep(0.005)
        state, reward, done = fb.step(state, action, rng)
        pipes += reward > 0
        stdscr.erase()
        _draw(stdscr, fb.colorize(state.grid), pairs)
        stdscr.addstr(fb.H, 0, f"pipes {pipes}   space / w / up flaps, q quits.  "
                               "The white mark on the right shows where the next gap will be.")
        stdscr.refresh()
        if done:
            return f"crashed, {pipes} pipes"


def _play_pacman(stdscr: "curses.window") -> str:
    curses.curs_set(0)
    curses.start_color()
    stdscr.nodelay(True)
    rng = np.random.default_rng()
    state, score, pairs = pm.reset(rng), 0, {}
    keys = {curses.KEY_UP: pm.UP, ord("w"): pm.UP, curses.KEY_RIGHT: pm.RIGHT, ord("d"): pm.RIGHT,
            curses.KEY_DOWN: pm.DOWN, ord("s"): pm.DOWN, curses.KEY_LEFT: pm.LEFT, ord("a"): pm.LEFT}
    while True:
        deadline, action = time.monotonic() + 0.15, pm.KEEP
        while time.monotonic() < deadline:
            key = stdscr.getch()
            if key in (ord("q"), 27):
                return f"quit, {score} pellets"
            action = keys.get(key, action)
            time.sleep(0.005)
        state, reward, done = pm.step(state, action, rng)
        score += reward > 0
        stdscr.erase()
        _draw(stdscr, pm.colorize(state.grid), pairs)
        stdscr.addstr(pm.H, 0, f"pellets {score}  lives {state.lives}   wasd / arrows turn, Pac-Man keeps going, q quits")
        stdscr.refresh()
        if done:
            return f"{'you win' if state.left() == 0 else 'game over'}, {score} pellets"


def _play_pong(stdscr: "curses.window") -> str:
    curses.curs_set(0)
    curses.start_color()
    stdscr.nodelay(True)
    rng = np.random.default_rng()
    state, action, pairs = pg.reset(rng), pg.STAY, {}
    keys = {curses.KEY_UP: pg.UP, ord("w"): pg.UP, curses.KEY_DOWN: pg.DOWN, ord("s"): pg.DOWN, ord(" "): pg.STAY}
    while True:
        deadline = time.monotonic() + TICK
        while time.monotonic() < deadline:
            key = stdscr.getch()
            if key in (ord("q"), 27):
                return f"quit, {state.score}:{state.lost}"
            action = keys.get(key, action)
            time.sleep(0.005)
        state, _, done = pg.step(state, action, rng)
        stdscr.erase()
        _draw(stdscr, pg.colorize(state.grid), pairs)
        stdscr.addstr(pg.H, 0, f"you {state.score} : {state.lost} cpu   w/s or up/down move, space stops, q quits")
        stdscr.refresh()
        if done:
            return f"{'you win' if state.score >= pg.POINTS else 'cpu wins'}, {state.score}:{state.lost}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=("breakout", "doomlite", "flappy", "pacman", "pong"), default="breakout")
    game = p.parse_args().game
    print(curses.wrapper({"breakout": _play, "doomlite": _play_doom, "flappy": _play_flappy, "pacman": _play_pacman, "pong": _play_pong}[game]))


if __name__ == "__main__":
    main()
