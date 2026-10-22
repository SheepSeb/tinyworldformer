"""The games the world model can dream, and what the pipeline needs to know about each."""

from dataclasses import dataclass, field
from types import ModuleType

from tinyworldformer.games import breakout, doomlite, flappy, pacman, pong


@dataclass(frozen=True)
class Game:
    name: str
    module: ModuleType  # reset(rng, ...), step(state, action, rng), State.grid / .lives, colorize
    patch: int
    rewards: tuple[float, ...]  # reward value of each reward class, ascending
    # rewards whose next frame holds the game's random event (for data recorded before the
    # per-step `random` flag; Flappy Bird's pipe spawns only exist as that flag)
    random_rewards: tuple[float, ...]
    # (first token, end token, token underneath): exactly one cell of every frame holds a token
    # of each group. Decoded as one global choice; groups take the top token ids.
    singletons: tuple[tuple[int, int, int], ...]
    lives: int  # dream counter: a negative reward costs one
    done_when_absent: int  # episode ends once no cell holds this token (-1: never)
    max_steps: int
    # (singleton token, row offset, col offset, token): sprite cells drawn around a placed
    # singleton instead of predicted. Their tokens are background ids that only appear there.
    stamps: tuple[tuple[int, int, int, int], ...] = ()
    # (groups, H, W) bool: the cells each singleton group can ever occupy in the real game; the
    # dream never places one elsewhere (a respawn sampled into a wall would stay there). None: anywhere.
    masks: object = field(default=None, compare=False, repr=False)
    # (H, W, V) bool: the background tokens each cell can hold in the real game. A slip in a cell
    # that never changes (a wall) would otherwise be copied for the rest of a dream. None: any.
    cell_tokens: object = field(default=None, compare=False, repr=False)

    @property
    def H(self) -> int:
        return self.module.H

    @property
    def W(self) -> int:
        return self.module.W

    @property
    def V(self) -> int:
        return self.module.NUM_TOKENS

    @property
    def A(self) -> int:
        return self.module.NUM_ACTIONS

    @property
    def GH(self) -> int:
        return self.H // self.patch

    @property
    def GW(self) -> int:
        return self.W // self.patch

    @property
    def T(self) -> int:
        return self.GH * self.GW

    @property
    def SUB(self) -> int:
        return self.patch * self.patch

    @property
    def STAMPED(self) -> list[int]:
        """Background tokens that only exist as stamps."""
        return sorted({tok for *_, tok in self.stamps})

    def stamp_under(self, tok: int) -> int:
        """The token underneath a stamped token: that of its singleton's group."""
        anchor = next(t for t, _, _, s in self.stamps if s == tok)
        return next(under for lo, hi, under in self.singletons if lo <= anchor < hi)

    @property
    def BG(self) -> int:
        """Number of background tokens (all ids below the first singleton group)."""
        return min(lo for lo, _, _ in self.singletons)


GAMES = {
    "breakout": Game(
        "breakout", breakout, patch=2, rewards=(-1.0, 0.0, 1.0, 2.0), random_rewards=(-1.0,),
        singletons=((breakout.BALL_UL, breakout.NUM_TOKENS, breakout.EMPTY),),
        lives=breakout.LIVES, done_when_absent=breakout.BRICK, max_steps=5000,
    ),
    "flappy": Game(
        "flappy", flappy, patch=4, rewards=(-1.0, 0.0, 1.0), random_rewards=(),  # random events are flagged per step
        singletons=((flappy.MARKER, flappy.BIRD, flappy.HUD), (flappy.BIRD, flappy.NUM_TOKENS, flappy.SKY)),
        lives=1, done_when_absent=-1, max_steps=flappy.MAX_STEPS,
    ),
    "doomlite": Game(
        "doomlite", doomlite, patch=4, rewards=(-1.0, 0.0, 1.0), random_rewards=(1.0,),
        singletons=((doomlite.MAP_ENEMY, doomlite.MAP_PLAYER, doomlite.MAP_FLOOR),
                    (doomlite.MAP_PLAYER, doomlite.NUM_TOKENS, doomlite.MAP_FLOOR)),
        lives=doomlite.HEALTH, done_when_absent=-1, max_steps=doomlite.MAX_STEPS,
    ),
    "pacman": Game(
        "pacman", pacman, patch=6, rewards=(-1.0, 0.0, 1.0), random_rewards=(-1.0,),  # a catch respawns the ghosts
        singletons=((pacman.GHOST_A, pacman.GHOST_B, pacman.EMPTY), (pacman.GHOST_B, pacman.PAC, pacman.EMPTY),
                    (pacman.PAC, pacman.NUM_TOKENS, pacman.EMPTY)),
        lives=pacman.LIVES, done_when_absent=pacman.PELLET, max_steps=pacman.MAX_STEPS, stamps=pacman.STAMPS,
        masks=pacman.singleton_masks(), cell_tokens=pacman.cell_tokens(),
    ),
    "pong": Game(
        "pong", pong, patch=4, rewards=(-1.0, 0.0, 1.0), random_rewards=(-1.0, 1.0),  # every point is followed by a serve
        singletons=((pong.BALL, pong.NUM_TOKENS, pong.EMPTY),),
        lives=pong.POINTS, done_when_absent=pong.SLOT_P, max_steps=5000,
    ),
}


def get(name: str) -> Game:
    return GAMES[name]
