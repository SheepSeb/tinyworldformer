"""Looped transformer world model with int8 quantization-aware training.

One frame + action in, next frame + reward out. The grid is cut into patches (Breakout: 2x2, 80
tokens; Doom-lite: 4x4, 64 tokens), a single weight-shared block is applied `loops` times, and
every loop is supervised so the loop count is a speed/quality knob at inference.

The block and every quantization point come from `tinyworldformer.rustexport.qat`, which
`rustexport.intref` mirrors as the integer spec that the Rust engine implements bit-exactly.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyworldformer.games import Game, get
from tinyworldformer.rustexport.qat import LUT_N, LUT_RES, ActQuant, LoopedBlock, QLinear  # noqa: F401 (re-exported)


@dataclass
class Config:
    d: int = 32
    heads: int = 2
    ff: int = 64
    loops: int = 4
    game: str = "breakout"


def patchify(grid: torch.Tensor, g: Game) -> torch.Tensor:
    """(B, H, W) -> (B, T, SUB), sub-cell index = dy * P + dx."""
    B, P = grid.shape[0], g.patch
    return grid.view(B, g.GH, P, g.GW, P).permute(0, 1, 3, 2, 4).reshape(B, g.T, g.SUB)


def unpatchify(x: torch.Tensor, g: Game) -> torch.Tensor:
    """(B, T, SUB, ...) -> (B, H, W, ...)."""
    B, P, rest = x.shape[0], g.patch, x.shape[3:]
    x = x.reshape(B, g.GH, g.GW, P, P, *rest).permute(0, 1, 3, 2, 4, *range(5, 5 + len(rest)))
    return x.reshape(B, g.H, g.W, *rest)


class WorldModel(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c, self.g = c, get(c.game)
        g = self.g
        self.tok = nn.Parameter(torch.randn(g.SUB, g.V, c.d) * 0.3)
        self.pos = nn.Parameter(torch.randn(g.T, c.d) * 0.3)
        self.act = nn.Parameter(torch.randn(g.A, c.d) * 0.3)
        self.block = LoopedBlock(c.d, c.heads, c.ff, c.loops)
        self.norm = nn.RMSNorm(c.d, eps=1e-5)
        self.cell_head = QLinear(c.d, g.SUB * g.V, c.loops)
        self.reward_head = QLinear(c.d, len(g.rewards), c.loops)

    def embed(self, grid: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # one-hot matmuls: a gather's backward into a small table is slow on GPU
        onehot = F.one_hot(patchify(grid.long(), self.g), self.g.V).to(self.tok.dtype)
        x = torch.einsum("btsv,svd->btd", onehot, self.tok)
        act = F.one_hot(action.long(), self.g.A).to(self.act.dtype) @ self.act
        return x + self.pos + act[:, None]

    def forward(self, grid: torch.Tensor, action: torch.Tensor, loops: int | None = None):
        """Returns one (cell_logits (B,T,SUB,V), reward_logits (B,R)) pair per loop."""
        inp = self.embed(grid, action)
        x, outs = torch.zeros_like(inp), []
        for i in range(loops or self.c.loops):
            x = self.block(x + inp, i)
            h = self.norm(x)
            cells = self.cell_head(h, i).view(*h.shape[:2], self.g.SUB, self.g.V)
            outs.append((cells, self.reward_head(h.mean(1), i)))
        return outs

    def config(self) -> dict:
        return asdict(self.c)


def decode(cells: torch.Tensor, g: Game, generator: torch.Generator | None = None, window: float = 2.2) -> torch.Tensor:
    """Cell logits (B, T, SUB, V) -> next grid (B, H, W) uint8.

    Every cell takes its argmax among the background tokens (stamped tokens excluded). Then each
    singleton group (Breakout's ball, Doom-lite's minimap enemy and player, Pac-Man and the ghosts)
    is placed exactly once: the argmax over all
    (cell, token) pairs, or with `generator` a sample among the pairs whose logit is within
    `window` of the best (that is how random events, the serve or the respawn, are dreamed). Last,
    the game's stamps draw the rest of each placed sprite around it.
    """
    c = unpatchify(cells, g)  # (B, H, W, V)
    B = c.shape[0]
    bg = c[..., : g.BG]
    if g.stamps:  # stamped tokens are drawn around their singleton, never predicted
        bg = bg.clone()
        bg[..., g.STAMPED] = float("-inf")
    grid = bg.argmax(-1).to(torch.uint8).view(B, -1)
    rows = torch.arange(B, device=grid.device)
    placed = []
    for lo, hi, _ in g.singletons:
        logits = c[..., lo:hi].reshape(B, -1)
        if generator is None:
            idx = logits.argmax(-1)
        else:
            best = logits.amax(-1, keepdim=True)
            p = torch.where(logits >= best - window, (logits - best).exp(), torch.zeros_like(logits))
            idx = torch.multinomial(p, 1, generator=generator)[:, 0]
        cell, tok = idx // (hi - lo), lo + idx % (hi - lo)
        grid[rows, cell] = tok.to(torch.uint8)
        placed.append((cell, tok))
    taken = torch.stack([cell for cell, _ in placed], 1) if placed else None
    for anchor, dr, dc, tok in g.stamps:  # never over another singleton
        for cell, t in placed:
            r, col = cell // g.W + dr, cell % g.W + dc
            ok = (t == anchor) & (r >= 0) & (r < g.H) & (col >= 0) & (col < g.W)
            ok &= ~(taken == (r * g.W + col)[:, None]).any(1)
            grid[rows[ok], (r * g.W + col)[ok]] = tok
    return grid.view(B, g.H, g.W)
