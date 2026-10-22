"""Train the looped world model with QAT.

    python -m tinyworldformer.training.train --steps 30000 --out models/wm.pt
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tinyworldformer.games import breakout as b
from tinyworldformer.training.dataset import Transitions
from tinyworldformer.games import Game
from tinyworldformer.training.model import ActQuant, Config, WorldModel, decode, patchify, unpatchify


def to_device(paths: list[Path], device: str, max_gpu_bytes: float = 4e9) -> tuple[dict[str, torch.Tensor], str]:
    """Load and concatenate datasets; on the GPU when they fit in `max_gpu_bytes`, else they stay
    in CPU memory and each batch is copied over (Doom-lite frames are 1 KB each)."""
    keys = ("grid", "action", "next_grid", "reward_cls", "random")
    ds = [Transitions(p) for p in paths]
    games = {t.game for t in ds}
    assert len(games) == 1, f"mixed games in {paths}: {games}"
    data = {k: torch.from_numpy(np.concatenate([t.d[k] for t in ds])) for k in keys}
    if sum(v.nbytes for v in data.values()) <= max_gpu_bytes:
        data = {k: v.to(device) for k, v in data.items()}
    return data, games.pop()


def focus(grid: torch.Tensor, g: Game) -> tuple[torch.Tensor, torch.Tensor]:
    """Where things can happen, judged from the *input* frame only (so the weights do not bias
    the predicted probabilities). Breakout: cells within 2 of the ball plus the paddle rows, and
    whether the ball is close enough to the bricks or the bottom for a reward event. Other games
    weight everything equally."""
    if g.name != "breakout":
        ones = torch.ones(grid.shape[0], device=grid.device, dtype=torch.bool)
        return torch.ones_like(patchify(grid, g), dtype=torch.bool), ones
    ball = (grid >= b.BALL_UL).float()[:, None]
    near = F.max_pool2d(ball, 5, stride=1, padding=2)[:, 0] > 0
    near[:, b.PADDLE_ROW - 1 :] = True
    ball_row = (grid >= b.BALL_UL).any(2).float().argmax(1)
    event_zone = (ball_row <= b.BRICK_ROWS.stop + 1) | (ball_row >= b.PADDLE_ROW - 1)
    return patchify(near, g), event_zone


def corrupt(grid: torch.Tensor, noise: float, game: str = "doomlite") -> torch.Tensor:
    """Noise augmentation (as in GameNGen), so the model redraws what it can instead of copying it.

    Pac-Man: flip a few maze cells between wall and path in the *input*. The maze never changes,
    so the model learns to draw it from position; otherwise one wrong wall cell stays wrong for
    the rest of a dream.

    Doom-lite: damage the *input* 3D view so the model
    learns to redraw it from the game state (minimap, HUD) instead of copying it. Otherwise one
    dropped enemy sprite stays dropped for the rest of a dream. Per frame, with probability
    `noise`: scatter random view pixels, and half of those frames also lose the enemy sprite."""
    if game == "pacman":
        from tinyworldformer.games import pacman as pm

        B = grid.shape[0]
        hit = torch.rand(B, 1, 1, device=grid.device) < noise
        maze = torch.zeros_like(grid, dtype=torch.bool)
        maze[:, : pm.S * pm.MAP_H, : pm.S * pm.MAP_W] = True
        rate = torch.rand(B, 1, 1, device=grid.device) * 0.03
        flip = hit & maze & (torch.rand(grid.shape, device=grid.device) < rate)
        wall = (grid == pm.WALL) | (grid == pm.WALL_FILL)
        grid = torch.where(flip & wall, pm.EMPTY, grid)
        return torch.where(flip & (grid == pm.EMPTY) & ~wall, pm.WALL, grid)
    from tinyworldformer.games import doomlite as dl

    B = grid.shape[0]
    hit = torch.rand(B, 1, 1, device=grid.device) < noise
    view = torch.zeros_like(grid, dtype=torch.bool)
    view[:, : dl.VIEW_H] = True
    rate = torch.rand(B, 1, 1, device=grid.device) * 0.05
    speckle = hit & view & (torch.rand(grid.shape, device=grid.device) < rate)
    grid = torch.where(speckle, torch.randint_like(grid, 0, dl.HURT), grid)
    erase = hit & (torch.rand(B, 1, 1, device=grid.device) < 0.5) & view & ((grid == dl.ENEMY) | (grid == dl.ENEMY_EYE))
    return torch.where(erase, dl.WALL_DARK_FAR, grid)


def losses(model, batch, focus_w: float, event_w: float, singleton_w: float = 2.0, noise: float = 0.0):
    g = model.g
    grid, nxt = batch["grid"].long(), batch["next_grid"].long()
    background = patchify(nxt, g)  # (B, T, SUB)
    for lo, hi, under in g.singletons:  # the singleton objects sit on an `under` cell
        background = torch.where((background >= lo) & (background < hi), under, background)
    for tok in g.STAMPED:  # so do the sprite cells drawn around them
        background = torch.where(background == tok, g.stamp_under(tok), background)
    near, event_zone = focus(grid, g)
    cell_w = (1 + focus_w * near.float()).flatten()
    targets = []
    for lo, hi, _ in g.singletons:
        inside = (nxt >= lo) & (nxt < hi)
        pos = inside.flatten(1).float().argmax(1)
        targets.append(pos * (hi - lo) + nxt.flatten(1).gather(1, pos[:, None])[:, 0] - lo)
    reward_w = 1 + event_w * event_zone.float()
    total = 0.0
    model_in = corrupt(grid, noise, g.name) if noise > 0 else grid
    for cells, reward in model(model_in, batch["action"]):
        bg = F.cross_entropy(cells[..., : g.BG].flatten(0, 2), background.flatten(), reduction="none")
        total = total + (bg * cell_w).sum() / cell_w.sum()
        for (lo, hi, _), target in zip(g.singletons, targets):
            logits = unpatchify(cells[..., lo:hi], g).flatten(1)
            total = total + singleton_w * F.cross_entropy(logits, target)
        rce = F.cross_entropy(reward, batch["reward_cls"].long(), reduction="none")
        total = total + (rce * reward_w).sum() / reward_w.sum()
    return total


@torch.no_grad()
def evaluate(model, data, batch_size=4096) -> list[dict[str, float]]:
    """Per loop: exact next-frame accuracy (frames after a random event excluded), all singleton
    objects placed right, and reward accuracy on non-zero rewards."""
    model.eval()
    g = model.g
    zero = g.rewards.index(0.0)
    n = len(data["action"])
    stats = None
    for i in range(0, n, batch_size):
        batch = {k: v[i : i + batch_size] for k, v in data.items()}
        nxt = batch["next_grid"]
        random = batch["random"]
        event = batch["reward_cls"] != zero
        outs = model(batch["grid"], batch["action"])
        if stats is None:
            stats = [dict(frame=0.0, objects=0.0, event=0.0) for _ in outs]
        for s, (cells, reward) in zip(stats, outs):
            pred = decode(cells, g)
            s["frame"] += ((pred == nxt).flatten(1).all(1) & ~random).sum().item()
            single = (pred >= g.BG) == (nxt >= g.BG)
            s["objects"] += single.flatten(1).all(1).sum().item()
            s["event"] += ((reward.argmax(-1) == batch["reward_cls"]) & event).sum().item()
    n_random = data["random"].sum().item()
    n_event = max((data["reward_cls"] != zero).sum().item(), 1)
    model.train()
    return [dict(frame=s["frame"] / (n - n_random), objects=s["objects"] / n, event=s["event"] / n_event) for s in stats]


def fmt(stats: list[dict[str, float]]) -> str:
    return " | ".join(
        f"L{i + 1} frame {s['frame']:.4f} objects {s['objects']:.4f} rew_evt {s['event']:.3f}"
        for i, s in enumerate(stats)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, nargs="+", default=[Path("data/breakout")], help="dataset folders under data/")
    p.add_argument("--init", type=Path, default=None, help="fine-tune from this checkpoint")
    p.add_argument("--out", type=Path, default=Path("models/wm.pt"))
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--warmup-float", type=int, default=3000, help="steps before QAT turns on")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--ff", type=int, default=64)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--focus-w", type=float, default=5.0)
    p.add_argument("--event-w", type=float, default=3.0)
    p.add_argument("--noise", type=float, default=0.0,
                   help="share of input frames damaged (Doom-lite: the 3D view; Pac-Man: maze cells)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snapshots", type=lambda s: {int(float(x)) for x in s.split(",")}, default=set(),
                   help="also keep the model at these steps (e.g. 0,100,1e3): <out>_<step>.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, game = to_device([d / "train.npz" for d in args.data], dev)
    assert args.noise == 0 or game in ("doomlite", "pacman"), "--noise is defined for Doom-lite and Pac-Man"
    val, _ = to_device([d / "val.npz" for d in args.data], dev)
    if args.init:
        ck = torch.load(args.init)
        model = WorldModel(Config(**ck["config"])).to(dev)
        model.load_state_dict(ck["state"])
        assert model.g.name == game, f"checkpoint is for {model.g.name}, data is {game}"
    else:
        model = WorldModel(Config(args.d, args.heads, args.ff, args.loops, game)).to(dev)
    print(f"game {game}, params {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * s / args.steps))
    )

    loss_fn = torch.compile(losses)  # fuses the elementwise QAT ops: ~4x faster
    n, t0 = len(train["action"]), time.time()
    ActQuant.enabled = False
    for step in range(args.steps):
        if step in args.snapshots:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": model.config(), "state": model.state_dict()}, args.out.with_name(f"{args.out.stem}_{step}.pt"))
        if step == args.warmup_float:
            ActQuant.enabled = True
            print("QAT on")
        on = train["action"].device
        idx = torch.randint(n, (args.batch,), device=on)
        batch = {k: v[idx].to(dev, non_blocking=True) for k, v in train.items()}
        loss = loss_fn(model, batch, args.focus_w, args.event_w, noise=args.noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == args.steps - 1:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s | {fmt(evaluate(model, val))}",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config(), "state": model.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
