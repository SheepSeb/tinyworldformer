"""Turn a world-model checkpoint into the integer model, check it, and write it out for the
tinyworldformer-rs crate. The generic parts live in `tinyworldformer.rustexport`; this file adds
what the crate needs to know about the game (sizes, singletons, colours, opening frames).

    python -m tinyworldformer.training.export --ckpt models/wm.pt --name breakout

Writes the env `<name>` (default: the game's name) of the crate: `models/envs/<name>/weights.rs`
(every float as its exact bit pattern) and `golden.bin` (inputs plus the spec's logits, which
the Rust engine must reproduce bit-for-bit). Builds pick an env with `TWF_ENV=<name>`.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from tinyworldformer.games import Game
from tinyworldformer.rustexport import Golden, intref, rust_array, rust_const, write_module
from tinyworldformer.rustexport.convert import int_model_arrays, looped_model
from tinyworldformer.rustexport.qat import LUT_N
from tinyworldformer.training.dataset import Transitions
from tinyworldformer.training.model import ActQuant, Config, WorldModel, decode, patchify

f32 = np.float32


def load(ckpt: Path) -> WorldModel:
    ck = torch.load(ckpt, map_location="cpu")
    model = WorldModel(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    return model.eval()


def to_int(model: WorldModel) -> intref.IntModel:
    return looped_model(model.block, tok=model.tok, pos=model.pos, act=model.act, norm=model.norm,
                        cell=model.cell_head, reward=model.reward_head, loops=model.c.loops)


def predict(m: intref.IntModel, g: Game, grid: np.ndarray, action: np.ndarray, loops: int):
    patches = patchify(torch.from_numpy(grid).long(), g).numpy()
    cells, reward = intref.forward(m, patches, action.astype(np.int64), loops)
    cells = torch.from_numpy(cells).view(len(grid), g.T, g.SUB, g.V)
    return decode(cells, g).numpy(), reward.argmax(-1)


def agreement(model: WorldModel, m: intref.IntModel, d: dict, loops: int, n: int = 4096) -> str:
    """Accuracy (frames after a random event excluded) of the integer spec and of the QAT torch
    model, and how often the two decode to the same frame."""
    g = model.g
    grid, action, nxt = d["grid"][:n], d["action"][:n], d["next_grid"][:n]
    serve = d["random"][:n]
    pi, ri = predict(m, g, grid, action, loops)
    ActQuant.enabled = True
    with torch.no_grad():
        cells, rew = model(torch.from_numpy(grid), torch.from_numpy(action), loops)[-1]
    pt, rt = decode(cells, g).numpy(), rew.argmax(-1).numpy()
    frame = lambda p: ((p == nxt).all((1, 2)) & ~serve).sum() / (~serve).sum()  # noqa: E731
    same = (pi == pt).all((1, 2)).mean()
    return (f"loops {loops}: int frame acc {frame(pi):.4f}, torch frame acc {frame(pt):.4f}, "
            f"int==torch {same:.4f}, reward int==torch {(ri == rt).mean():.4f}")


HUMAN_FRAME_MS = {"breakout": 66, "doomlite": 150, "flappy": 100, "pacman": 150, "pong": 66}  # how fast a dreamed frame ticks while a person plays


def cell_bits(g: Game) -> np.ndarray:
    """`Game.cell_tokens` as one u32 bit set per cell over the background tokens (all set: any)."""
    assert g.BG <= 32
    allowed = g.cell_tokens[..., : g.BG] if g.cell_tokens is not None else np.ones((g.H, g.W, g.BG), bool)
    return (allowed.astype(np.uint64) << np.arange(g.BG, dtype=np.uint64)).sum(-1).astype(np.uint32)


def write_rust(m: intref.IntModel, g: Game, path: Path) -> None:
    rng = np.random.default_rng(0)
    inits = np.unique(np.stack([g.module.reset(rng).grid for _ in range(16)]), axis=0)
    colors = np.stack([g.module.colorize(np.full((g.H, g.W), t, np.uint8))[:, 0] for t in range(g.V)], 1)  # (H, V, 3)
    consts = [
        rust_const("GAME", g.name, "&str"),
        rust_const("D", m.d), rust_const("HEADS", m.heads), rust_const("FF", m.up.w.shape[0]),
        rust_const("LOOPS", m.loops), rust_const("P", g.patch), rust_const("T", g.T), rust_const("SUB", g.SUB),
        rust_const("V", g.V), rust_const("ACTIONS", g.A), rust_const("H", g.H), rust_const("W", g.W),
        rust_const("LUT_N", LUT_N), rust_const("REWARD", m.reward.w.shape[0]),
        rust_const("BG", g.BG), rust_const("LIVES", g.lives, "u8"),
        rust_const("DONE_WHEN_ABSENT", g.done_when_absent, "i16"), rust_const("MAX_STEPS", g.max_steps, "u32"),
        rust_const("HUMAN_FRAME_MS", HUMAN_FRAME_MS[g.name], "u64"), rust_const("N_INIT", len(inits)),
        "/// (first token, end token, token underneath) of each group placed exactly once per frame\n",
        f"pub static SINGLETONS: [(u8, u8, u8); {len(g.singletons)}] = [{', '.join(map(str, g.singletons))}];\n",
        "/// (singleton token, row offset, col offset, token): sprite cells drawn around a placed singleton\n",
        f"pub static STAMPS: [(u8, i8, i8, u8); {len(g.stamps)}] = [{', '.join(map(str, g.stamps))}];\n",
        "/// per cell, bit t set if background token t is allowed there\n",
        rust_array("CELL_TOKENS", cell_bits(g), "u32"),
        "/// 1 where each singleton group can ever be in the real game, per group then row-major cell\n",
        rust_array("SINGLETON_MASK", g.masks if g.masks is not None else np.ones((len(g.singletons), g.H, g.W), bool), "u8"),
        rust_array("REWARD_VALUES", np.array(g.rewards), "f32"),
        rust_array("INIT_GRIDS", inits, "u8"),
        "/// RGB of token t on row r at [(r * V + t) * 3..]\n",
        rust_array("COLORS", colors, "u8"),
    ]
    write_module(path, consts + int_model_arrays(m), "python -m tinyworldformer.training.export")


def write_golden(m: intref.IntModel, g: Game, d: dict, path: Path, n: int = 24) -> None:
    """Per sample: grid u8[H*W], action u8, loops u8, cells f32[T*SUB*V], reward f32[R]."""
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(len(d["action"]), n - 8, replace=False),
                          np.flatnonzero(d["reward"] != 0)[:8]])  # include hits and misses
    out = Golden().u32(len(idx))
    for j, i in enumerate(idx):
        loops = 1 + j % m.loops
        grid, action = d["grid"][i : i + 1], d["action"][i : i + 1]
        patches = patchify(torch.from_numpy(grid).long(), g).numpy()
        cells, reward = intref.forward(m, patches, action.astype(np.int64), loops)
        out.array(grid, np.uint8).u8(int(action[0]), loops).array(cells).array(reward)
    out.save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("models/wm.pt"))
    p.add_argument("--data", type=Path, default=None, help="held-out transitions (default: data/<game>/val.npz)")
    p.add_argument("--rust", type=Path, default=Path("tinyworldformer-rs"))
    p.add_argument("--name", default=None, help="env name in the crate (default: the game's name)")
    args = p.parse_args()

    model = load(args.ckpt)
    m = to_int(model)
    d = Transitions(args.data or Path("data") / model.g.name / "val.npz").d
    for loops in range(1, m.loops + 1):
        print(agreement(model, m, d, loops))
    out = args.rust / "models" / "envs" / (args.name or model.g.name)
    write_rust(m, model.g, out / "weights.rs")
    write_golden(m, model.g, d, out / "golden.bin")
    print(f"wrote env {out.name} ({model.g.name}): {out}/weights.rs and golden vectors")


if __name__ == "__main__":
    main()
