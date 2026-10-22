"""`tinyworldformer.rustexport` is meant to be reused in other projects: it must stand alone."""

import ast
import pathlib

import numpy as np
import torch

from tinyworldformer.rustexport import Golden, intref, rust_array, rust_const
from tinyworldformer.rustexport.convert import looped_model
from tinyworldformer.rustexport.qat import ActQuant, LoopedBlock, QLinear

PKG = pathlib.Path(__file__).parents[1] / "src" / "tinyworldformer" / "rustexport"


def test_imports_nothing_else_from_the_project():
    for path in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tinyworldformer"):
                assert node.module.startswith("tinyworldformer.rustexport"), f"{path.name} imports {node.module}"
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("tinyworldformer") for a in node.names), path.name


def test_rust_literals_keep_every_bit():
    x = np.array([0.1, -2.5, np.float32(1) / 3], np.float32)
    src = rust_array("X", x, "f32")
    bits = [int(t.split("0x")[1].rstrip(")"), 16) for t in src[src.index("[", 20) + 1 : src.rindex("]")].split(", ")]
    assert np.array_equal(np.array(bits, np.uint32).view(np.float32), x)
    assert rust_array("Q", np.array([-3, 4], np.int8), "i8") == "pub static Q: [i8; 2] = [-3, 4];\n"
    assert rust_const("N", 7) == "pub const N: usize = 7;\n"
    assert rust_const("GAME", "pong", "&str") == 'pub const GAME: &str = "pong";\n'


def test_golden_layout(tmp_path):
    Golden().u32(2).u8(5, 6).array(np.array([1.5], np.float32)).save(tmp_path / "g.bin")
    assert (tmp_path / "g.bin").read_bytes() == b"\x02\x00\x00\x00\x05\x06" + np.float32(1.5).tobytes()


def test_a_looped_transformer_converts_to_a_matching_integer_spec():
    torch.manual_seed(0)
    d, loops, T, sub, V, A = 16, 2, 6, 4, 5, 3
    block, norm = LoopedBlock(d, 2, 32, loops), torch.nn.RMSNorm(d, eps=1e-5)
    tok, pos, act = torch.randn(sub, V, d), torch.randn(T, d), torch.randn(A, d)
    cell, reward = QLinear(d, sub * V, loops), QLinear(d, 3, loops)
    patches, action = torch.randint(V, (8, T, sub)), torch.randint(A, (8,))

    def forward(n_loops):
        inp = sum(tok[s][patches[..., s]] for s in range(sub)) + pos + act[action][:, None]
        x = torch.zeros_like(inp)
        for i in range(n_loops):
            x = block(x + inp, i)
        h = norm(x)
        return cell(h, n_loops - 1), reward(h.mean(1), n_loops - 1)

    ActQuant.enabled = False
    for _ in range(3):  # calibrate the activation scales
        forward(loops)
    block.eval(), cell.eval(), reward.eval()
    ActQuant.enabled = True
    try:
        with torch.no_grad():
            want, _ = forward(loops)
        m = looped_model(block, tok=tok, pos=pos, act=act, norm=norm, cell=cell, reward=reward, loops=loops)
        got, _ = intref.forward(m, patches.numpy(), action.numpy(), loops)
    finally:
        ActQuant.enabled = False
    assert np.abs(got - want.numpy()).max() < 0.05 * np.abs(want.numpy()).max()
