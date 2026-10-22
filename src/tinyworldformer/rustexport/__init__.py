"""Export small PyTorch models to `no_std` Rust, bit for bit.

Self-contained on purpose (it imports only numpy and torch, never the rest of tinyworldformer),
so other projects can reuse it:

- `qat`: quantization-aware building blocks (`QLinear`, `ActQuant`, `LoopedBlock`, `softmax_u8`).
  Train with these and the model has an exact int8 twin.
- `intref`: that int8 twin as a numpy spec, with every float op in a fixed order. A Rust engine
  that implements it reproduces it bit for bit (`tinyworldformer-rs/src/model.rs` is one).
- `convert`: trained modules -> the integer spec -> Rust statics; plain float MLPs -> Rust statics.
- `codegen`: Rust literals that keep every bit (`rust_array`, `rust_const`, `write_module`) and
  golden-vector files for the Rust tests (`Golden`).
"""

from tinyworldformer.rustexport import codegen, convert, intref, qat
from tinyworldformer.rustexport.codegen import Golden, rust_array, rust_const, write_module

__all__ = ["Golden", "codegen", "convert", "intref", "qat", "rust_array", "rust_const", "write_module"]
