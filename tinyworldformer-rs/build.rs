//! Picks the world model ("env") and the agent this build runs.
//!
//! Exports live side by side under `models/`:
//!
//!     models/envs/<env>/weights.rs, golden.bin                     (python -m tinyworldformer.training.export)
//!     models/agents/<game>/<agent>/agent_weights.rs, golden_*.bin  (python -m tinyworldformer.training.streamx --export)
//!     models/default                                               env = ..., agent = ...
//!
//! `TWF_ENV` and `TWF_AGENT` choose (else `models/default`); the agent must be one exported for the
//! env's game. With no agent named, an env takes its game's agent from `models/default` if that
//! one fits, else the first by name. The choice reaches the code as `TWF_ENV_DIR` /
//! `TWF_AGENT_DIR` (included by `lib.rs`, read by the golden tests) and `TWF_ENV_NAME` /
//! `TWF_AGENT_NAME` (shown on screen).

use std::path::{Path, PathBuf};
use std::{env, fs};

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .map(|it| it.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn game_of(weights: &Path) -> String {
    let src = fs::read_to_string(weights).unwrap_or_else(|e| panic!("{}: {e}", weights.display()));
    let line = src.lines().find(|l| l.starts_with("pub const GAME: &str = ")).expect("weights.rs without GAME");
    line.split('"').nth(1).unwrap().to_string()
}

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("models");
    println!("cargo:rerun-if-env-changed=TWF_ENV");
    println!("cargo:rerun-if-env-changed=TWF_AGENT");
    println!("cargo:rerun-if-changed=models");

    let defaults: Vec<(String, String)> = fs::read_to_string(root.join("default"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let default = |key: &str| defaults.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());

    let envs = names(&root.join("envs"));
    let env_name = env::var("TWF_ENV").ok().or_else(|| default("env")).unwrap_or_else(|| {
        panic!("no env chosen: set TWF_ENV to one of {envs:?} (or write models/default)")
    });
    let env_dir = root.join("envs").join(&env_name);
    if !env_dir.join("weights.rs").exists() {
        panic!("unknown env {env_name:?}: exported envs are {envs:?}");
    }
    let game = game_of(&env_dir.join("weights.rs"));

    let agents = names(&root.join("agents").join(&game));
    let agent_name = env::var("TWF_AGENT")
        .ok()
        .or_else(|| default("agent").filter(|a| agents.contains(a)))
        .or_else(|| agents.first().cloned())
        .unwrap_or_else(|| panic!("no agent exported for {game} (env {env_name}): export one with streamx --export"));
    let agent_dir = root.join("agents").join(&game).join(&agent_name);
    if !agent_dir.join("agent_weights.rs").exists() {
        panic!("unknown agent {agent_name:?} for {game} (env {env_name}): agents for {game} are {agents:?}");
    }

    println!("cargo:rustc-env=TWF_ENV_DIR={}", env_dir.display());
    println!("cargo:rustc-env=TWF_AGENT_DIR={}", agent_dir.display());
    println!("cargo:rustc-env=TWF_ENV_NAME={env_name}");
    println!("cargo:rustc-env=TWF_AGENT_NAME={agent_name}");
    println!("cargo:warning=tinyworldformer: env {env_name} ({game}), agent {agent_name}");
}
