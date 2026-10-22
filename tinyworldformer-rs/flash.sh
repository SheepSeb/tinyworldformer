#!/usr/bin/env bash
# Build and flash the Pico 2 W with a chosen world model ("env") and agent, or run them on the host.
#
#   ./flash.sh --list                               exported envs and agents
#   ./flash.sh --env doomlite --agent doom_agent_roam_p0.1 [--display]
#   ./flash.sh --env pacman --host                  play it in this terminal instead
#   ./flash.sh --env flappy --build                 only build the firmware
#   ./flash.sh --env pong --default                 also make this the default for plain cargo builds
#
# Leaving out --agent takes the env's game's agent from models/default when it fits, else the
# first one by name. Flashing needs picotool and the board in BOOTSEL mode (hold BOOTSEL while
# plugging it in).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
models="$here/models"
env="" agent="" mode=flash features=() make_default=0

game_of() { sed -n 's/^pub const GAME: &str = "\(.*\)";$/\1/p' "$models/envs/$1/weights.rs"; }

list() {
    echo "envs (world models):"
    for d in "$models"/envs/*/; do
        n="$(basename "$d")"; echo "  $n  ($(game_of "$n"))"
    done
    echo "agents, per game:"
    for g in "$models"/agents/*/; do
        echo "  $(basename "$g"): $(ls "$g" | tr '\n' ' ')"
    done
    echo "default: $(tr '\n' ' ' < "$models/default" 2>/dev/null || echo none)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list) list; exit 0 ;;
        --env) env="$2"; shift ;;
        --agent) agent="$2"; shift ;;
        --display) features=(--features display) ;;
        --host) mode=host ;;
        --build) mode=build ;;
        --default) make_default=1 ;;
        -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1 (try --help)"; exit 1 ;;
    esac
    shift
done

if [[ -n "$env" && ! -f "$models/envs/$env/weights.rs" ]]; then
    echo "no env '$env'"; list; exit 1
fi
if [[ -n "$agent" ]]; then
    game="$(game_of "${env:-$(sed -n 's/^env *= *//p' "$models/default")}")"
    if [[ ! -f "$models/agents/$game/$agent/agent_weights.rs" ]]; then
        echo "no agent '$agent' for $game"; list; exit 1
    fi
fi

export TWF_ENV="${env:-}" TWF_AGENT="${agent:-}"
[[ -z "$TWF_ENV" ]] && unset TWF_ENV
[[ -z "$TWF_AGENT" ]] && unset TWF_AGENT

if [[ $make_default == 1 ]]; then
    [[ -n "$env" ]] || { echo "--default needs --env"; exit 1; }
    printf 'env = %s\n' "$env" > "$models/default"
    [[ -n "$agent" ]] && printf 'agent = %s\n' "$agent" >> "$models/default"
fi

case "$mode" in
    host) cd "$here" && exec cargo run --release ;;
    build) cd "$here/firmware" && exec cargo build --release "${features[@]}" ;;
    flash) cd "$here/firmware" && exec cargo run --release "${features[@]}" ;;
esac
