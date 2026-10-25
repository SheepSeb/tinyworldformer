#!/usr/bin/env bash
# Build the browser demo: one WebAssembly module per exported env, into www/.
#
#   ./build.sh                      every env, each with its game's default agent
#   ./build.sh flappy pong          only these
#   TWF_AGENT=doom_agent_roam_p0.3 ./build.sh doomlite
#   python3 -m http.server -d www   then open http://localhost:8000
#
# Needs the wasm target once: rustup target add wasm32-unknown-unknown
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
models="$here/../models"
envs=("$@")
[[ ${#envs[@]} -eq 0 ]] && envs=($(ls "$models/envs"))

cd "$here"
for env in "${envs[@]}"; do
    [[ -f "$models/envs/$env/weights.rs" ]] || { echo "no env '$env' (have: $(ls "$models/envs" | tr '\n' ' '))"; exit 1; }
    TWF_ENV="$env" cargo build --release
    cp target/wasm32-unknown-unknown/release/tinyworldformer_web.wasm "www/$env.wasm"
done
ls www/*.wasm | xargs -n1 basename | sed 's/\.wasm$//' | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin]))' > www/envs.json
echo "built: $(cat www/envs.json)"
