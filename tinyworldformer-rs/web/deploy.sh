#!/usr/bin/env bash
# Publish the browser demo as a static Hugging Face Space. Everything runs in the visitor's
# browser (WebAssembly on the CPU), so the free cpu-basic Space with no server is enough.
#
#   ./deploy.sh <user>/<space>      build every env, create the Space if needed, upload
#   ./deploy.sh <user>/<space> --no-build   upload what is already in www/
#
# Needs a Hugging Face login once: uvx --from huggingface_hub hf auth login
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
space="${1:?usage: ./deploy.sh <user>/<space> [--no-build]}"
hf() { uvx --from huggingface_hub hf "$@"; }

cd "$here"
[[ "${2:-}" == "--no-build" ]] || ./build.sh
ls www/*.wasm >/dev/null 2>&1 || { echo "nothing built in www/: run ./build.sh"; exit 1; }

# The Space is www/ plus the README whose front matter tells Hugging Face it is a static site.
stage="$here/target/space"
rm -rf "$stage" && mkdir -p "$stage" && cp www/index.html www/envs.json www/*.wasm "$stage/"
cat > "$stage/README.md" <<'EOF'
---
title: tinyworldformer
emoji: 🎮
colorFrom: indigo
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
short_description: Games dreamed by a tiny int8 transformer, in your browser
---

Five games (Breakout, Doom-lite, Flappy Bird, Pac-Man lite, Pong) with no game code: every frame
is dreamed by the int8 looped transformer that runs on a Raspberry Pi Pico 2 W, compiled to
WebAssembly and run on your CPU. A stream-x agent plays; press any game key to take over.
EOF

hf repos create "$space" --type space --space-sdk static --exist-ok
hf upload "$space" "$stage" . --repo-type space --delete "*.wasm" --commit-message "Update the dreamed games"
echo "live at https://huggingface.co/spaces/$space"
