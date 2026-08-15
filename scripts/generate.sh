#!/bin/bash
# MiniMax-H3 本地生成脚本(MLX 4-bit, 基于 uetuluk2 重建流程)
# 前置: download.py 完成 + convert.py + verify.py 通过
# 用法: bash generate.sh ["提示词"] [输出文件]
# 默认: 5 秒 832x480 预览, turbo 4 步(实测 M5 Pro/48G 约 247s; M4 Pro 预计 7-12 分钟)
set -eu

PROMPT="${1:-a blacksmith hammering a glowing horseshoe, sparks flying, forge firelight}"
OUT="${2:-$HOME/llm-lab/outputs/h3-test.mp4}"

mkdir -p "$HOME/llm-lab/outputs"

# 1. 启动 mlx-serve(如未运行)
if ! curl -s http://127.0.0.1:11234/v1/models >/dev/null 2>&1; then
  echo "starting mlx-serve ..."
  nohup mlx-serve --serve --model-dir "$HOME/.mlx-serve/models" \
    > "$HOME/llm-lab/scripts/mlx-serve.log" 2>&1 &
  sleep 10
fi

# 2. 生成(must use nohup: 前台长时间任务在后台终端会被静默杀掉)
cd "$HOME/llm-lab/src/minimax-h3-mlx-rebuild"
nohup ~/llm-lab/.venv-h3/bin/python generate.py "$PROMPT" \
  --seconds 5 --width 832 --height 480 --steps 4 --turbo \
  -o "$OUT" > "$HOME/llm-lab/scripts/h3-generate.log" 2>&1 &

echo "generation started -> $OUT (log: ~/llm-lab/scripts/h3-generate.log)"
echo "watch: tail -f ~/llm-lab/scripts/h3-generate.log"
