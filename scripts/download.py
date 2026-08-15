#!/usr/bin/env python3
# 从 ModelScope CDN 用 aria2 并行下载 MiniMax-H3 FL2VA 权重(已验证)
# 用法: ~/llm-lab/.venv-h3/bin/python download.py
#   (venv 需 pip install requests, 见部署指南 §3)
# 背景: modelscope CLI 只有 2-7 MB/s; CDN 单文件 8 连接实测最高 ~57 MB/s,
#       但持续吞吐被限到每文件 ~1.5-1.8 MB/s,所以用 3 文件并行池拉聚合带宽
#       (实测 3 并发最优; 6 并发反而被压到每文件 ~0.5MB/s)
# 依赖: requests, aria2(brew install aria2)
# 输出: ~/llm-lab/src/minimax-h3-mlx-rebuild/official/FL2VA/
#       (自动跳过 convert 不用的 text_encoder 00012/00013 分片
#        ——只装第 53-63 层, 立省 ~10GB: 80 个文件, ~134GB; 全量 144GB)
# 断点续传: 已完成文件跳过; 部分文件靠 aria2 --continue 从 .aria2 控制文件续传
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.expanduser("~/llm-lab/src/minimax-h3-mlx-rebuild/official")
API = ("https://modelscope.cn/api/v1/models/MiniMax/MiniMax-H3/"
       "repo/files?Revision=master&Recursive=true")
CDN = ("https://modelscope.cn/api/v1/models/MiniMax/MiniMax-H3/"
       "repo?Revision=master&FilePath=")
WORKERS = 3  # 实测: 4 并行时每文件 ~1.5MB/s; 6 并行反而被压到 ~0.5MB/s
T0 = time.time()
STATS = {"done": 0, "skip": 0, "bytes": 0}


def fetch_file(f):
    dst = os.path.join(ROOT, f["Path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == f["Size"]:
        STATS["skip"] += 1
        STATS["bytes"] += f["Size"]
        return f"skip   {f['Path']}"
    # modelscope CLI 的 .incomplete 无法被 aria2 续传,删除重下
    inc = dst + ".incomplete"
    if os.path.exists(inc):
        os.remove(inc)
    url = CDN + requests.utils.quote(f["Path"], safe="/")
    cmd = ["aria2c", "-x8", "-s8", "-k1M", "--file-allocation=none",
           "--continue=true", "--auto-file-renaming=false",
           "--console-log-level=warn", "--summary-interval=10",
           "-d", os.path.dirname(dst), "-o", os.path.basename(dst), url]
    subprocess.run(cmd, check=True)
    sz = os.path.getsize(dst)
    if sz != f["Size"]:
        print(f"FAILED size mismatch {f['Path']}: {sz} != {f['Size']}", flush=True)
        os._exit(1)
    STATS["done"] += 1
    STATS["bytes"] += sz
    el = time.time() - T0
    return (f"[{STATS['done']}/{len(TODO)}] {f['Path']}  {sz / 1e9:.2f}GB  "
            f"(pool {STATS['bytes'] / 1e9:.1f}/{TOTAL / 1e9:.1f}GB, "
            f"{STATS['bytes'] / 1e6 / el:.1f}MB/s avg)")


def main():
    global TODO, TOTAL
    files = requests.get(API, timeout=30).json()["Data"]["Files"]
    # H3 只加载文本编码器的语言层 0-49, 第 50-63 层用不到;
    # 其中 53-63 层整装在 TE 00012/00013 两个分片, 跳过立省 ~10GB
    SKIP = {"FL2VA/text_encoder/model-00012-of-00014.safetensors",
            "FL2VA/text_encoder/model-00013-of-00014.safetensors"}
    TODO = [f for f in files
            if f["Path"].startswith("FL2VA/") and f["Path"] not in SKIP]
    TODO.sort(key=lambda f: -f["Size"])
    TOTAL = sum(f["Size"] for f in TODO)
    print(f"{len(TODO)} files, {TOTAL / 1e9:.1f}GB, {WORKERS} parallel workers",
          flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for msg in ex.map(fetch_file, TODO):
            print(msg, flush=True)
    print(f"ALL DONE: {STATS['done']} downloaded, {STATS['skip']} skipped, "
          f"{TOTAL / 1e9:.1f}GB total", flush=True)


if __name__ == "__main__":
    main()
