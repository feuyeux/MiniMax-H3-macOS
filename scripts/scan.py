#!/usr/bin/env python3
"""Fine-grained hole scan + patch for all convert-used FL2VA files.

Coarse scan (4MB min run) missed a 0.8MB hole in TE shard 00003. This pass
uses 64KB zero blocks. Candidate regions are fetched from ModelScope:
  - remote also zero  -> genuine data (e.g. TF-00013 adaln zeros), skip
  - remote differs    -> real hole: write remote bytes (with 1MB pad), recount
Non-zero garbage is not expected from the aria2 prealloc failure mode; the
post-patch pack byte-verify + functional generation test are the backstop.
"""
import json, os, subprocess, sys, time
import numpy as np

REPO = os.path.expanduser("~/llm-lab/src/minimax-h3-mlx-rebuild")
SRC  = f"{REPO}/official/FL2VA"
MS   = ("https://modelscope.cn/api/v1/models/MiniMax/MiniMax-H3/repo"
        "?Revision=master&FilePath=FL2VA/")
BLK, PAD = 2**16, 2**20
# fully byte-verified legit-zero region: (file, lo, hi) — skip fetching
LEGIT = [("transformer/model-00013-of-00013.safetensors", 2893520024, 3413744792)]

files = [f"text_encoder/model-{i:05d}-of-00014.safetensors" for i in
         list(range(1, 12)) + [14]] \
      + [f"transformer/model-{i:05d}-of-00013.safetensors" for i in range(1, 14)] \
      + ["video_vae/source/model.safetensors", "audio_vae/model.safetensors"]

def zero_runs(path):
    size = os.path.getsize(path)
    runs, cur = [], None
    with open(path, "rb") as f:
        off = 0
        while off < size:
            chunk = f.read(1024 * BLK)          # 64MB
            if not chunk:
                break
            a = np.frombuffer(chunk, dtype=np.uint8)
            nb = len(a) // BLK
            z = ~(a[: nb * BLK].reshape(nb, BLK)).any(axis=1)
            for i, zz in enumerate(z):
                b = off // BLK + i
                if zz and cur is None:
                    cur = b
                elif not zz and cur is not None:
                    runs.append((cur * BLK, b * BLK)); cur = None
            off += len(a)
    if cur is not None:
        runs.append((cur * BLK, (size // BLK + 1) * BLK))
    return [(s, min(e, size)) for s, e in runs]

def fetch(url, a, b, tries=4):
    for t in range(tries):
        r = subprocess.run(["curl", "-sL", "--max-time", "900",
                            "-r", f"{a}-{b}", url], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == b - a + 1:
            return r.stdout
        print(f"    retry {t+1} rc={r.returncode} got={len(r.stdout)}", flush=True)
        time.sleep(4)
    sys.exit(f"FATAL fetch {url} {a}-{b}")

tot_holes = tot_legit = tot_patched = 0
for rel in files:
    path = f"{SRC}/{rel}"
    if not os.path.exists(path):
        print(f"MISSING {rel}", flush=True); continue
    runs = zero_runs(path)
    if not runs:
        print(f"ok     {rel}", flush=True); continue
    url = MS + rel
    size = os.path.getsize(path)
    print(f"SCAN   {rel}: {len(runs)} candidate zero runs", flush=True)
    with open(path, "r+b") as f:
        for s, e in runs:
            if any(rel == lf and s >= lo and e <= hi for lf, lo, hi in LEGIT):
                tot_legit += 1
                print(f"    legit zeros [{s},{e}) skipped (verified region)", flush=True)
                continue
            ps, pe = max(0, s - PAD), min(size, e + PAD)
            rem = fetch(url, ps, pe - 1)
            f.seek(ps); loc = f.read(pe - ps)
            if loc == rem:
                tot_legit += 1
                print(f"    legit zeros [{s},{e}) match remote", flush=True)
                continue
            f.seek(ps); f.write(rem); f.flush(); os.fsync(f.fileno())
            tot_holes += 1; tot_patched += pe - ps
            print(f"    PATCHED hole [{s},{e}) ({(e-s)/1e6:.2f} MB, wrote {(pe-ps)/1e6:.2f} MB)", flush=True)

print(f"DONE: {tot_holes} real holes patched ({tot_patched/1e6:.1f} MB written), "
      f"{tot_legit} legit zero regions", flush=True)
