#!/usr/bin/env python3
"""Patch zero-holes in corrupted FL2VA shards with ModelScope range fetches.

Holes were found by scanning for >=4MB all-zero runs (aria2 prealloc + lost
control file). Each hole is padded +/-4MB, merged, refetched via HTTP range,
and written back in place. Post-checks: zero-rescan must be empty and random
spot ranges must match remote byte-for-byte.
"""
import json, os, random, subprocess, struct, sys, time
import numpy as np

REPO = os.path.expanduser("~/llm-lab/src/minimax-h3-mlx-rebuild")
SRC  = f"{REPO}/official/FL2VA"
MS   = ("https://modelscope.cn/api/v1/models/MiniMax/MiniMax-H3/repo"
        "?Revision=master&FilePath=FL2VA/")
PAD  = 4 * 2**20
random.seed(7)

def used_shards(part, mapfn):
    hdr = json.load(open(f"{REPO}/reference/ref_{part}.json"))
    hdr.pop("__metadata__", None)
    targets = [k for k in hdr if not k.endswith((".scales", ".biases"))]
    index = json.load(open(f"{SRC}/{part}/model.safetensors.index.json"))["weight_map"]
    return {index[mapfn(t)] for t in targets}

def te_map(k):
    if k.startswith("visual."):
        return "model." + k
    if k.startswith("model."):
        return k.replace("model.", "model.language_model.", 1)
    return k

used = {"text_encoder": used_shards("text_encoder", te_map),
        "transformer":  used_shards("transformer", lambda k: k)}
print("TE shards used:", sorted(used["text_encoder"]), flush=True)
print("TF shards used:", sorted(used["transformer"]), flush=True)

holes = json.load(open("/tmp/h3-holes.json"))
todo = []
for path, runs in holes.items():
    part = "text_encoder" if "text_encoder" in path else "transformer"
    if os.path.basename(path) not in used[part]:
        print(f"SKIP {path} (unused by convert)", flush=True)
        continue
    todo.append((path, runs))

def fetch(url, a, b, out, tries=5):
    for t in range(tries):
        r = subprocess.run(["curl", "-sL", "--max-time", "1800",
                            "-r", f"{a}-{b}", "-o", out, url])
        ok = r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) == b - a + 1
        if ok:
            return True
        got = os.path.getsize(out) if os.path.exists(out) else 0
        print(f"    retry {t+1}/5 rc={r.returncode} got={got}", flush=True)
        time.sleep(5)
    return False

def zero_free(path, block=2**20, min_run=4):
    size = os.path.getsize(path)
    cur, cnt = None, 0
    with open(path, "rb") as f:
        off = 0
        while off < size:
            chunk = f.read(64 * block)
            if not chunk:
                break
            a = np.frombuffer(chunk, dtype=np.uint8)
            nb = len(a) // block
            zero = ~(a[: nb * block].reshape(nb, block)).any(axis=1)
            for i, z in enumerate(zero):
                b = off // block + i
                if z and cur is None:
                    cur = b
                elif not z and cur is not None:
                    if b - cur >= min_run:
                        cnt += 1
                    cur = None
            off += len(a)
    if cur is not None and size // block - cur >= min_run:
        cnt += 1
    return cnt == 0

total, t_start = 0, time.time()
for path, runs in todo:
    size = os.path.getsize(path)
    regs = sorted((max(0, s - PAD), min(size, e + PAD)) for s, e in runs)
    merged = []
    for s, e in regs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    url = MS + path.split("FL2VA/")[1]
    need = sum(e - s for s, e in merged)
    print(f"REPAIR {os.path.basename(path)}: {len(merged)} regions, {need/1e9:.2f} GB", flush=True)
    with open(path, "r+b") as f:
        for i, (s, e) in enumerate(merged):
            tmp = f"/tmp/h3-fix-{os.getpid()}.bin"
            t0 = time.time()
            if not fetch(url, s, e - 1, tmp):
                sys.exit(f"FATAL fetch failed {path} {s}-{e-1}")
            f.seek(s)
            with open(tmp, "rb") as g:
                while True:
                    b64 = g.read(64 * 2**20)
                    if not b64:
                        break
                    f.write(b64)
            f.flush()
            os.fsync(f.fileno())
            total += e - s
            el = time.time() - t0
            print(f"    {i+1}/{len(merged)} {s}-{e} ({(e-s)/1e6:.0f} MB) {el:.0f}s "
                  f"({(e-s)/1e6/max(el,1):.1f} MB/s)", flush=True)
    # post-checks: no zero holes left + 2 random spot ranges match remote
    if not zero_free(path):
        sys.exit(f"FATAL {path} still has zero holes after repair")
    for _ in range(2):
        s = random.randrange(0, size - 8 * 2**20)
        e = s + 8 * 2**20
        tmp = f"/tmp/h3-fix-{os.getpid()}.bin"
        if not fetch(url, s, e - 1, tmp):
            sys.exit(f"FATAL spot fetch failed {path}")
        with open(path, "rb") as f:
            f.seek(s)
            loc = f.read(e - s)
        if loc != open(tmp, "rb").read():
            sys.exit(f"FATAL spot check mismatch {path} at {s}")
    print(f"    checks ok: zero-free + 2 spot ranges match remote", flush=True)

el = time.time() - t_start
print(f"ALL REPAIRS DONE: wrote {total/1e9:.2f} GB in {el/60:.1f} min", flush=True)
