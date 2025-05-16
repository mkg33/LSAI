#!/usr/bin/env python3
import argparse, os, signal, sys, time

p = argparse.ArgumentParser()
p.add_argument("--max-iter", type=int, default=None)
p.add_argument("--resume",   type=str, default=None)
args = p.parse_args()

CKPT_DIR = os.environ["CKPT_DIR"]; os.makedirs(CKPT_DIR, exist_ok=True)

start = 0
if args.resume:
    with open(args.resume) as f: start = int(f.read()) + 1
    print(f"[compute] resumed at {start}", file=sys.stderr, flush=True)

stop = False
def usr1(*_):
    global stop; stop = True
    print("[compute] SIGUSR1 received", file=sys.stderr, flush=True)
signal.signal(signal.SIGUSR1, usr1)

for i in range(start, 10**9):
    _ = sum(j*j for j in range(10_000))
    if i % 100 == 0:
        print(f"[compute] iter={i}", file=sys.stderr, flush=True)

    if stop:
        ck = f"{CKPT_DIR}/ckpt_{i:08d}.txt"
        with open(ck, "w") as f: f.write(str(i)); f.flush(); os.fsync(f.fileno())
        sys.exit(0)              # wrapper chains

    if args.max_iter and i+1 >= args.max_iter:
        sys.exit(1)              # normal finish, so wrapper stops
