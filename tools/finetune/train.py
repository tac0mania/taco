"""Fine-tune distilgpt2 on the Route A training split (run on a CUDA machine, the RTX 3060).

    python train.py --data DATA.npz --out DIR [--steps 2446 | --budget-min 110] [--lr 2e-4]

DATA.npz (from `python -m finetune.pack`) holds `<system>.train` and `<system>.val` (uint16
token ids of the GPT-2 tokenizer) and `token_bytes` (the byte length of each token id).

A training sequence is a block of 1,024 tokens of one system. The model input is
<|endoftext|> and then the first 1,023 tokens, and the loss covers all 1,024 tokens. This is
the window of Taco (plan §7.7). The blocks of an epoch are a random permutation of all blocks.

With --steps N, the run has N steps (cosine schedule). Without it, the tool measures the time of
the first steps, then sets the step count so that the run ends in --budget-min minutes. It evaluates the val blocks every --eval-every steps and
keeps two checkpoints in --ckpt (default DIR/ckpt), and no more:

- `latest/`: the model, the optimiser, the scaler, the random state and the data position, every
  --save-every steps (overwritten in place; `COMPLETE` marks a whole checkpoint);
- `best/`: the model with the lowest val loss (safetensors, float32).

If `latest/` exists, the tool resumes from it: the same step plan, schedule and data order. Thus
a run that stops (a dropped session, --stop-after-min) continues on any CUDA machine. A resume
can change --micro and --accum if the product is the same (the same sequences in each step).
DIR/train.log, --ckpt/train.log and --ckpt/history.json record the run.

NOTE: this is a tool that runs before publication. It uses floats and a GPU. Nothing in
crates/ uses it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EOT = 50256
SENTINEL = 10**9  # the step plan before the first 20 steps measure the speed
N_CTX = 1024


LOG_FILES: list[Path] = []


def log(out: Path, msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    for p in [out / "train.log", *LOG_FILES]:
        with p.open("a") as f:
            f.write(line + "\n")


def blocks_of(a: np.ndarray, limit: int | None = None) -> np.ndarray:
    n = len(a) // N_CTX
    if limit is not None:
        n = min(n, limit)
    return a[: n * N_CTX].reshape(n, N_CTX)


def batch_tensors(blocks: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.from_numpy(blocks.astype(np.int64))
    x = torch.cat([torch.full((len(blocks), 1), EOT, dtype=torch.int64), y[:, :-1]], dim=1)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, val: dict[str, np.ndarray], tb: np.ndarray, device: str, bs: int) -> dict:
    """Bits per byte of the val blocks of each system, and of all of them."""
    model.eval()
    res = {}
    tot_bits = tot_bytes = 0.0
    for name, blk in val.items():
        bits = 0.0
        for i in range(0, len(blk), bs):
            x, y = batch_tensors(blk[i : i + bs], device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits = model(input_ids=x).logits
            nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                  reduction="sum")
            bits += float(nll) / math.log(2.0)
        nbytes = float(tb[blk].sum())
        res[name] = bits / nbytes
        tot_bits += bits
        tot_bytes += nbytes
    res["ALL"] = tot_bits / tot_bytes
    model.train()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--steps", type=int, default=0,
                    help="the number of steps (0: set it from --budget-min)")
    ap.add_argument("--budget-min", type=float, default=110.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--micro", type=int, default=8, help="sequences per micro batch")
    ap.add_argument("--accum", type=int, default=4, help="micro batches per step")
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=400, help="steps between checkpoints")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint directory (default OUT/ckpt): latest/ and best/ only")
    ap.add_argument("--stop-after-min", type=float, default=0.0,
                    help="save and stop this run after this many minutes (0: no limit)")
    ap.add_argument("--val-blocks", type=int, default=24, help="val blocks per system")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--device", default="cuda", help="cuda, or cpu for a smoke test only")
    ap.add_argument("--train-blocks", type=int, default=0,
                    help="use only this many train blocks of each system (smoke test only)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from transformers import GPT2LMHeadModel

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    cuda = device.startswith("cuda")
    z = np.load(args.data)
    tb = z["token_bytes"].astype(np.int64)
    systems = sorted(k[: -len(".train")] for k in z.files if k.endswith(".train"))
    train = {s: blocks_of(z[f"{s}.train"], args.train_blocks or None) for s in systems}
    val = {s: blocks_of(z[f"{s}.val"], args.val_blocks) for s in systems}
    index = np.concatenate([np.stack([np.full(len(b), i), np.arange(len(b))], axis=1)
                            for i, b in enumerate(train.values())])
    blist = list(train.values())
    log(args.out, f"systems {systems}; train blocks {len(index):,} "
                  f"({len(index) * N_CTX / 1e6:.1f} M tokens); val blocks "
                  f"{sum(len(v) for v in val.values())}; device "
                  f"{torch.cuda.get_device_name() if cuda else device}; torch {torch.__version__}")

    ckpt = args.ckpt or args.out / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    LOG_FILES.append(ckpt / "train.log")
    latest = ckpt / "latest"
    resume = False
    if (latest / "COMPLETE").is_file():
        done = json.loads((latest / "COMPLETE").read_text())
        resume = all((latest / n).is_file() and (latest / n).stat().st_size == size
                     for n, size in done["sizes"].items())
        if not resume:
            log(args.out, f"{latest} is not complete: start again from {args.model}")
    src = str(latest) if resume else args.model
    model = GPT2LMHeadModel.from_pretrained(src, attn_implementation="sdpa").to(device)
    model.train()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    plan = {"total": args.steps or SENTINEL}

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = min(1.0, (step - args.warmup) / max(1, plan["total"] - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    if resume:
        st = torch.load(latest / "train_state.pt", map_location=device, weights_only=False)
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"].cpu())
        plan["total"], step, perm, pos, epoch = (st["total"], st["step"], st["perm"], st["pos"],
                                                 st["epoch"])
        best, history, t_train = st["best"], st["history"], st["t_train"]
        if st["micro"] * st["accum"] != args.micro * args.accum:
            raise SystemExit(f"resume needs {st['micro'] * st['accum']} sequences per step")
        if (st["micro"], st["accum"]) != (args.micro, args.accum):
            log(args.out, f"NOTE: micro x accum was {st['micro']} x {st['accum']}; now "
                          f"{args.micro} x {args.accum} (same sequences per step: "
                          f"{st['micro'] * st['accum'] == args.micro * args.accum})")
        log(args.out, f"resumed from {latest} at step {step}/{plan['total']}, best {best:.4f}")
    else:
        history = []
        r0 = evaluate(model, val, tb, device, args.micro)
        history.append({"step": 0, "val": r0})
        log(args.out, f"step 0 val bpb {r0['ALL']:.4f} " +
            " ".join(f"{k}={v:.3f}" for k, v in r0.items() if k != "ALL"))
        best = r0["ALL"]
        perm = rng.permutation(len(index))
        pos = epoch = step = 0
        t_train = 0.0

    def save_dir(target: Path, write) -> None:
        """Write the files into a local stage directory, then copy them over the files of
        `target`, and then write `target/COMPLETE` (the step and the file sizes).

        NOTE: the copy overwrites the files in place and deletes nothing large. On Google Drive,
        a deleted file goes to the trash and still uses the quota; an overwrite does not make a
        trash item. A termination during the copy leaves `target` without COMPLETE, and the
        resume refuses it.
        """
        stage = args.out / "stage" / target.name
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        write(stage)
        target.mkdir(parents=True, exist_ok=True)
        (target / "COMPLETE").unlink(missing_ok=True)
        sizes = {}
        for f in sorted(stage.iterdir()):
            shutil.copyfile(f, target / f.name)
            sizes[f.name] = f.stat().st_size
        (target / "COMPLETE").write_text(json.dumps({"step": step, "sizes": sizes}))
        shutil.rmtree(stage, ignore_errors=True)

    def save_latest() -> None:
        def write(d: Path) -> None:
            model.save_pretrained(d)
            torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "total": plan["total"], "step": step, "perm": perm, "pos": pos,
                        "epoch": epoch, "best": best, "history": history, "t_train": t_train,
                        "micro": args.micro, "accum": args.accum}, d / "train_state.pt")
        t = time.time()
        save_dir(latest, write)
        log(args.out, f"step {step}: saved {latest} ({time.time() - t:.0f} s)")

    t_start = time.time()
    t_run0, step_run0 = t_train, step
    budget = args.budget_min * 60.0
    while step < plan["total"]:
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_sum = 0.0
        for _ in range(args.accum):
            if pos + args.micro > len(perm):
                perm, pos, epoch = rng.permutation(len(index)), 0, epoch + 1
            sel = index[perm[pos : pos + args.micro]]
            pos += args.micro
            blk = np.stack([blist[i][j] for i, j in sel])
            x, y = batch_tensors(blk, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                logits = model(input_ids=x).logits
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
            scaler.scale(loss / args.accum).backward()
            loss_sum += loss.item() / args.accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        if cuda:
            torch.cuda.synchronize()
        t_train += time.time() - ts
        if step - step_run0 == 20:
            per = (t_train - t_run0) / (step - step_run0)
            if plan["total"] == SENTINEL:
                # The eval passes and the checkpoints cost about 5 % of the budget (est.).
                plan["total"] = max(args.warmup + 100, int(budget * 0.95 / per))
            tok_s = args.micro * args.accum * N_CTX / per
            log(args.out, f"{per:.2f} s/step, {tok_s:,.0f} tokens/s; plan {plan['total']} "
                          f"steps = {plan['total'] * args.micro * args.accum * N_CTX / 1e6:.0f} "
                          f"M tokens ({plan['total'] * args.micro * args.accum / len(index):.2f}"
                          f" epochs)")
        if step % 50 == 0:
            el = time.time() - t_start
            per = (t_train - t_run0) / max(1, el)  # train share of the wall time
            eta = (plan["total"] - step) * (t_train / step) / 60
            log(args.out, f"step {step}/{plan['total']} loss {loss_sum / math.log(2):.3f} "
                          f"bits/token lr {lr_at(step):.2e} epoch {epoch} "
                          f"run {el / 60:.1f} min eta {eta:.1f} min (train {100 * per:.0f} %)" +
                (f" gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if cuda else ""))
        if step % args.eval_every == 0 or step == plan["total"]:
            r = evaluate(model, val, tb, device, args.micro)
            history.append({"step": step, "val": r})
            mark = ""
            if r["ALL"] < best:
                best = r["ALL"]
                save_dir(ckpt / "best", lambda d: model.save_pretrained(d))
                mark = " (best, saved)"
            log(args.out, f"step {step} val bpb {r['ALL']:.4f}{mark} " +
                " ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "ALL"))
            (ckpt / "history.json").write_text(json.dumps(
                {"args": {k: str(v) for k, v in vars(args).items()}, "history": history,
                 "best_val_bpb": best, "total_steps": plan["total"]}, indent=1))
        if step % args.save_every == 0 and step < plan["total"]:
            save_latest()
        if args.stop_after_min and time.time() - t_start > args.stop_after_min * 60:
            save_latest()
            log(args.out, f"stop: --stop-after-min {args.stop_after_min} reached at step {step}")
            return 3
    save_latest()
    log(args.out, f"done: {step} steps, {t_train / 60:.1f} min of training, "
                  f"best val bpb {best:.4f}")
    (ckpt / "DONE").write_text("done\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
