import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd

from grokking_common import (
    DEVICE,
    TrainConfig,
    ensure_dir,
    find_first_step,
    make_dataset,
    plot_phase_diagram,
    save_json,
    train_run,
)

DEFAULT_OUT = "outputs/ablation"


def _run_single(job):
    """Worker function: each process imports torch and creates its own CUDA context."""
    from grokking_common import TrainConfig, make_dataset, train_run, find_first_step

    frac, wd, seed, steps, lr, eval_every, grad_clip = (
        job["frac"], job["wd"], job["seed"], job["steps"],
        job["lr"], job["eval_every"], job["grad_clip"],
    )

    ds = make_dataset(97, frac, seed=seed)
    cfg = TrainConfig(
        train_frac=frac,
        data_seed=seed,
        model_seed=seed,
        n_steps=steps,
        lr=lr,
        weight_decay=wd,
        eval_every=eval_every,
        probe_every=steps + 1,
        grad_clip=grad_clip,
    )

    _, metrics, _, _ = train_run(ds, cfg, collect_fourier=False, verbose=False)

    s = np.array(metrics["step"])
    tr = np.array(metrics["train_acc"])
    te = np.array(metrics["test_acc"])

    t99 = find_first_step(s, tr, 0.99)
    t95 = find_first_step(s, te, 0.95)

    if t99 is None:
        delay, status = -1, "no_memorization"
    elif t95 is None:
        delay, status = steps, "no_generalization"
    else:
        delay = t95 - t99
        status = "delayed_generalization" if delay >= 500 else "immediate_generalization"

    return {
        "train_frac": frac,
        "weight_decay": wd,
        "seed": seed,
        "t_train_99": t99,
        "t_test_95": t95,
        "tau_g": delay,
        "final_train_acc": float(tr[-1]),
        "final_test_acc": float(te[-1]),
        "status": status,
    }


def summarize_run(metrics, max_steps):
    s = np.array(metrics["step"])
    tr = np.array(metrics["train_acc"])
    te = np.array(metrics["test_acc"])

    t99 = find_first_step(s, tr, 0.99)
    t95 = find_first_step(s, te, 0.95)

    if t99 is None:
        delay = -1
        status = "no_memorization"
    elif t95 is None:
        delay = max_steps
        status = "no_generalization"
    else:
        delay = t95 - t99
        status = "delayed_generalization" if delay >= 500 else "immediate_generalization"

    return {
        "t_train_99": t99,
        "t_test_95": t95,
        "tau_g": delay,
        "final_train_acc": float(tr[-1]),
        "final_test_acc": float(te[-1]),
        "status": status,
    }


def majority_vote(series):
    return series.value_counts().idxmax()


def aggregate_and_plot(per_seed_df, fracs, wds, steps, out_dir):
    grouped = (
        per_seed_df.groupby(["train_frac", "weight_decay"], as_index=False)
        .agg(
            mean_tau_g=("tau_g", "mean"),
            std_tau_g=("tau_g", "std"),
            mean_final_train_acc=("final_train_acc", "mean"),
            mean_final_test_acc=("final_test_acc", "mean"),
            grokking_successes=("status", lambda x: int((x == "delayed_generalization").sum())),
            immediate_successes=("status", lambda x: int((x == "immediate_generalization").sum())),
            no_generalization_count=("status", lambda x: int((x == "no_generalization").sum())),
            no_memorization_count=("status", lambda x: int((x == "no_memorization").sum())),
            n_runs=("seed", "count"),
        )
    )

    status_df = (
        per_seed_df.groupby(["train_frac", "weight_decay"])["status"]
        .agg(majority_vote)
        .reset_index(name="majority_status")
    )
    grouped = grouped.merge(status_df, on=["train_frac", "weight_decay"], how="left")

    grouped_path = os.path.join(out_dir, "ablation_results_grouped.csv")
    grouped.to_csv(grouped_path, index=False)

    plot_phase_diagram(grouped, fracs, wds, steps, out_dir)
    return grouped_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--fracs", type=float, nargs="+", default=[0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--wds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (processes). "
                             "Each worker runs one (frac, wd, seed) config at a time.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    print(f"Device: {DEVICE}")
    print(f"Workers: {args.workers}")

    jobs = []
    for wd in args.wds:
        for frac in args.fracs:
            for seed in args.seeds:
                jobs.append({
                    "frac": frac, "wd": wd, "seed": seed,
                    "steps": args.steps, "lr": args.lr,
                    "eval_every": args.eval_every, "grad_clip": args.grad_clip,
                })

    total = len(jobs)
    print(f"Total runs: {total}  ({len(args.fracs)} fracs x {len(args.wds)} wds x {len(args.seeds)} seeds)")

    per_seed_rows = []
    csv_path = os.path.join(args.out_dir, "ablation_results_per_seed.csv")
    t0 = time.time()

    def _save_incremental():
        pd.DataFrame(per_seed_rows).to_csv(csv_path, index=False)

    if args.workers <= 1:
        for i, job in enumerate(jobs):
            print(f"\n[{i+1}/{total}] frac={job['frac']}, wd={job['wd']}, seed={job['seed']}")

            ds = make_dataset(97, job["frac"], seed=job["seed"])
            cfg = TrainConfig(
                train_frac=job["frac"],
                data_seed=job["seed"],
                model_seed=job["seed"],
                n_steps=args.steps,
                lr=args.lr,
                weight_decay=job["wd"],
                eval_every=args.eval_every,
                probe_every=args.steps + 1,
                grad_clip=args.grad_clip,
            )
            _, metrics, _, _ = train_run(ds, cfg, collect_fourier=False, verbose=False)
            summary = summarize_run(metrics, args.steps)

            row = {"train_frac": job["frac"], "weight_decay": job["wd"],
                   "seed": job["seed"], **summary}
            per_seed_rows.append(row)
            _save_incremental()

            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (total - i - 1)
            print(f"  -> {row['status']}  tau_g={row['tau_g']}  "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            future_to_job = {pool.submit(_run_single, job): job for job in jobs}
            done_count = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                done_count += 1
                try:
                    row = future.result()
                    per_seed_rows.append(row)
                    _save_incremental()
                    elapsed = time.time() - t0
                    remaining = elapsed / done_count * (total - done_count)
                    print(f"[{done_count}/{total}] frac={row['train_frac']}, "
                          f"wd={row['weight_decay']}, seed={row['seed']} -> "
                          f"{row['status']}  tau_g={row['tau_g']}  "
                          f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")
                except Exception as e:
                    print(f"[{done_count}/{total}] FAILED frac={job['frac']}, "
                          f"wd={job['wd']}, seed={job['seed']}: {e}")

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(csv_path, index=False)

    grouped_path = aggregate_and_plot(per_seed_df, args.fracs, args.wds, args.steps, args.out_dir)

    save_json(
        {
            "steps": args.steps,
            "fracs": args.fracs,
            "wds": args.wds,
            "seeds": args.seeds,
            "lr": args.lr,
            "eval_every": args.eval_every,
            "grad_clip": args.grad_clip,
            "workers": args.workers,
        },
        os.path.join(args.out_dir, "ablation_config.json"),
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("Per-seed CSV:", csv_path)
    print("Grouped CSV:", grouped_path)
    print("Phase diagram:", os.path.join(args.out_dir, "phase_diagram.png"))


if __name__ == "__main__":
    main()
