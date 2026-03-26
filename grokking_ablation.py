import argparse
import os
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
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    print(f"Device: {DEVICE}")

    per_seed_rows = []

    for wd in args.wds:
        for frac in args.fracs:
            for seed in args.seeds:
                print(f"\nRunning frac={frac}, wd={wd}, seed={seed}")

                ds = make_dataset(97, frac, seed=seed)

                cfg = TrainConfig(
                    train_frac=frac,
                    data_seed=seed,
                    model_seed=seed,
                    n_steps=args.steps,
                    lr=args.lr,
                    weight_decay=wd,
                    eval_every=args.eval_every,
                    probe_every=args.steps + 1,
                    grad_clip=args.grad_clip,
                )

                _, metrics, _, _ = train_run(
                    ds,
                    cfg,
                    collect_fourier=False,
                    verbose=False,
                )

                summary = summarize_run(metrics, args.steps)

                row = {
                    "train_frac": frac,
                    "weight_decay": wd,
                    "seed": seed,
                    **summary,
                }
                per_seed_rows.append(row)
                print(row)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(args.out_dir, "ablation_results_per_seed.csv")
    per_seed_df.to_csv(per_seed_path, index=False)

    # Aggregate across seeds
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

    grouped_path = os.path.join(args.out_dir, "ablation_results_grouped.csv")
    grouped.to_csv(grouped_path, index=False)

    # Plot using grouped results
    plot_phase_diagram(grouped, args.fracs, args.wds, args.steps, args.out_dir)

    save_json(
        {
            "steps": args.steps,
            "fracs": args.fracs,
            "wds": args.wds,
            "seeds": args.seeds,
            "lr": args.lr,
            "eval_every": args.eval_every,
            "grad_clip": args.grad_clip,
        },
        os.path.join(args.out_dir, "ablation_config.json"),
    )

    print("\nSaved outputs to", args.out_dir)
    print("Per-seed CSV:", per_seed_path)
    print("Grouped CSV:", grouped_path)
    print("Phase diagram:", os.path.join(args.out_dir, "phase_diagram.png"))


if __name__ == "__main__":
    main()