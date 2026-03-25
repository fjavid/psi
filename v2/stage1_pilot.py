import argparse
import os

import pandas as pd

from grokking_common import (
    DEVICE,
    TrainConfig,
    ensure_dir,
    make_dataset,
    pilot_score,
    plot_pilot_heatmaps,
    save_json,
    summarize_pilot_history,
    train_run,
)

DEFAULT_OUT = "outputs/pilot"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--train_fracs", type=float, nargs="+", default=[0.25, 0.30, 0.35, 0.40, 0.50])
    parser.add_argument("--weight_decays", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    rows = []

    print(f"Device: {DEVICE}")
    print("Running pilot sweep...")
    for frac in args.train_fracs:
        for wd in args.weight_decays:
            for seed in args.seeds:
                ds = make_dataset(97, frac, seed=1000 + seed)
                cfg = TrainConfig(
                    train_frac=frac,
                    data_seed=1000 + seed,
                    model_seed=seed,
                    n_steps=args.steps,
                    lr=args.lr,
                    weight_decay=wd,
                    eval_every=max(50, args.steps // 10),
                    probe_every=args.steps + 1,
                    grad_clip=1.0,
                )
                _, metrics, _, _ = train_run(ds, cfg, collect_fourier=False, verbose=False)
                summary = summarize_pilot_history(metrics)
                row = {
                    "train_frac": frac,
                    "weight_decay": wd,
                    "seed": seed,
                    **summary,
                }
                row["score"] = pilot_score(row)
                rows.append(row)
                print(
                    f"frac={frac:.2f} wd={wd:.3f} seed={seed} "
                    f"train_end={row['train_acc_end']:.3f} test_end={row['test_acc_end']:.3f} "
                    f"score={row['score']:.3f}"
                )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "pilot_results.csv")
    df.to_csv(csv_path, index=False)

    grouped = (
        df.groupby(["train_frac", "weight_decay"], as_index=False)[
            ["train_acc_end", "test_acc_end", "train_acc_mid", "test_acc_mid", "score"]
        ]
        .mean()
        .sort_values("score", ascending=False)
    )
    grouped.to_csv(os.path.join(args.out_dir, "pilot_grouped.csv"), index=False)

    best = grouped.iloc[0].to_dict()
    selected = {
        "train_frac": float(best["train_frac"]),
        "weight_decay": float(best["weight_decay"]),
        "pilot_steps": int(args.steps),
        "selection_note": "Chosen by pilot score: high train acc, nontrivial but not immediate held-out acc, positive held-out trend.",
    }
    save_json(selected, os.path.join(args.out_dir, "selected_config.json"))

    plot_pilot_heatmaps(grouped, args.train_fracs, args.weight_decays, args.out_dir)

    print("\nTop pilot configs:")
    print(grouped.head(10).to_string(index=False))
    print(f"\nSaved:\n- {csv_path}\n- {os.path.join(args.out_dir, 'pilot_grouped.csv')}\n- {os.path.join(args.out_dir, 'selected_config.json')}")
    print(f"Selected config: frac={selected['train_frac']}, wd={selected['weight_decay']}")


if __name__ == "__main__":
    main()