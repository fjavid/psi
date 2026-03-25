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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.30, 0.35, 0.40, 0.50])
    parser.add_argument("--wds", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    print(f"Device: {DEVICE}")

    rows = []
    phase = np.full((len(args.wds), len(args.fracs)), np.nan)

    for i, wd in enumerate(args.wds):
        for j, frac in enumerate(args.fracs):
            ds = make_dataset(97, frac, seed=args.seed)
            cfg = TrainConfig(
                train_frac=frac,
                data_seed=args.seed,
                model_seed=args.seed,
                n_steps=args.steps,
                lr=args.lr,
                weight_decay=wd,
                eval_every=max(50, args.steps // 20),
                probe_every=args.steps + 1,
                grad_clip=1.0,
            )
            _, metrics, _, _ = train_run(ds, cfg, collect_fourier=False, verbose=False)
            s = np.array(metrics["step"])
            tr = np.array(metrics["train_acc"])
            te = np.array(metrics["test_acc"])

            t99 = find_first_step(s, tr, 0.99)
            t95 = find_first_step(s, te, 0.95)
            if t99 is None:
                delay = -1
                status = "no_memorization"
            elif t95 is None:
                delay = args.steps
                status = "no_generalization"
            else:
                delay = t95 - t99
                status = "delayed_generalization" if delay >= 500 else "immediate_generalization"

            phase[i, j] = delay
            row = {
                "train_frac": frac,
                "weight_decay": wd,
                "t_train_99": t99,
                "t_test_95": t95,
                "tau_g": delay,
                "final_train_acc": float(tr[-1]),
                "final_test_acc": float(te[-1]),
                "status": status,
            }
            rows.append(row)
            print(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "ablation_results.csv"), index=False)
    plot_phase_diagram(phase, args.fracs, args.wds, args.steps, args.out_dir)
    save_json({"steps": args.steps, "fracs": args.fracs, "wds": args.wds}, os.path.join(args.out_dir, "ablation_config.json"))
    print("\nSaved outputs to", args.out_dir)


if __name__ == "__main__":
    main()