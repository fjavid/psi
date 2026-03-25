import argparse
import os
import numpy as np

from grokking_common import (
    DEVICE,
    TrainConfig,
    classify_snapshot,
    ensure_dir,
    find_first_step,
    load_json,
    make_dataset,
    metrics_to_csv,
    plot_fourier,
    plot_norms,
    plot_operation_tables,
    plot_pca,
    plot_training_curves,
    save_json,
    train_run,
)

DEFAULT_PILOT = "outputs/pilot/selected_config.json"
DEFAULT_OUT = "outputs/main"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_config", type=str, default=DEFAULT_PILOT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    selected = None
    if os.path.exists(args.pilot_config):
        selected = load_json(args.pilot_config)

    train_frac = args.train_frac if args.train_frac is not None else (selected["train_frac"] if selected else 0.35)
    weight_decay = args.weight_decay if args.weight_decay is not None else (selected["weight_decay"] if selected else 0.1)

    print(f"Device: {DEVICE}")
    print(f"Main run config: frac={train_frac}, wd={weight_decay}, steps={args.steps}, lr={args.lr}")

    ds = make_dataset(97, train_frac, seed=args.seed)
    checkpoint_steps = sorted(
        set(np.unique(np.geomspace(1, args.steps, 36).astype(int)).tolist() + [100, 500, 1000, 2000, 5000, 10000, 20000, args.steps])
    )
    cfg = TrainConfig(
        train_frac=train_frac,
        data_seed=args.seed,
        model_seed=args.seed,
        n_steps=args.steps,
        lr=args.lr,
        weight_decay=weight_decay,
        eval_every=100,
        probe_every=250,
        checkpoint_steps=checkpoint_steps,
        grad_clip=1.0,
    )

    _, metrics, probe_metrics, checkpoints = train_run(ds, cfg, collect_fourier=True, verbose=True)

    metrics_to_csv(metrics, os.path.join(args.out_dir, "metrics.csv"))
    metrics_to_csv(
        {
            "step": probe_metrics["step"],
            "emb_eff_rank": probe_metrics["emb_eff_rank"],
            "fourier_structured_ratio": probe_metrics["fourier_structured_ratio"],
            "fourier_entropy": probe_metrics["fourier_entropy"],
        },
        os.path.join(args.out_dir, "probe_metrics.csv"),
    )

    steps = np.array(metrics["step"])
    train_acc = np.array(metrics["train_acc"])
    test_acc = np.array(metrics["test_acc"])
    t_train_99 = find_first_step(steps, train_acc, 0.99)
    t_test_95 = find_first_step(steps, test_acc, 0.95)
    tau_g = (t_test_95 - t_train_99) if (t_train_99 is not None and t_test_95 is not None) else None

    summary = {
        "train_frac": train_frac,
        "weight_decay": weight_decay,
        "n_steps": args.steps,
        "lr": args.lr,
        "t_train_99": t_train_99,
        "t_test_95": t_test_95,
        "tau_g": tau_g,
        "final_train_acc": float(train_acc[-1]),
        "final_test_acc": float(test_acc[-1]),
        "status": "grokking_observed" if t_test_95 is not None else ("memorized_not_generalized" if t_train_99 is not None else "not_memorized"),
    }
    save_json(summary, os.path.join(args.out_dir, "summary.json"))

    plot_training_curves(metrics, t_train_99, t_test_95, args.out_dir, title_suffix=f"(frac={train_frac}, wd={weight_decay})")
    plot_norms(metrics, probe_metrics, t_train_99, t_test_95, args.out_dir)
    plot_fourier(metrics, probe_metrics, t_train_99, t_test_95, args.out_dir)

    ckpt_keys = sorted(checkpoints.keys())
    early = 0
    mid = min(
        [k for k in ckpt_keys if t_train_99 is not None and k >= t_train_99],
        default=ckpt_keys[min(len(ckpt_keys) - 1, len(ckpt_keys) // 3)],
    )
    late = min(
        [k for k in ckpt_keys if t_test_95 is not None and k >= t_test_95],
        default=ckpt_keys[-1],
    )

    snapshot_steps = [early, mid, late]
    snapshot_labels = [classify_snapshot(s, t_train_99, t_test_95, float(test_acc[-1])) for s in snapshot_steps]

    plot_pca(checkpoints, snapshot_steps, snapshot_labels, args.out_dir)
    plot_operation_tables(ds, checkpoints, snapshot_steps, snapshot_labels, cfg, args.out_dir)

    print("\nSaved outputs to", args.out_dir)
    print("Summary:", summary)


if __name__ == "__main__":
    main()