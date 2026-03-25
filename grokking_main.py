"""
Phase 2: Main grokking run with full diagnostics.

Reads the config chosen by grokking_pilot.py from outputs/pilot_config.json.
Runs 3 seeds, picks the clearest grokking trajectory for detailed plots.
Generates: training curves, norm evolution, Fourier concentration,
           embedding PCA, and separate train / held-out operation tables.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')

from grokking_utils import (
    P, DEVICE, OUT_DIR, FOURIER_BASIS,
    make_dataset, train_run, grokking_stats,
    savefig,
    plot_training_curves, plot_norms, plot_fourier,
    pick_snapshot_steps, plot_pca, plot_op_tables,
)

MAIN_STEPS = 100_000
SEEDS = [42, 123, 7]


def load_config():
    path = os.path.join(OUT_DIR, 'pilot_config.json')
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        print(f'Loaded config from {path}: {cfg}')
        return cfg
    else:
        print('No pilot_config.json found, using defaults.')
        return dict(train_frac=0.3, weight_decay=1.0, lr=3e-4, grad_clip=1.0)


def main():
    print(f'Device: {DEVICE}')
    cfg = load_config()
    frac = cfg['train_frac']
    wd = cfg['weight_decay']
    lr = cfg.get('lr', 3e-4)
    gc = cfg.get('grad_clip', 1.0)

    # ── Multi-seed runs ──────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Main Run: frac={frac}, wd={wd}, lr={lr}, clip={gc}, steps={MAIN_STEPS}')
    print(f'Running {len(SEEDS)} seeds: {SEEDS}')
    print(f'{"="*60}')

    all_runs = []
    for seed in SEEDS:
        print(f'\n--- Seed {seed} ---')
        train_x, train_y, test_x, test_y, all_a, all_b, all_labels, train_mask = \
            make_dataset(P, frac, seed=seed)
        print(f'Train: {len(train_y)}, Test: {len(test_y)}')

        log_ckpts = set(np.unique(np.geomspace(1, MAIN_STEPS, 40).astype(int)).tolist())
        log_ckpts.update([100, 500, 1000, 2000, 5000, 10000, 20000, 50000])

        model, metrics, probe_metrics, checkpoints = train_run(
            train_x, train_y, test_x, test_y,
            n_steps=MAIN_STEPS, lr=lr, wd=wd, seed=seed,
            log_every=10, probe_every=100,
            checkpoint_steps=log_ckpts, fourier_basis=FOURIER_BASIS,
            grad_clip=gc, verbose=True,
        )
        t_tr, t_te, tau = grokking_stats(metrics)
        print(f'  Train 99%: {t_tr}, Test 95%: {t_te}, τ_g: {tau}')

        all_runs.append(dict(
            seed=seed, metrics=metrics, probe_metrics=probe_metrics,
            checkpoints=checkpoints, t_train_99=t_tr, t_test_95=t_te,
            tau_g=tau, train_mask=train_mask,
            all_a=all_a, all_b=all_b, all_labels=all_labels,
        ))

    # ── Pick best seed (clearest grokking, or highest final test acc) ──
    def score_run(r):
        if r['tau_g'] is not None:
            return (2, -r['tau_g'])  # prefer grokking, shorter delay first
        final_te = r['metrics']['test_acc'][-1]
        return (1 if final_te > 0.5 else 0, final_te)

    all_runs.sort(key=score_run, reverse=True)
    best = all_runs[0]

    grokked_seeds = [r['seed'] for r in all_runs if r['tau_g'] is not None]
    print(f'\n{"="*60}')
    print(f'Seeds that grokked: {grokked_seeds if grokked_seeds else "NONE"}')
    print(f'Using seed {best["seed"]} for detailed plots.')
    if best['tau_g'] is not None:
        print(f'  τ_g = {best["tau_g"]:,} steps')
    else:
        print(f'  No full grokking observed. Final test acc: {best["metrics"]["test_acc"][-1]:.3f}')
    print(f'{"="*60}')

    # ── Plots ─────────────────────────────────────────────────
    print('\nGenerating plots...')
    m = best['metrics']
    pm = best['probe_metrics']
    t_tr = best['t_train_99']
    t_te = best['t_test_95']
    tau = best['tau_g']

    fig = plot_training_curves(m, t_tr, t_te, tau, title_extra=f'(seed={best["seed"]})')
    savefig(fig, 'training_curves.png')

    fig = plot_norms(m, pm, t_tr, t_te)
    savefig(fig, 'norm_evolution.png')

    plot_fourier(m, pm, t_tr, t_te)

    snap_steps, snap_labels = pick_snapshot_steps(best['checkpoints'], t_tr, t_te, MAIN_STEPS)
    fig = plot_pca(best['checkpoints'], snap_steps, snap_labels)
    savefig(fig, 'embedding_pca.png')

    plot_op_tables(best['checkpoints'], snap_steps, snap_labels,
                   best['all_a'], best['all_b'], best['all_labels'], best['train_mask'])

    # ── Multi-seed summary plot ───────────────────────────────
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for r in all_runs:
        steps = np.array(r['metrics']['step'])
        te = np.array(r['metrics']['test_acc'])
        label = f'seed={r["seed"]}'
        if r['tau_g'] is not None:
            label += f' (τ={r["tau_g"]:,})'
        else:
            label += f' (no grok, final={te[-1]:.2f})'
        ax.plot(steps, te, lw=1.2, label=label)
    ax.set_xscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('Test Accuracy')
    ax.set_title(f'Test Accuracy Across Seeds (frac={frac}, wd={wd})')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    savefig(fig, 'multi_seed_test_acc.png')

    # ── Save summary ──────────────────────────────────────────
    summary = []
    for r in all_runs:
        summary.append(dict(
            seed=r['seed'],
            t_train_99=r['t_train_99'],
            t_test_95=r['t_test_95'],
            tau_g=r['tau_g'],
            final_train_acc=float(r['metrics']['train_acc'][-1]),
            final_test_acc=float(r['metrics']['test_acc'][-1]),
        ))
    with open(os.path.join(OUT_DIR, 'main_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nAll plots saved to {OUT_DIR}/')
    print('Done.')


if __name__ == '__main__':
    main()
