"""
Phase 1: Pilot sweep to find the grokking regime.

Two-phase approach:
  1. Coarse sweep (5k steps) across a fine grid of (train_frac, weight_decay).
  2. Re-run the most promising configs for 15k steps and pick one where
     train saturates early AND test is still low but *trending upward*.

Saves results to outputs/pilot_results.json for use by grokking_main.py.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grokking_utils import (
    P, DEVICE, OUT_DIR, make_dataset, train_run, grokking_stats, savefig
)

# ── Phase 1a: Coarse sweep ───────────────────────────────────

TRAIN_FRACS = [0.25, 0.30, 0.35, 0.40, 0.50]
WEIGHT_DECAYS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
SEEDS = [0, 1]
PHASE1_STEPS = 5_000


def run_coarse_sweep():
    print(f'Device: {DEVICE}')
    print(f'\n{"="*60}')
    print('Phase 1a: Coarse sweep (5k steps)')
    print(f'{"="*60}')
    print(f'{len(TRAIN_FRACS)} fracs × {len(WEIGHT_DECAYS)} wds × {len(SEEDS)} seeds '
          f'= {len(TRAIN_FRACS)*len(WEIGHT_DECAYS)*len(SEEDS)} runs')

    results = {}
    for frac in TRAIN_FRACS:
        for wd in WEIGHT_DECAYS:
            accs_tr, accs_te = [], []
            for seed in SEEDS:
                trx, try_, tex, tey, *_ = make_dataset(P, frac, seed=seed + 100)
                _, m, _, _ = train_run(
                    trx, try_, tex, tey,
                    n_steps=PHASE1_STEPS, lr=3e-4, wd=wd, seed=seed,
                    log_every=PHASE1_STEPS, probe_every=PHASE1_STEPS + 1,
                    grad_clip=1.0, verbose=False
                )
                accs_tr.append(m['train_acc'][-1])
                accs_te.append(m['test_acc'][-1])
            tr_mean, te_mean = np.mean(accs_tr), np.mean(accs_te)
            results[(frac, wd)] = (tr_mean, te_mean)
            print(f'  frac={frac:.2f}  wd={wd:<5}  train={tr_mean:.3f}  test={te_mean:.3f}  gap={tr_mean - te_mean:.3f}')
    return results


def plot_coarse(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, (key_i, title) in enumerate([(0, 'Train Acc @ 5k'), (1, 'Test Acc @ 5k')]):
        ax = axes[idx]
        data = np.zeros((len(WEIGHT_DECAYS), len(TRAIN_FRACS)))
        for i, wd in enumerate(WEIGHT_DECAYS):
            for j, frac in enumerate(TRAIN_FRACS):
                data[i, j] = results[(frac, wd)][key_i]
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(TRAIN_FRACS)))
        ax.set_xticklabels([f'{f:.0%}' for f in TRAIN_FRACS])
        ax.set_yticks(range(len(WEIGHT_DECAYS)))
        ax.set_yticklabels([str(w) for w in WEIGHT_DECAYS])
        ax.set_xlabel('Train Fraction'); ax.set_ylabel('Weight Decay')
        ax.set_title(title)
        for i in range(len(WEIGHT_DECAYS)):
            for j in range(len(TRAIN_FRACS)):
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, color='black' if data[i,j] > 0.4 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Pilot Sweep — Coarse (5k steps)', fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, 'pilot_coarse.png')


# ── Phase 1b: Longer runs for promising configs ──────────────

PHASE2_STEPS = 15_000


def select_candidates(results):
    """Pick configs where train > 0.9 and test is low — potential grokking zone.
    We want configs that memorize but haven't generalized yet."""
    candidates = []
    for (frac, wd), (tr, te) in results.items():
        if tr > 0.9 and te < 0.5 and wd > 0:
            candidates.append((frac, wd, tr, te))
    candidates.sort(key=lambda x: x[2] - x[3], reverse=True)
    return candidates[:6]


def run_longer_candidates(candidates):
    print(f'\n{"="*60}')
    print(f'Phase 1b: Longer runs (15k steps) for {len(candidates)} candidates')
    print(f'{"="*60}')

    longer_results = {}
    for frac, wd, _, _ in candidates:
        trx, try_, tex, tey, *_ = make_dataset(P, frac, seed=42)
        _, m, _, _ = train_run(
            trx, try_, tex, tey,
            n_steps=PHASE2_STEPS, lr=3e-4, wd=wd, seed=42,
            log_every=100, probe_every=PHASE2_STEPS + 1,
            grad_clip=1.0, verbose=False
        )
        steps_arr = np.array(m['step'])
        te_arr = np.array(m['test_acc'])
        tr_arr = np.array(m['train_acc'])

        # Compute test-accuracy trend in the second half
        mid = len(te_arr) // 2
        te_early = np.mean(te_arr[:mid])
        te_late = np.mean(te_arr[mid:])
        trend = te_late - te_early

        final_tr = tr_arr[-1]
        final_te = te_arr[-1]

        longer_results[(frac, wd)] = dict(
            final_train=float(final_tr),
            final_test=float(final_te),
            test_trend=float(trend),
            steps=int(PHASE2_STEPS),
        )
        marker = '  ← TRENDING UP' if trend > 0.02 else ''
        print(f'  frac={frac:.2f}  wd={wd:<5}  train={final_tr:.3f}  '
              f'test={final_te:.3f}  trend={trend:+.4f}{marker}')

    return longer_results


def pick_best_config(longer_results):
    """Pick the config where train is saturated and test shows the most upward trend.
    This is the sweet spot for grokking — memorized but about to generalize."""
    scored = []
    for (frac, wd), info in longer_results.items():
        if info['final_train'] > 0.95:
            scored.append((frac, wd, info['test_trend'], info['final_test']))
    if not scored:
        # Fallback: just pick best memorized with any weight decay
        for (frac, wd), info in longer_results.items():
            scored.append((frac, wd, info['test_trend'], info['final_test']))

    # Prefer positive test trend, break ties by lower final test (more room to grok)
    scored.sort(key=lambda x: (x[2] > 0.01, x[2], -x[3]), reverse=True)
    best_frac, best_wd = scored[0][0], scored[0][1]
    return best_frac, best_wd


def plot_longer(longer_results):
    items = list(longer_results.items())
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = [f'f={frac:.2f}\nwd={wd}' for (frac, wd) in [k for k, _ in items]]
    trends = [v['test_trend'] for _, v in items]
    colors = ['#4caf50' if t > 0.02 else '#ff9800' if t > 0 else '#d32f2f' for t in trends]
    ax.bar(range(len(items)), trends, color=colors)
    ax.set_xticks(range(len(items))); ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel('Test Acc Trend (late − early)')
    ax.set_title('Test Accuracy Trend in 15k Longer Runs')
    ax.axhline(0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    savefig(fig, 'pilot_longer.png')


# ── Main ─────────────────────────────────────────────────────

def main():
    coarse = run_coarse_sweep()
    plot_coarse(coarse)

    candidates = select_candidates(coarse)
    if not candidates:
        print('\nWARNING: No good candidates found. Using fallback config.')
        best_frac, best_wd = 0.3, 1.0
    else:
        longer = run_longer_candidates(candidates)
        plot_longer(longer)
        best_frac, best_wd = pick_best_config(longer)

    config = dict(train_frac=best_frac, weight_decay=best_wd, lr=3e-4, grad_clip=1.0)
    out_path = f'{OUT_DIR}/pilot_config.json'
    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\n{"="*60}')
    print(f'SELECTED CONFIG: frac={best_frac}, wd={best_wd}')
    print(f'Saved to {out_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
