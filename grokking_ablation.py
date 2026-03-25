"""
Phase 3: Small ablation — train fraction × weight decay phase diagram.

Scaled down to be practical:
  - 4 train fractions × 4 weight decays = 16 runs
  - 15k steps each (enough to see if grokking starts or not)
  - Single seed

Reads lr and grad_clip from outputs/pilot_config.json if available.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from grokking_utils import (
    P, DEVICE, OUT_DIR, make_dataset, train_run, grokking_stats, savefig
)

ABLATION_FRACS = [0.3, 0.4, 0.5]
ABLATION_WDS = [0.0, 0.1, 1.0]
ABLATION_STEPS = 5_000
ABLATION_SEED = 42


def load_config():
    path = os.path.join(OUT_DIR, 'pilot_config.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(lr=3e-4, grad_clip=1.0)


def main():
    print(f'Device: {DEVICE}')
    cfg = load_config()
    lr = cfg.get('lr', 3e-4)
    gc = cfg.get('grad_clip', 1.0)

    print(f'\n{"="*60}')
    print(f'Ablation: {len(ABLATION_FRACS)} fracs × {len(ABLATION_WDS)} wds = '
          f'{len(ABLATION_FRACS)*len(ABLATION_WDS)} runs @ {ABLATION_STEPS} steps')
    print(f'lr={lr}, grad_clip={gc}')
    print(f'{"="*60}')

    phase = np.full((len(ABLATION_WDS), len(ABLATION_FRACS)), np.nan)

    for i, wd in enumerate(ABLATION_WDS):
        for j, frac in enumerate(ABLATION_FRACS):
            trx, try_, tex, tey, *_ = make_dataset(P, frac, seed=ABLATION_SEED)
            _, m, _, _ = train_run(
                trx, try_, tex, tey,
                n_steps=ABLATION_STEPS, lr=lr, wd=wd, seed=ABLATION_SEED,
                log_every=50, probe_every=ABLATION_STEPS + 1,
                grad_clip=gc, verbose=False,
            )
            t_tr, t_te, tau = grokking_stats(m)
            if t_tr is not None and t_te is not None:
                delay = t_te - t_tr
            elif t_tr is not None:
                delay = ABLATION_STEPS  # memorized but never generalized
            else:
                delay = -1  # didn't even memorize

            phase[i, j] = delay
            tag = f'τ={delay}' if delay >= 0 else 'no mem'
            print(f'  frac={frac:.1f}  wd={wd:<5}  train={m["train_acc"][-1]:.3f}  '
                  f'test={m["test_acc"][-1]:.3f}  {tag}')

    # ── Phase diagram plots ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Continuous delay
    ax = axes[0]
    dd = phase.copy()
    dd[dd < 0] = np.nan
    valid = dd[dd > 0]
    vmin = max(1, np.nanmin(valid)) if len(valid) > 0 else 1
    im = ax.imshow(dd, cmap='magma_r', aspect='auto',
                   norm=mcolors.LogNorm(vmin=vmin, vmax=ABLATION_STEPS))
    ax.set_xticks(range(len(ABLATION_FRACS)))
    ax.set_xticklabels([f'{f:.0%}' for f in ABLATION_FRACS])
    ax.set_yticks(range(len(ABLATION_WDS)))
    ax.set_yticklabels([str(w) for w in ABLATION_WDS])
    ax.set_xlabel('Train Fraction'); ax.set_ylabel('Weight Decay')
    ax.set_title('Grokking Delay τ_g (steps)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='τ_g')
    for i in range(len(ABLATION_WDS)):
        for j in range(len(ABLATION_FRACS)):
            v = phase[i, j]
            txt = 'N/M' if v < 0 else ('N/G' if v >= ABLATION_STEPS else f'{int(v)}')
            ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                    color='white' if (v > 5000 or v < 0) else 'black')

    # Regime classification
    ax = axes[1]
    regime = np.zeros_like(phase)
    regime[phase < 0] = 0                                          # no memorization
    regime[(phase >= 0) & (phase < 500)] = 1                       # immediate
    regime[(phase >= 500) & (phase < ABLATION_STEPS)] = 2          # grokking
    regime[phase >= ABLATION_STEPS] = 3                            # no gen
    cmap_r = mcolors.ListedColormap(['#9e9e9e', '#4caf50', '#ff9800', '#d32f2f'])
    ax.imshow(regime, cmap=cmap_r, aspect='auto', vmin=0, vmax=3)
    ax.set_xticks(range(len(ABLATION_FRACS)))
    ax.set_xticklabels([f'{f:.0%}' for f in ABLATION_FRACS])
    ax.set_yticks(range(len(ABLATION_WDS)))
    ax.set_yticklabels([str(w) for w in ABLATION_WDS])
    ax.set_xlabel('Train Fraction'); ax.set_ylabel('Weight Decay')
    ax.set_title('Grokking Regimes')
    ax.legend(handles=[
        Patch(facecolor='#9e9e9e', label='No memorization'),
        Patch(facecolor='#4caf50', label='Immediate gen. (τ<500)'),
        Patch(facecolor='#ff9800', label='Grokking (500≤τ<15k)'),
        Patch(facecolor='#d32f2f', label='No generalization'),
    ], loc='upper left', fontsize=9)

    fig.suptitle('Train Fraction × Weight Decay Phase Diagram', fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, 'phase_diagram.png')

    # Save raw data
    with open(os.path.join(OUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(dict(
            fracs=ABLATION_FRACS, wds=ABLATION_WDS,
            phase_diagram=phase.tolist(),
            steps=ABLATION_STEPS,
        ), f, indent=2)

    print(f'\nPhase diagram saved to {OUT_DIR}/phase_diagram.png')
    print('Done.')


if __name__ == '__main__':
    main()
