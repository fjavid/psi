# Grokking in Neural Networks

Reproducing and studying the grokking phenomenon on modular addition mod 97 with a small decoder-only transformer. The central question: does Fourier structure in the learned embeddings emerge before the sharp test-accuracy jump, suggesting grokking is externally abrupt but internally gradual?

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU. Tested on NVIDIA RTX 6000 Ada with PyTorch 2.4+.

## Files

| File | Purpose |
|---|---|
| `grokking_common.py` | Shared code: model, dataset, Fourier probe, training loop, plotting |
| `grokking_pilot.py` | Stage 1: hyperparameter sweep to find the grokking regime |
| `grokking_main.py` | Stage 2: long diagnostic run with full metric and checkpoint logging |
| `grokking_ablation.py` | Stage 3: train-fraction x weight-decay phase diagram |
| `comments.md` | Write-up of experimental design, metrics, and interpretation |

## Running

The experiment runs in three stages. Each stage saves outputs to `outputs/`.

### Stage 1: Pilot sweep

Searches a grid of train fractions and weight decays to find configs where the model memorizes but hasn't yet generalized (the grokking zone). Saves the best config to `outputs/pilot/selected_config.json`.

```bash
python grokking_pilot.py
```

Optional arguments:
- `--steps 10000` — steps per pilot run (default: 10000)
- `--train_fracs 0.25 0.30 0.35 0.40 0.50` — fractions to sweep
- `--weight_decays 0.0 0.01 0.03 0.1 0.3 1.0` — weight decays to sweep
- `--seeds 0 1` — random seeds for averaging
- `--lr 3e-4` — learning rate

### Stage 2: Main run

Runs a single long training with dense logging of accuracy, loss, norms, gradient norms, Fourier concentration, embedding PCA, and operation-table snapshots. Reads the pilot config by default, or accepts manual overrides.

```bash
python grokking_main.py --steps 100000
```

Optional arguments:
- `--train_frac 0.4` — override train fraction
- `--weight_decay 0.3` — override weight decay
- `--steps 100000` — total training steps (default: 50000)
- `--lr 3e-4` — learning rate
- `--seed 42` — random seed
- `--pilot_config outputs/pilot/selected_config.json` — path to pilot config

### Stage 3: Ablation

Maps the grokking delay across a grid of train fractions and weight decays, producing a phase diagram that shows immediate generalization, delayed generalization (grokking), and no-generalization regimes.

```bash
python grokking_ablation.py --steps 15000
```

Optional arguments:
- `--fracs 0.35 0.40 0.45 0.50` — train fractions
- `--wds 0.1 0.3 0.5 1.0` — weight decays
- `--seeds 0 1 2` — seeds (results are averaged)
- `--steps 50000` — steps per run (default: 50000)
- `--eval_every 250` — evaluation frequency

## Outputs

All saved to `outputs/`. Key plots:

- `outputs/pilot/` — pilot sweep heatmaps and selected config
- `outputs/main/training_curves.png` — train/test accuracy and loss
- `outputs/main/norms.png` — parameter norms, gradient norm, embedding effective rank
- `outputs/main/fourier.png` — Fourier structured ratio, entropy, top-5 frequency energies
- `outputs/main/fourier_vs_accuracy.png` — overlay of Fourier structure vs test accuracy
- `outputs/main/embedding_pca.png` — PCA of number embeddings at 3 checkpoints
- `outputs/main/operation_tables_split.png` — correctness on train and held-out pairs separately
- `outputs/ablation/phase_diagram.png` — train-fraction x weight-decay regime map
