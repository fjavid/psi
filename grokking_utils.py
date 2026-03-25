"""
Shared utilities for the grokking experiment:
  - Dataset (modular addition mod p)
  - Model (decoder-only transformer)
  - Fourier probe
  - Training loop with dense logging
  - Plotting helpers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import math
import copy
import os
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

P = 97
PLUS_TOKEN = P
EQUALS_TOKEN = P + 1
VOCAB_SIZE = P + 2
SEQ_LEN = 4
NUM_CLASSES = P

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

def make_dataset(p, train_frac, seed=42):
    """Generate all (a+b) mod p pairs and split into train/test.
    Returns train_x, train_y, test_x, test_y, all_a, all_b, all_labels, train_mask.
    train_mask is a boolean array of shape (p*p,) indicating which pairs are in train.
    """
    rng = np.random.RandomState(seed)
    aa, bb = np.meshgrid(np.arange(p), np.arange(p))
    aa, bb = aa.flatten(), bb.flatten()
    labels = (aa + bb) % p

    seqs = np.stack([aa, np.full_like(aa, p), bb, np.full_like(aa, p + 1)], axis=1)

    n = len(labels)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True

    return (
        torch.tensor(seqs[train_idx], dtype=torch.long),
        torch.tensor(labels[train_idx], dtype=torch.long),
        torch.tensor(seqs[test_idx], dtype=torch.long),
        torch.tensor(labels[test_idx], dtype=torch.long),
        aa, bb, labels, train_mask,
    )


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_heads=4,
                 n_layers=2, d_ff=512, seq_len=SEQ_LEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.seq_len = seq_len
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        )

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h, self.causal_mask)
        h = self.ln_f(h[:, -1, :])
        return self.head(h)

    def get_number_embeddings(self):
        return self.tok_emb.weight[:P].detach().cpu()


# ═══════════════════════════════════════════════════════════════
# Fourier Probe
# ═══════════════════════════════════════════════════════════════

def build_fourier_basis(p):
    n = np.arange(p, dtype=np.float64)
    cols = [np.ones(p) / np.sqrt(p)]
    for k in range(1, (p - 1) // 2 + 1):
        c = np.cos(2 * np.pi * k * n / p)
        s = np.sin(2 * np.pi * k * n / p)
        cols.append(c / np.linalg.norm(c))
        cols.append(s / np.linalg.norm(s))
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)


FOURIER_BASIS = build_fourier_basis(P)


def fourier_energy(emb_weights, fourier_basis=FOURIER_BASIS):
    P_proj = fourier_basis.T @ emb_weights
    energy_per_col = (P_proj ** 2).sum(dim=1)
    n_freqs = (len(energy_per_col) - 1) // 2
    freq_energies = np.array([
        energy_per_col[1 + 2 * k].item() + energy_per_col[2 + 2 * k].item()
        for k in range(n_freqs)
    ])
    total_energy = (emb_weights ** 2).sum().item()
    top5_idx = np.argsort(freq_energies)[-5:]
    structured_ratio = freq_energies[top5_idx].sum() / (total_energy + 1e-12)
    probs = freq_energies / (freq_energies.sum() + 1e-12)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return freq_energies, structured_ratio, entropy, top5_idx


# ═══════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════

def compute_norms(model):
    norms = {}
    total_sq = sum(p.data.pow(2).sum().item() for p in model.parameters())
    norms['total'] = math.sqrt(total_sq)
    norms['tok_emb'] = model.tok_emb.weight.data.norm().item()
    norms['pos_emb'] = model.pos_emb.weight.data.norm().item()
    for i, block in enumerate(model.blocks):
        bsq = sum(p.data.pow(2).sum().item() for p in block.parameters())
        norms[f'block_{i}'] = math.sqrt(bsq)
    norms['head'] = model.head.weight.data.norm().item()
    return norms


def effective_rank(matrix):
    s = torch.linalg.svdvals(matrix.float())
    s = s / (s.sum() + 1e-12)
    ent = -(s * torch.log(s + 1e-12)).sum().item()
    return math.exp(ent)


@torch.no_grad()
def evaluate(model, x, y):
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    preds = logits.argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    correct_logits = logits[torch.arange(len(y)), y]
    logits_masked = logits.clone()
    logits_masked[torch.arange(len(y)), y] = -1e9
    margin = (correct_logits - logits_masked.max(dim=-1).values).mean().item()
    return loss, acc, margin


# ═══════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════

def train_run(train_x, train_y, test_x, test_y,
              n_steps=50000, lr=1e-3, wd=1.0, seed=0,
              log_every=10, probe_every=100,
              checkpoint_steps=None, fourier_basis=None,
              grad_clip=1.0, verbose=True):
    """Full training run. Saves a true step-0 checkpoint. Applies gradient clipping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GrokkingTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                  betas=(0.9, 0.98))

    tx, ty = train_x.to(DEVICE), train_y.to(DEVICE)
    vx, vy = test_x.to(DEVICE), test_y.to(DEVICE)

    metrics = dict(step=[], train_loss=[], test_loss=[],
                   train_acc=[], test_acc=[],
                   train_margin=[], test_margin=[], grad_norm=[])
    probe_metrics = dict(step=[], norms=[], emb_eff_rank=[],
                         fourier_structured_ratio=[], fourier_entropy=[],
                         fourier_freq_energies=[], fourier_top5_idx=[])
    checkpoints = {}
    checkpoint_steps = set(checkpoint_steps or [])

    # ── True step-0 checkpoint (before any training) ──
    model.eval()
    checkpoints[0] = dict(
        emb=model.get_number_embeddings().clone(),
        state_dict=copy.deepcopy(model.state_dict()),
    )
    tr_loss0, tr_acc0, tr_m0 = evaluate(model, tx, ty)
    te_loss0, te_acc0, te_m0 = evaluate(model, vx, vy)
    metrics['step'].append(0)
    metrics['train_loss'].append(tr_loss0)
    metrics['test_loss'].append(te_loss0)
    metrics['train_acc'].append(tr_acc0)
    metrics['test_acc'].append(te_acc0)
    metrics['train_margin'].append(tr_m0)
    metrics['test_margin'].append(te_m0)
    metrics['grad_norm'].append(0.0)

    probe_metrics['step'].append(0)
    probe_metrics['norms'].append(compute_norms(model))
    emb0 = model.get_number_embeddings()
    probe_metrics['emb_eff_rank'].append(effective_rank(emb0))
    if fourier_basis is not None:
        fe, sr, ent, t5 = fourier_energy(emb0, fourier_basis)
        probe_metrics['fourier_freq_energies'].append(fe)
        probe_metrics['fourier_structured_ratio'].append(sr)
        probe_metrics['fourier_entropy'].append(ent)
        probe_metrics['fourier_top5_idx'].append(t5)

    pbar = tqdm(range(1, n_steps + 1), disable=not verbose, desc='Training')
    for step in pbar:
        model.train()
        logits = model(tx)
        loss = F.cross_entropy(logits, ty)
        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        optimizer.step()

        if step % log_every == 0:
            model.eval()
            tr_l, tr_a, tr_m = evaluate(model, tx, ty)
            te_l, te_a, te_m = evaluate(model, vx, vy)
            metrics['step'].append(step)
            metrics['train_loss'].append(tr_l)
            metrics['test_loss'].append(te_l)
            metrics['train_acc'].append(tr_a)
            metrics['test_acc'].append(te_a)
            metrics['train_margin'].append(tr_m)
            metrics['test_margin'].append(te_m)
            metrics['grad_norm'].append(gn)
            if verbose:
                pbar.set_postfix(tr=f'{tr_a:.3f}', te=f'{te_a:.3f}',
                                 tl=f'{tr_l:.3f}', vl=f'{te_l:.3f}')

        if step % probe_every == 0:
            model.eval()
            probe_metrics['step'].append(step)
            probe_metrics['norms'].append(compute_norms(model))
            emb = model.get_number_embeddings()
            probe_metrics['emb_eff_rank'].append(effective_rank(emb))
            if fourier_basis is not None:
                fe, sr, ent, t5 = fourier_energy(emb, fourier_basis)
                probe_metrics['fourier_freq_energies'].append(fe)
                probe_metrics['fourier_structured_ratio'].append(sr)
                probe_metrics['fourier_entropy'].append(ent)
                probe_metrics['fourier_top5_idx'].append(t5)

        if step in checkpoint_steps:
            model.eval()
            checkpoints[step] = dict(
                emb=model.get_number_embeddings().clone(),
                state_dict=copy.deepcopy(model.state_dict()),
            )

    model.eval()
    if n_steps not in checkpoints:
        checkpoints[n_steps] = dict(
            emb=model.get_number_embeddings().clone(),
            state_dict=copy.deepcopy(model.state_dict()),
        )

    return model, metrics, probe_metrics, checkpoints


# ═══════════════════════════════════════════════════════════════
# Grokking delay helper
# ═══════════════════════════════════════════════════════════════

def grokking_stats(metrics):
    """Return t_train_99, t_test_95, tau_g from a metrics dict."""
    steps = np.array(metrics['step'])
    tr = np.array(metrics['train_acc'])
    te = np.array(metrics['test_acc'])
    t_tr = int(steps[tr >= 0.99][0]) if np.any(tr >= 0.99) else None
    t_te = int(steps[te >= 0.95][0]) if np.any(te >= 0.95) else None
    tau = (t_te - t_tr) if (t_tr is not None and t_te is not None) else None
    return t_tr, t_te, tau


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')


def plot_training_curves(metrics, t_train_99, t_test_95, tau_g, title_extra=''):
    steps = np.array(metrics['step'])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(steps, metrics['train_acc'], label='Train', lw=1.5)
    ax.plot(steps, metrics['test_acc'], label='Test', lw=1.5)
    ax.set_xscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy'); ax.set_ylim(-0.02, 1.05); ax.grid(True, alpha=0.3)
    if t_train_99 is not None:
        ax.axvline(t_train_99, color='blue', ls='--', alpha=0.5, label=f'Train 99% @ {t_train_99}')
    if t_test_95 is not None:
        ax.axvline(t_test_95, color='orange', ls='--', alpha=0.5, label=f'Test 95% @ {t_test_95}')
    ax.legend()

    ax = axes[1]
    ax.plot(steps, metrics['train_loss'], label='Train', lw=1.5)
    ax.plot(steps, metrics['test_loss'], label='Test', lw=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    tau_str = f', τ_g={tau_g:,}' if tau_g is not None else ''
    fig.suptitle(f'Grokking — Modular Addition mod {P}{tau_str} {title_extra}', fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_norms(metrics, probe_metrics, t_train_99, t_test_95):
    probe_steps = np.array(probe_metrics['step'])
    steps = np.array(metrics['step'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax in axes.flat:
        if t_train_99: ax.axvline(t_train_99, color='blue', ls='--', alpha=0.4)
        if t_test_95: ax.axvline(t_test_95, color='orange', ls='--', alpha=0.4)

    ax = axes[0, 0]
    ax.plot(probe_steps, [n['total'] for n in probe_metrics['norms']], lw=1.5, color='black')
    ax.set_xscale('log'); ax.set_ylabel('L2 Norm'); ax.set_title('Total Parameter Norm'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for key, lbl in [('tok_emb','Token Emb'), ('block_0','Block 0'),
                      ('block_1','Block 1'), ('head','Output Head')]:
        ax.plot(probe_steps, [n[key] for n in probe_metrics['norms']], label=lbl, lw=1.2)
    ax.set_xscale('log'); ax.set_ylabel('L2 Norm'); ax.set_title('Per-Component Norms')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, metrics['grad_norm'], lw=0.8, alpha=0.7)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel('Gradient Norm')
    ax.set_title('Global Gradient Norm (clipped)'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(probe_steps, probe_metrics['emb_eff_rank'], lw=1.5, color='purple')
    ax.set_xscale('log'); ax.set_ylabel('Effective Rank')
    ax.set_title('Embedding Effective Rank'); ax.grid(True, alpha=0.3)

    fig.suptitle('Internal Dynamics (blue=train 99%, orange=test 95%)', fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_fourier(metrics, probe_metrics, t_train_99, t_test_95):
    probe_steps = np.array(probe_metrics['step'])
    steps = np.array(metrics['step'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax in axes:
        if t_train_99: ax.axvline(t_train_99, color='blue', ls='--', alpha=0.4)
        if t_test_95: ax.axvline(t_test_95, color='orange', ls='--', alpha=0.4)

    axes[0].plot(probe_steps, probe_metrics['fourier_structured_ratio'], lw=1.5, color='crimson')
    axes[0].set_xscale('log'); axes[0].set_ylabel('Structured Ratio')
    axes[0].set_title('Top-5 Fourier / Total Energy'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(probe_steps, probe_metrics['fourier_entropy'], lw=1.5, color='darkgreen')
    axes[1].set_xscale('log'); axes[1].set_ylabel('Entropy (nats)')
    axes[1].set_title('Fourier Frequency Entropy'); axes[1].grid(True, alpha=0.3)

    all_e = np.array(probe_metrics['fourier_freq_energies'])
    top5 = probe_metrics['fourier_top5_idx'][-1]
    for fi in top5:
        axes[2].plot(probe_steps, all_e[:, fi], lw=1.2, label=f'Freq {fi+1}')
    axes[2].set_xscale('log'); axes[2].set_ylabel('Energy')
    axes[2].set_title('Top-5 Fourier Frequencies'); axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.suptitle('Fourier Concentration in Number Embeddings', fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, 'fourier_concentration.png')

    # Overlay
    fig2, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(steps, metrics['test_acc'], color='tab:orange', lw=1.5, label='Test Accuracy')
    ax1.set_xscale('log'); ax1.set_xlabel('Step')
    ax1.set_ylabel('Test Accuracy', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    ax2 = ax1.twinx()
    ax2.plot(probe_steps, probe_metrics['fourier_structured_ratio'],
             color='crimson', lw=1.5, label='Fourier Structured Ratio')
    ax2.set_ylabel('Structured Ratio', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')
    ax1.set_title('Fourier Structure vs. Test Accuracy')
    ax1.grid(True, alpha=0.3)
    fig2.tight_layout()
    savefig(fig2, 'fourier_vs_accuracy.png')


def pick_snapshot_steps(checkpoints, t_train_99, t_test_95, n_steps):
    """Pick 3 checkpoint steps with honest labels."""
    ckpt_keys = sorted(checkpoints.keys())
    early_step = 0 if 0 in checkpoints else ckpt_keys[0]

    if t_train_99 is not None:
        post_mem = min((k for k in ckpt_keys if k >= t_train_99), default=ckpt_keys[-1])
    else:
        post_mem = ckpt_keys[len(ckpt_keys) // 2]

    late_step = ckpt_keys[-1]

    if t_test_95 is not None:
        late_label = f'Step {late_step} (generalized)'
    else:
        late_label = f'Step {late_step} (late training — no generalization observed)'

    snapshot_steps = [early_step, post_mem, late_step]
    snapshot_labels = [
        f'Step {early_step} (init)',
        f'Step {post_mem} (memorized)',
        late_label,
    ]
    return snapshot_steps, snapshot_labels


def plot_pca(checkpoints, snapshot_steps, snapshot_labels):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, step, label in zip(axes, snapshot_steps, snapshot_labels):
        emb = checkpoints[step]['emb'].numpy()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb)
        ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(P), cmap='hsv', s=20, alpha=0.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    fig.suptitle('Number Embedding PCA (colored by token value)', fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_op_tables(checkpoints, snapshot_steps, snapshot_labels,
                   all_a, all_b, all_labels, train_mask):
    """Separate train vs held-out operation tables."""
    all_pairs_x = torch.tensor(
        np.stack([all_a, np.full_like(all_a, PLUS_TOKEN),
                  all_b, np.full_like(all_a, EQUALS_TOKEN)], axis=1),
        dtype=torch.long
    ).to(DEVICE)
    all_labels_t = torch.tensor(all_labels, dtype=torch.long).to(DEVICE)

    for table_type, mask, cmap_label in [
        ('train', train_mask, 'Train Set'),
        ('heldout', ~train_mask, 'Held-Out Set'),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, step, label in zip(axes, snapshot_steps, snapshot_labels):
            tmp = GrokkingTransformer().to(DEVICE)
            tmp.load_state_dict(checkpoints[step]['state_dict'])
            tmp.eval()
            with torch.no_grad():
                preds = tmp(all_pairs_x).argmax(dim=-1)
            correct_flat = (preds == all_labels_t).cpu().numpy()
            # -1 = not in this split, 0 = wrong, 1 = correct
            display = np.full(P * P, -1, dtype=int)
            display[mask] = correct_flat[mask].astype(int)
            display = display.reshape(P, P)
            cmap = mcolors.ListedColormap(['#e0e0e0', '#d32f2f', '#4caf50'])
            ax.imshow(display + 1, cmap=cmap, interpolation='nearest',
                      aspect='equal', vmin=0, vmax=2)
            subset_acc = correct_flat[mask].mean()
            ax.set_title(f'{label}\n({cmap_label} acc={subset_acc:.1%})', fontsize=10)
            ax.set_xlabel('b'); ax.set_ylabel('a')
            ax.set_xticks(np.arange(0, P, 20)); ax.set_yticks(np.arange(0, P, 20))
            del tmp
        fig.suptitle(f'{cmap_label}: (a+b) mod {P} — Green=Correct, Red=Incorrect, Gray=Other split',
                     fontsize=13, y=1.02)
        fig.tight_layout()
        savefig(fig, f'operation_table_{table_type}.png')
