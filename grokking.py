"""
Grokking in Neural Networks: Modular Addition mod 97
=====================================================

Central Question: What predicts grokking better — raw weight-norm decay,
or the emergence of structured Fourier-like representations?

Hypothesis: The model first memorizes, then gradually builds a Fourier
representation of modular arithmetic. Fourier concentration increases
before the test-accuracy jump — grokking is externally abrupt but
internally gradual.
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
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. Dataset
# ═══════════════════════════════════════════════════════════════

P = 97
PLUS_TOKEN = P
EQUALS_TOKEN = P + 1
VOCAB_SIZE = P + 2
SEQ_LEN = 4
NUM_CLASSES = P


def make_dataset(p, train_frac, seed=42):
    rng = np.random.RandomState(seed)
    a_vals = np.arange(p)
    b_vals = np.arange(p)
    aa, bb = np.meshgrid(a_vals, b_vals)
    aa, bb = aa.flatten(), bb.flatten()
    labels = (aa + bb) % p

    seqs = np.stack([aa,
                     np.full_like(aa, PLUS_TOKEN),
                     bb,
                     np.full_like(aa, EQUALS_TOKEN)], axis=1)

    n = len(labels)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    return (
        torch.tensor(seqs[train_idx], dtype=torch.long),
        torch.tensor(labels[train_idx], dtype=torch.long),
        torch.tensor(seqs[test_idx], dtype=torch.long),
        torch.tensor(labels[test_idx], dtype=torch.long),
        aa, bb, labels,
    )


# ═══════════════════════════════════════════════════════════════
# 2. Model
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
# 3. Fourier Probe
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


def fourier_energy(emb_weights, fourier_basis):
    P_proj = fourier_basis.T @ emb_weights
    energy_per_col = (P_proj ** 2).sum(dim=1)

    n_freqs = (len(energy_per_col) - 1) // 2
    freq_energies = np.array([
        energy_per_col[1 + 2*k].item() + energy_per_col[2 + 2*k].item()
        for k in range(n_freqs)
    ])

    total_energy = (emb_weights ** 2).sum().item()
    top5_idx = np.argsort(freq_energies)[-5:]
    structured_ratio = freq_energies[top5_idx].sum() / (total_energy + 1e-12)

    probs = freq_energies / (freq_energies.sum() + 1e-12)
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    return freq_energies, structured_ratio, entropy, top5_idx


FOURIER_BASIS = build_fourier_basis(P)


# ═══════════════════════════════════════════════════════════════
# 4. Helpers
# ═══════════════════════════════════════════════════════════════

def compute_norms(model):
    norms = {}
    total_sq = sum(p.data.pow(2).sum().item() for p in model.parameters())
    norms['total'] = math.sqrt(total_sq)
    norms['tok_emb'] = model.tok_emb.weight.data.norm().item()
    norms['pos_emb'] = model.pos_emb.weight.data.norm().item()
    for i, block in enumerate(model.blocks):
        block_sq = sum(p.data.pow(2).sum().item() for p in block.parameters())
        norms[f'block_{i}'] = math.sqrt(block_sq)
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
# 5. Training Loop
# ═══════════════════════════════════════════════════════════════

def train_run(train_x, train_y, test_x, test_y,
              n_steps=50000, lr=1e-3, wd=1.0, seed=0,
              log_every=10, probe_every=100,
              checkpoint_steps=None, fourier_basis=None,
              verbose=True):
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

    pbar = tqdm(range(1, n_steps + 1), disable=not verbose, desc='Training')
    for step in pbar:
        model.train()
        logits = model(tx)
        loss = F.cross_entropy(logits, ty)
        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        optimizer.step()

        if step % log_every == 0:
            model.eval()
            tr_loss, tr_acc, tr_margin = evaluate(model, tx, ty)
            te_loss, te_acc, te_margin = evaluate(model, vx, vy)
            metrics['step'].append(step)
            metrics['train_loss'].append(tr_loss)
            metrics['test_loss'].append(te_loss)
            metrics['train_acc'].append(tr_acc)
            metrics['test_acc'].append(te_acc)
            metrics['train_margin'].append(tr_margin)
            metrics['test_margin'].append(te_margin)
            metrics['grad_norm'].append(gn)
            if verbose:
                pbar.set_postfix(tr_acc=f'{tr_acc:.3f}', te_acc=f'{te_acc:.3f}',
                                 tr_loss=f'{tr_loss:.3f}', te_loss=f'{te_loss:.3f}')

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
    checkpoints[n_steps] = dict(
        emb=model.get_number_embeddings().clone(),
        state_dict=copy.deepcopy(model.state_dict()),
    )
    return model, metrics, probe_metrics, checkpoints


# ═══════════════════════════════════════════════════════════════
# 6. Plotting helpers
# ═══════════════════════════════════════════════════════════════

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {OUT_DIR}/{name}')


def plot_pilot(pilot_results, train_fracs, weight_decays):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (key, title) in enumerate([
        (0, 'Train Accuracy @ 5k steps'), (1, 'Test Accuracy @ 5k steps')
    ]):
        ax = axes[ax_idx]
        data = np.zeros((len(weight_decays), len(train_fracs)))
        for i, wd in enumerate(weight_decays):
            for j, frac in enumerate(train_fracs):
                data[i, j] = pilot_results[(frac, wd)][key]
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(train_fracs)))
        ax.set_xticklabels([f'{f:.0%}' for f in train_fracs])
        ax.set_yticks(range(len(weight_decays)))
        ax.set_yticklabels([str(w) for w in weight_decays])
        ax.set_xlabel('Train Fraction'); ax.set_ylabel('Weight Decay')
        ax.set_title(title)
        for i in range(len(weight_decays)):
            for j in range(len(train_fracs)):
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                        fontsize=11, color='black' if data[i,j] > 0.4 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Pilot Sweep: Finding the Grokking Regime', fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, 'pilot_sweep.png')


def plot_training_curves(steps, metrics, train_accs, test_accs,
                         t_train_99, t_test_95, tau_g, frac, wd):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(steps, train_accs, label='Train Accuracy', lw=1.5)
    ax.plot(steps, test_accs, label='Test Accuracy', lw=1.5)
    ax.set_xscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy over Training'); ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    if t_train_99 is not None:
        ax.axvline(t_train_99, color='blue', ls='--', alpha=0.5,
                   label=f'Train 99% @ {t_train_99}')
    if t_test_95 is not None:
        ax.axvline(t_test_95, color='orange', ls='--', alpha=0.5,
                   label=f'Test 95% @ {t_test_95}')
    ax.legend()

    ax = axes[1]
    ax.plot(steps, metrics['train_loss'], label='Train Loss', lw=1.5)
    ax.plot(steps, metrics['test_loss'], label='Test Loss', lw=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title('Loss over Training'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Grokking on Modular Addition mod {P} (frac={frac}, wd={wd})',
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, 'training_curves.png')


def plot_norms(steps, metrics, probe_steps, probe_metrics, t_train_99, t_test_95):
    total_norms = [n['total'] for n in probe_metrics['norms']]
    tok_norms = [n['tok_emb'] for n in probe_metrics['norms']]
    head_norms = [n['head'] for n in probe_metrics['norms']]
    b0 = [n['block_0'] for n in probe_metrics['norms']]
    b1 = [n['block_1'] for n in probe_metrics['norms']]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax in axes.flat:
        if t_train_99: ax.axvline(t_train_99, color='blue', ls='--', alpha=0.4)
        if t_test_95: ax.axvline(t_test_95, color='orange', ls='--', alpha=0.4)

    ax = axes[0, 0]
    ax.plot(probe_steps, total_norms, lw=1.5, color='black')
    ax.set_xscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('L2 Norm')
    ax.set_title('Total Parameter Norm'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for vals, lbl in [(tok_norms,'Token Emb'),(b0,'Block 0'),(b1,'Block 1'),(head_norms,'Output Head')]:
        ax.plot(probe_steps, vals, label=lbl, lw=1.2)
    ax.set_xscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('L2 Norm')
    ax.set_title('Per-Component Norms'); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, metrics['grad_norm'], lw=0.8, alpha=0.7)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Step'); ax.set_ylabel('Gradient Norm')
    ax.set_title('Global Gradient Norm'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(probe_steps, probe_metrics['emb_eff_rank'], lw=1.5, color='purple')
    ax.set_xscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('Effective Rank')
    ax.set_title('Embedding Matrix Effective Rank'); ax.grid(True, alpha=0.3)

    fig.suptitle('Internal Dynamics (blue=train 99%, orange=test 95%)', fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, 'norm_evolution.png')


def plot_fourier(steps, test_accs, probe_steps, probe_metrics, t_train_99, t_test_95):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(probe_steps, probe_metrics['fourier_structured_ratio'], lw=1.5, color='crimson')
    ax.set_xscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('Structured Ratio')
    ax.set_title('Top-5 Fourier Energy / Total Energy'); ax.grid(True, alpha=0.3)
    if t_train_99: ax.axvline(t_train_99, color='blue', ls='--', alpha=0.4)
    if t_test_95: ax.axvline(t_test_95, color='orange', ls='--', alpha=0.4)

    ax = axes[1]
    ax.plot(probe_steps, probe_metrics['fourier_entropy'], lw=1.5, color='darkgreen')
    ax.set_xscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('Entropy (nats)')
    ax.set_title('Fourier Frequency Entropy'); ax.grid(True, alpha=0.3)
    if t_train_99: ax.axvline(t_train_99, color='blue', ls='--', alpha=0.4)
    if t_test_95: ax.axvline(t_test_95, color='orange', ls='--', alpha=0.4)

    ax = axes[2]
    all_energies = np.array(probe_metrics['fourier_freq_energies'])
    final_top5 = probe_metrics['fourier_top5_idx'][-1]
    for freq_idx in final_top5:
        ax.plot(probe_steps, all_energies[:, freq_idx], lw=1.2, label=f'Freq {freq_idx+1}')
    ax.set_xscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('Energy')
    ax.set_title('Top-5 Fourier Frequency Energies'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if t_train_99: ax.axvline(t_train_99, color='blue', ls='--', alpha=0.4)
    if t_test_95: ax.axvline(t_test_95, color='orange', ls='--', alpha=0.4)

    fig.suptitle('Fourier Concentration in Number Embeddings', fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, 'fourier_concentration.png')

    # Overlay plot
    fig2, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(steps, test_accs, color='tab:orange', lw=1.5, label='Test Accuracy')
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
    ax1.set_title('Fourier Structure Emerges Before Test Accuracy Jumps')
    ax1.grid(True, alpha=0.3)
    fig2.tight_layout()
    save(fig2, 'fourier_vs_accuracy.png')


def plot_pca(checkpoints, snapshot_steps, snapshot_labels):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, step, label in zip(axes, snapshot_steps, snapshot_labels):
        emb = checkpoints[step]['emb'].numpy()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb)
        ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(P), cmap='hsv', s=20, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    fig.suptitle('Number Embedding PCA (colored by token value)', fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, 'embedding_pca.png')


def plot_op_tables(checkpoints, snapshot_steps, snapshot_labels, all_a, all_b, all_labels):
    all_pairs_x = torch.tensor(
        np.stack([all_a, np.full_like(all_a, PLUS_TOKEN),
                  all_b, np.full_like(all_a, EQUALS_TOKEN)], axis=1),
        dtype=torch.long
    ).to(DEVICE)
    all_labels_t = torch.tensor(all_labels, dtype=torch.long).to(DEVICE)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, step, label in zip(axes, snapshot_steps, snapshot_labels):
        temp_model = GrokkingTransformer().to(DEVICE)
        temp_model.load_state_dict(checkpoints[step]['state_dict'])
        temp_model.eval()
        with torch.no_grad():
            preds = temp_model(all_pairs_x).argmax(dim=-1)
        correct = (preds == all_labels_t).cpu().numpy().reshape(P, P)
        cmap = mcolors.ListedColormap(['#d32f2f', '#4caf50'])
        ax.imshow(correct, cmap=cmap, interpolation='nearest', aspect='equal')
        ax.set_title(f'{label}\n(acc={correct.mean():.1%})')
        ax.set_xlabel('b'); ax.set_ylabel('a')
        ax.set_xticks(np.arange(0, P, 20)); ax.set_yticks(np.arange(0, P, 20))
        del temp_model
    fig.suptitle(f'Operation Table: (a+b) mod {P} — Green=Correct, Red=Incorrect',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, 'operation_tables.png')


def plot_phase_diagram(phase_diagram, ablation_fracs, ablation_wds, ablation_steps):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ax = axes[0]
    dd = phase_diagram.copy()
    dd[dd < 0] = np.nan
    valid_pos = dd[dd > 0]
    vmin = max(1, np.nanmin(valid_pos)) if len(valid_pos) > 0 else 1
    im = ax.imshow(dd, cmap='magma_r', aspect='auto',
                   norm=mcolors.LogNorm(vmin=vmin, vmax=ablation_steps))
    ax.set_xticks(range(len(ablation_fracs)))
    ax.set_xticklabels([f'{f:.0%}' for f in ablation_fracs])
    ax.set_yticks(range(len(ablation_wds)))
    ax.set_yticklabels([str(w) for w in ablation_wds])
    ax.set_xlabel('Train Fraction'); ax.set_ylabel('Weight Decay')
    ax.set_title('Grokking Delay τ_g (steps)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='τ_g (steps)')
    for i in range(len(ablation_wds)):
        for j in range(len(ablation_fracs)):
            val = phase_diagram[i, j]
            txt = 'N/M' if val < 0 else ('N/G' if val >= ablation_steps else f'{int(val)}')
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='white' if (val > 5000 or val < 0) else 'black')

    ax = axes[1]
    regime = np.zeros_like(phase_diagram)
    regime[phase_diagram < 0] = 0
    regime[(phase_diagram >= 0) & (phase_diagram < 500)] = 1
    regime[(phase_diagram >= 500) & (phase_diagram < ablation_steps)] = 2
    regime[phase_diagram >= ablation_steps] = 3
    cmap_r = mcolors.ListedColormap(['#9e9e9e', '#4caf50', '#ff9800', '#d32f2f'])
    ax.imshow(regime, cmap=cmap_r, aspect='auto', vmin=0, vmax=3)
    ax.set_xticks(range(len(ablation_fracs)))
    ax.set_xticklabels([f'{f:.0%}' for f in ablation_fracs])
    ax.set_yticks(range(len(ablation_wds)))
    ax.set_yticklabels([str(w) for w in ablation_wds])
    ax.set_xlabel('Train Fraction'); ax.set_ylabel('Weight Decay')
    ax.set_title('Phase Diagram: Grokking Regimes')
    ax.legend(handles=[
        Patch(facecolor='#9e9e9e', label='No memorization'),
        Patch(facecolor='#4caf50', label='Immediate gen. (τ<500)'),
        Patch(facecolor='#ff9800', label='Grokking (500≤τ<20k)'),
        Patch(facecolor='#d32f2f', label='No generalization'),
    ], loc='upper left', fontsize=9)

    fig.suptitle('Train Fraction × Weight Decay Phase Diagram', fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, 'phase_diagram.png')


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')

    # ── 1. Pilot Sweep ────────────────────────────────────────
    print('\n' + '='*60)
    print('PHASE 1: Pilot Sweep')
    print('='*60)

    train_fracs = [0.3, 0.4, 0.5]
    weight_decays = [0.0, 0.1, 1.0]
    seeds = [0, 1]
    PILOT_STEPS = 5000
    pilot_results = {}

    for frac in train_fracs:
        for wd in weight_decays:
            accs = {'train': [], 'test': []}
            for seed in seeds:
                trx, try_, tex, tey, _, _, _ = make_dataset(P, frac, seed=seed+100)
                _, m, _, _ = train_run(
                    trx, try_, tex, tey,
                    n_steps=PILOT_STEPS, lr=1e-3, wd=wd, seed=seed,
                    log_every=PILOT_STEPS, probe_every=PILOT_STEPS+1,
                    verbose=False
                )
                accs['train'].append(m['train_acc'][-1])
                accs['test'].append(m['test_acc'][-1])
            tr_mean, te_mean = np.mean(accs['train']), np.mean(accs['test'])
            pilot_results[(frac, wd)] = (tr_mean, te_mean)
            print(f'  frac={frac}, wd={wd}: train={tr_mean:.3f}, test={te_mean:.3f}, gap={tr_mean-te_mean:.3f}')

    plot_pilot(pilot_results, train_fracs, weight_decays)

    best_key = max(pilot_results.keys(),
                   key=lambda k: pilot_results[k][0] - pilot_results[k][1]
                   if pilot_results[k][0] > 0.9 else -1)
    print(f'\nBest grokking config: frac={best_key[0]}, wd={best_key[1]}')

    # ── 2. Main Long Run ──────────────────────────────────────
    print('\n' + '='*60)
    print('PHASE 2: Main Long Run')
    print('='*60)

    MAIN_FRAC, MAIN_WD = best_key
    MAIN_STEPS = 100_000
    MAIN_SEED = 42

    print(f'Config: frac={MAIN_FRAC}, wd={MAIN_WD}, steps={MAIN_STEPS}')
    train_x, train_y, test_x, test_y, all_a, all_b, all_labels = \
        make_dataset(P, MAIN_FRAC, seed=MAIN_SEED)
    print(f'Train: {len(train_y)}, Test: {len(test_y)}')

    log_ckpts = set(np.unique(np.geomspace(1, MAIN_STEPS, 40).astype(int)).tolist())
    log_ckpts.update([100, 500, 1000, 2000, 5000, 10000, 20000, 50000])

    model, metrics, probe_metrics, checkpoints = train_run(
        train_x, train_y, test_x, test_y,
        n_steps=MAIN_STEPS, lr=1e-3, wd=MAIN_WD, seed=MAIN_SEED,
        log_every=10, probe_every=100,
        checkpoint_steps=log_ckpts, fourier_basis=FOURIER_BASIS, verbose=True
    )

    steps = np.array(metrics['step'])
    train_accs = np.array(metrics['train_acc'])
    test_accs = np.array(metrics['test_acc'])
    probe_steps = np.array(probe_metrics['step'])

    t_train_99 = steps[train_accs >= 0.99][0] if np.any(train_accs >= 0.99) else None
    t_test_95 = steps[test_accs >= 0.95][0] if np.any(test_accs >= 0.95) else None
    tau_g = (t_test_95 - t_train_99) if (t_train_99 is not None and t_test_95 is not None) else None

    if tau_g is not None:
        print(f'\nGrokking delay τ_g = {tau_g:,} steps')
        print(f'  Train 99% @ step {t_train_99},  Test 95% @ step {t_test_95}')
    else:
        print(f'\nGrokking not fully observed in {MAIN_STEPS} steps.')
        print(f'  Train 99%: {"step "+str(t_train_99) if t_train_99 else "not reached"}')
        print(f'  Test 95%: {"step "+str(t_test_95) if t_test_95 else "not reached"}')

    # ── 3. Plots ──────────────────────────────────────────────
    print('\n' + '='*60)
    print('PHASE 3: Generating Plots')
    print('='*60)

    plot_training_curves(steps, metrics, train_accs, test_accs,
                         t_train_99, t_test_95, tau_g, MAIN_FRAC, MAIN_WD)
    plot_norms(steps, metrics, probe_steps, probe_metrics, t_train_99, t_test_95)
    plot_fourier(steps, test_accs, probe_steps, probe_metrics, t_train_99, t_test_95)

    # Pick 3 snapshot checkpoints
    ckpt_keys = sorted(checkpoints.keys())
    early_step = ckpt_keys[0]
    post_mem_step = (min(k for k in ckpt_keys if k >= t_train_99)
                     if t_train_99 is not None else ckpt_keys[len(ckpt_keys)//2])
    post_grok_step = (min(k for k in ckpt_keys if k >= t_test_95)
                      if t_test_95 is not None else ckpt_keys[-1])
    snapshot_steps = [early_step, post_mem_step, post_grok_step]
    snapshot_labels = [
        f'Step {early_step} (init)',
        f'Step {post_mem_step} (memorized)',
        f'Step {post_grok_step} (generalized)',
    ]

    plot_pca(checkpoints, snapshot_steps, snapshot_labels)
    plot_op_tables(checkpoints, snapshot_steps, snapshot_labels, all_a, all_b, all_labels)

    # ── 4. Ablation Sweep ─────────────────────────────────────
    print('\n' + '='*60)
    print('PHASE 4: Ablation — Train Fraction × Weight Decay')
    print('='*60)

    ablation_fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ablation_wds = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
    ABLATION_STEPS = 20_000
    ABLATION_SEED = 42

    phase_diagram = np.full((len(ablation_wds), len(ablation_fracs)), np.nan)
    for i, wd in enumerate(ablation_wds):
        for j, frac in enumerate(ablation_fracs):
            trx, try_, tex, tey, _, _, _ = make_dataset(P, frac, seed=ABLATION_SEED)
            _, m, _, _ = train_run(
                trx, try_, tex, tey,
                n_steps=ABLATION_STEPS, lr=1e-3, wd=wd, seed=ABLATION_SEED,
                log_every=50, probe_every=ABLATION_STEPS+1, verbose=False
            )
            s = np.array(m['step'])
            tr_a = np.array(m['train_acc'])
            te_a = np.array(m['test_acc'])
            t99 = s[tr_a >= 0.99][0] if np.any(tr_a >= 0.99) else None
            t95 = s[te_a >= 0.95][0] if np.any(te_a >= 0.95) else None
            if t99 is not None and t95 is not None:
                delay = t95 - t99
            elif t99 is not None:
                delay = ABLATION_STEPS
            else:
                delay = -1
            phase_diagram[i, j] = delay
            status = f'τ={delay}' if delay >= 0 else 'no mem'
            print(f'  frac={frac:.1f}, wd={wd:.2f}: train={tr_a[-1]:.3f}, '
                  f'test={te_a[-1]:.3f}, {status}')

    plot_phase_diagram(phase_diagram, ablation_fracs, ablation_wds, ABLATION_STEPS)

    # ── Done ──────────────────────────────────────────────────
    print('\n' + '='*60)
    print('ALL DONE — plots saved to outputs/')
    print('='*60)


if __name__ == '__main__':
    main()
