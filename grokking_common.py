import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Patch
from sklearn.decomposition import PCA


P = 97
PLUS_TOKEN = P
EQUALS_TOKEN = P + 1
VOCAB_SIZE = P + 2
SEQ_LEN = 4
NUM_CLASSES = P

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(p: int, train_frac: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    a_vals = np.arange(p)
    b_vals = np.arange(p)
    aa, bb = np.meshgrid(a_vals, b_vals, indexing="ij")
    aa = aa.flatten()
    bb = bb.flatten()
    labels = (aa + bb) % p

    seqs = np.stack(
        [aa, np.full_like(aa, PLUS_TOKEN), bb, np.full_like(aa, EQUALS_TOKEN)],
        axis=1,
    )

    n = len(labels)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask

    return {
        "train_x": torch.tensor(seqs[train_idx], dtype=torch.long),
        "train_y": torch.tensor(labels[train_idx], dtype=torch.long),
        "test_x": torch.tensor(seqs[test_idx], dtype=torch.long),
        "test_y": torch.tensor(labels[test_idx], dtype=torch.long),
        "all_x": torch.tensor(seqs, dtype=torch.long),
        "all_y": torch.tensor(labels, dtype=torch.long),
        "all_a": aa,
        "all_b": bb,
        "labels_np": labels,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False, is_causal=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        seq_len: int = SEQ_LEN,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h, self.causal_mask)
        h = self.ln_f(h[:, -1, :])
        return self.head(h)

    def get_number_embeddings(self) -> torch.Tensor:
        return self.tok_emb.weight[:P].detach().cpu()


def build_fourier_basis(p: int) -> torch.Tensor:
    n = np.arange(p, dtype=np.float64)
    cols = [np.ones(p) / np.sqrt(p)]
    for k in range(1, (p - 1) // 2 + 1):
        c = np.cos(2 * np.pi * k * n / p)
        s = np.sin(2 * np.pi * k * n / p)
        cols.append(c / np.linalg.norm(c))
        cols.append(s / np.linalg.norm(s))
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)


FOURIER_BASIS = build_fourier_basis(P)


def fourier_energy(emb_weights: torch.Tensor, fourier_basis: torch.Tensor):
    proj = fourier_basis.T @ emb_weights
    energy_per_col = (proj ** 2).sum(dim=1)

    n_freqs = (len(energy_per_col) - 1) // 2
    freq_energies = np.array(
        [
            energy_per_col[1 + 2 * k].item() + energy_per_col[2 + 2 * k].item()
            for k in range(n_freqs)
        ]
    )

    total_energy = (emb_weights ** 2).sum().item()
    top5_idx = np.argsort(freq_energies)[-5:]
    structured_ratio = freq_energies[top5_idx].sum() / (total_energy + 1e-12)
    probs = freq_energies / (freq_energies.sum() + 1e-12)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return freq_energies, structured_ratio, entropy, top5_idx


def compute_norms(model: nn.Module) -> Dict[str, float]:
    norms = {}
    total_sq = sum(p.data.pow(2).sum().item() for p in model.parameters())
    norms["total"] = math.sqrt(total_sq)
    norms["tok_emb"] = model.tok_emb.weight.data.norm().item()
    norms["pos_emb"] = model.pos_emb.weight.data.norm().item()
    for i, block in enumerate(model.blocks):
        block_sq = sum(p.data.pow(2).sum().item() for p in block.parameters())
        norms[f"block_{i}"] = math.sqrt(block_sq)
    norms["head"] = model.head.weight.data.norm().item()
    return norms


def effective_rank(matrix: torch.Tensor) -> float:
    s = torch.linalg.svdvals(matrix.float())
    s = s / (s.sum() + 1e-12)
    ent = -(s * torch.log(s + 1e-12)).sum().item()
    return float(math.exp(ent))


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    preds = logits.argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    correct_logits = logits[torch.arange(len(y), device=y.device), y]
    logits_masked = logits.clone()
    logits_masked[torch.arange(len(y), device=y.device), y] = -1e9
    margin = (correct_logits - logits_masked.max(dim=-1).values).mean().item()
    return {"loss": loss, "acc": acc, "margin": margin}


@torch.no_grad()
def predict_all(model: nn.Module, all_x: torch.Tensor) -> torch.Tensor:
    return model(all_x.to(DEVICE)).argmax(dim=-1).cpu()


def find_first_step(steps: np.ndarray, values: np.ndarray, threshold: float) -> Optional[int]:
    idx = np.where(values >= threshold)[0]
    return int(steps[idx[0]]) if len(idx) else None


def classify_snapshot(
    step: int,
    t_train_99: Optional[int],
    t_test_95: Optional[int],
    final_test_acc: float,
) -> str:
    if step == 0:
        return f"Step {step} (init)"
    if t_train_99 is not None and step >= t_train_99 and (t_test_95 is None or step < t_test_95):
        return f"Step {step} (memorized)"
    if t_test_95 is not None and step >= t_test_95:
        return f"Step {step} (generalized)"
    if final_test_acc < 0.95:
        return f"Step {step} (late training; no generalization)"
    return f"Step {step}"


@dataclass
class TrainConfig:
    train_frac: float = 0.35
    data_seed: int = 42
    model_seed: int = 42
    n_steps: int = 50000
    lr: float = 3e-4
    weight_decay: float = 0.1
    eval_every: int = 100
    probe_every: int = 250
    checkpoint_steps: Optional[List[int]] = None
    grad_clip: float = 1.0
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.0


def make_model(cfg: TrainConfig) -> GrokkingTransformer:
    return GrokkingTransformer(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(DEVICE)


def train_run(
    dataset: Dict,
    cfg: TrainConfig,
    collect_fourier: bool = True,
    verbose: bool = False,
):
    set_seed(cfg.model_seed)
    model = make_model(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
    )

    train_x = dataset["train_x"].to(DEVICE)
    train_y = dataset["train_y"].to(DEVICE)
    test_x = dataset["test_x"].to(DEVICE)
    test_y = dataset["test_y"].to(DEVICE)

    metrics = {
        "step": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "train_margin": [],
        "test_margin": [],
        "grad_norm": [],
    }
    probe_metrics = {
        "step": [],
        "norms": [],
        "emb_eff_rank": [],
        "fourier_structured_ratio": [],
        "fourier_entropy": [],
        "fourier_freq_energies": [],
        "fourier_top5_idx": [],
    }
    checkpoints = {}

    checkpoints[0] = {
        "emb": model.get_number_embeddings().clone(),
        "state_dict": copy.deepcopy(model.state_dict()),
    }

    checkpoint_steps = set(cfg.checkpoint_steps or [])

    iterator = range(1, cfg.n_steps + 1)
    if verbose:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc="Training")

    for step in iterator:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()

        if step % cfg.eval_every == 0 or step == 1 or step == cfg.n_steps:
            model.eval()
            tr = evaluate(model, train_x, train_y)
            te = evaluate(model, test_x, test_y)
            metrics["step"].append(step)
            metrics["train_loss"].append(tr["loss"])
            metrics["test_loss"].append(te["loss"])
            metrics["train_acc"].append(tr["acc"])
            metrics["test_acc"].append(te["acc"])
            metrics["train_margin"].append(tr["margin"])
            metrics["test_margin"].append(te["margin"])
            metrics["grad_norm"].append(grad_norm)

        if step % cfg.probe_every == 0 or step == 1 or step == cfg.n_steps:
            model.eval()
            probe_metrics["step"].append(step)
            probe_metrics["norms"].append(compute_norms(model))
            emb = model.get_number_embeddings()
            probe_metrics["emb_eff_rank"].append(effective_rank(emb))
            if collect_fourier:
                fe, sr, ent, t5 = fourier_energy(emb, FOURIER_BASIS)
                probe_metrics["fourier_freq_energies"].append(fe)
                probe_metrics["fourier_structured_ratio"].append(sr)
                probe_metrics["fourier_entropy"].append(ent)
                probe_metrics["fourier_top5_idx"].append(t5)

        if step in checkpoint_steps or step == cfg.n_steps:
            model.eval()
            checkpoints[step] = {
                "emb": model.get_number_embeddings().clone(),
                "state_dict": copy.deepcopy(model.state_dict()),
            }

    return model, metrics, probe_metrics, checkpoints


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def metrics_to_csv(metrics: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    pd.DataFrame(metrics).to_csv(path, index=False)


def pilot_score(row: Dict[str, float]) -> float:
    train_end = row["train_acc_end"]
    test_mid = row["test_acc_mid"]
    test_end = row["test_acc_end"]

    if train_end < 0.98:
        return -1e9
    if test_end > 0.90:
        return -1000.0 - test_end
    if test_end < 0.02:
        return -500.0 + test_end

    slope = max(test_end - test_mid, 0.0)
    band_bonus = -abs(test_end - 0.20)
    return 5.0 * slope + band_bonus + 0.1 * train_end


def summarize_pilot_history(metrics: Dict) -> Dict[str, float]:
    steps = np.array(metrics["step"])
    train_acc = np.array(metrics["train_acc"])
    test_acc = np.array(metrics["test_acc"])

    mid_target = 0.5 * steps[-1]
    mid_idx = int(np.argmin(np.abs(steps - mid_target)))

    return {
        "train_acc_end": float(train_acc[-1]),
        "test_acc_end": float(test_acc[-1]),
        "train_acc_mid": float(train_acc[mid_idx]),
        "test_acc_mid": float(test_acc[mid_idx]),
        "train_acc_start": float(train_acc[0]),
        "test_acc_start": float(test_acc[0]),
    }


plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)


def save_fig(fig, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pilot_heatmaps(pilot_df: pd.DataFrame, train_fracs, weight_decays, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    items = [
        ("train_acc_end", "Train Accuracy @ pilot end"),
        ("test_acc_end", "Test Accuracy @ pilot end"),
        ("score", "Pilot Selection Score"),
    ]
    for ax, (col, title) in zip(axes, items):
        data = np.zeros((len(weight_decays), len(train_fracs)))
        for i, wd in enumerate(weight_decays):
            for j, frac in enumerate(train_fracs):
                rows = pilot_df[(pilot_df.train_frac == frac) & (pilot_df.weight_decay == wd)]
                data[i, j] = rows[col].mean()
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(train_fracs)))
        ax.set_xticklabels([f"{f:.0%}" for f in train_fracs])
        ax.set_yticks(range(len(weight_decays)))
        ax.set_yticklabels([str(w) for w in weight_decays])
        ax.set_xlabel("Train Fraction")
        ax.set_ylabel("Weight Decay")
        ax.set_title(title)
        for i in range(len(weight_decays)):
            for j in range(len(train_fracs)):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "pilot_heatmaps.png"))


def plot_training_curves(metrics: Dict, t_train_99, t_test_95, out_dir: str, title_suffix: str = ""):
    steps = np.array(metrics["step"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(steps, metrics["train_acc"], label="Train Accuracy", lw=1.5)
    ax.plot(steps, metrics["test_acc"], label="Held-out Accuracy", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    if t_train_99 is not None:
        ax.axvline(t_train_99, color="blue", ls="--", alpha=0.5, label=f"Train 99% @ {t_train_99}")
    if t_test_95 is not None:
        ax.axvline(t_test_95, color="orange", ls="--", alpha=0.5, label=f"Held-out 95% @ {t_test_95}")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, metrics["train_loss"], label="Train Loss", lw=1.5)
    ax.plot(steps, metrics["test_loss"], label="Held-out Loss", lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"Training Curves {title_suffix}".strip(), fontsize=15)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "training_curves.png"))


def plot_norms(metrics: Dict, probe_metrics: Dict, t_train_99, t_test_95, out_dir: str):
    steps = np.array(metrics["step"])
    probe_steps = np.array(probe_metrics["step"])

    total_norms = [n["total"] for n in probe_metrics["norms"]]
    tok_norms = [n["tok_emb"] for n in probe_metrics["norms"]]
    head_norms = [n["head"] for n in probe_metrics["norms"]]
    b0 = [n.get("block_0", np.nan) for n in probe_metrics["norms"]]
    b1 = [n.get("block_1", np.nan) for n in probe_metrics["norms"]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax in axes.flat:
        if t_train_99 is not None:
            ax.axvline(t_train_99, color="blue", ls="--", alpha=0.4)
        if t_test_95 is not None:
            ax.axvline(t_test_95, color="orange", ls="--", alpha=0.4)

    axes[0, 0].plot(probe_steps, total_norms, color="black")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Total Parameter Norm")
    axes[0, 0].grid(True, alpha=0.3)

    for vals, lbl in [(tok_norms, "Token Emb"), (b0, "Block 0"), (b1, "Block 1"), (head_norms, "Output Head")]:
        axes[0, 1].plot(probe_steps, vals, label=lbl)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Per-Component Norms")
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, metrics["grad_norm"], alpha=0.8)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Gradient Norm")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(probe_steps, probe_metrics["emb_eff_rank"], color="purple")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("Embedding Effective Rank")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "norms.png"))


def plot_fourier(metrics: Dict, probe_metrics: Dict, t_train_99, t_test_95, out_dir: str):
    if len(probe_metrics["fourier_structured_ratio"]) == 0:
        return
    steps = np.array(metrics["step"])
    probe_steps = np.array(probe_metrics["step"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(probe_steps, probe_metrics["fourier_structured_ratio"], color="crimson")
    axes[0].set_xscale("log")
    axes[0].set_title("Top-5 Fourier Energy / Total Energy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(probe_steps, probe_metrics["fourier_entropy"], color="darkgreen")
    axes[1].set_xscale("log")
    axes[1].set_title("Fourier Frequency Entropy")
    axes[1].grid(True, alpha=0.3)

    all_energies = np.array(probe_metrics["fourier_freq_energies"])
    final_top5 = probe_metrics["fourier_top5_idx"][-1]
    for idx in final_top5:
        axes[2].plot(probe_steps, all_energies[:, idx], label=f"Freq {idx+1}")
    axes[2].set_xscale("log")
    axes[2].set_title("Top-5 Fourier Energies")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        if t_train_99 is not None:
            ax.axvline(t_train_99, color="blue", ls="--", alpha=0.4)
        if t_test_95 is not None:
            ax.axvline(t_test_95, color="orange", ls="--", alpha=0.4)

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "fourier.png"))

    fig2, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(steps, metrics["test_acc"], color="tab:orange", label="Held-out Accuracy")
    ax1.set_xscale("log")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Held-out Accuracy", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:orange")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(probe_steps, probe_metrics["fourier_structured_ratio"], color="crimson", label="Fourier Structured Ratio")
    ax2.set_ylabel("Structured Ratio", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left")
    ax1.set_title("Held-out accuracy vs Fourier structure")
    fig2.tight_layout()
    save_fig(fig2, os.path.join(out_dir, "fourier_vs_accuracy.png"))


def plot_pca(checkpoints: Dict, snapshot_steps: List[int], labels: List[str], out_dir: str):
    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(6 * len(snapshot_steps), 5))
    if len(snapshot_steps) == 1:
        axes = [axes]
    for ax, step, label in zip(axes, snapshot_steps, labels):
        emb = checkpoints[step]["emb"].numpy()
        coords = PCA(n_components=2).fit_transform(emb)
        ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(P), cmap="hsv", s=22, alpha=0.8)
        ax.set_title(label)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "embedding_pca.png"))


def plot_operation_tables(
    dataset: Dict,
    checkpoints: Dict,
    snapshot_steps: List[int],
    labels: List[str],
    cfg: TrainConfig,
    out_dir: str,
):
    all_y = dataset["all_y"].numpy()
    train_mask = dataset["train_mask"]
    test_mask = dataset["test_mask"]
    side = P

    fig, axes = plt.subplots(2, len(snapshot_steps), figsize=(6 * len(snapshot_steps), 10))
    if len(snapshot_steps) == 1:
        axes = np.array(axes).reshape(2, 1)

    cmap = mcolors.ListedColormap(["#d32f2f", "#4caf50"])
    for col, (step, label) in enumerate(zip(snapshot_steps, labels)):
        temp_model = make_model(cfg)
        temp_model.load_state_dict(checkpoints[step]["state_dict"])
        temp_model.eval()
        preds = predict_all(temp_model, dataset["all_x"])
        correct = (preds.numpy() == all_y)

        train_correct = np.full_like(correct, fill_value=False, dtype=bool)
        held_correct = np.full_like(correct, fill_value=False, dtype=bool)
        train_correct[train_mask] = correct[train_mask]
        held_correct[test_mask] = correct[test_mask]

        train_panel = np.zeros_like(correct, dtype=int)
        held_panel = np.zeros_like(correct, dtype=int)
        train_panel[train_mask] = train_correct[train_mask].astype(int)
        held_panel[test_mask] = held_correct[test_mask].astype(int)

        axes[0, col].imshow(train_panel.reshape(side, side), cmap=cmap, interpolation="nearest", aspect="equal")
        axes[0, col].set_title(f"{label}\nTrain-only")
        axes[1, col].imshow(held_panel.reshape(side, side), cmap=cmap, interpolation="nearest", aspect="equal")
        held_acc = correct[test_mask].mean()
        axes[1, col].set_title(f"{label}\nHeld-out only (acc={held_acc:.1%})")

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "operation_tables_split.png"))


def plot_phase_diagram(grouped_df, fracs, wds, max_steps: int, out_dir: str):
    """
    grouped_df must contain columns:
      - train_frac
      - weight_decay
      - mean_tau_g
      - std_tau_g
      - majority_status
      - grokking_successes
      - n_runs
    """
    import numpy as np
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    # Numeric heatmap uses mean tau_g
    phase = np.full((len(wds), len(fracs)), np.nan, dtype=float)
    regime = np.zeros((len(wds), len(fracs)), dtype=int)
    annotations = [["" for _ in fracs] for _ in wds]

    status_to_regime = {
        "no_memorization": 0,
        "immediate_generalization": 1,
        "delayed_generalization": 2,
        "no_generalization": 3,
    }

    for i, wd in enumerate(wds):
        for j, frac in enumerate(fracs):
            row = grouped_df[
                (grouped_df["train_frac"] == frac) &
                (grouped_df["weight_decay"] == wd)
            ]
            if len(row) == 0:
                continue

            row = row.iloc[0]
            phase[i, j] = row["mean_tau_g"]
            regime[i, j] = status_to_regime.get(row["majority_status"], 0)

            mean_tau = row["mean_tau_g"]
            std_tau = row["std_tau_g"]
            succ = int(row["grokking_successes"])
            n_runs = int(row["n_runs"])
            status = row["majority_status"]

            if status == "no_memorization":
                tau_txt = "N/M"
            elif status == "no_generalization":
                tau_txt = "N/G"
            else:
                tau_txt = f"{int(mean_tau)}"

            std_txt = "NA" if np.isnan(std_tau) else f"{int(std_tau)}"
            annotations[i][j] = f"{tau_txt}\n{succ}/{n_runs} seeds\n±{std_txt}"

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: mean tau_g heatmap
    ax = axes[0]
    dd = phase.copy()
    dd[np.isnan(dd)] = np.nan

    valid = dd[np.isfinite(dd) & (dd > 0)]
    if len(valid) > 0:
        vmin = max(1, np.nanmin(valid))
        vmax = max_steps
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(dd, cmap="magma_r", aspect="auto", norm=norm)
    else:
        im = ax.imshow(np.nan_to_num(dd, nan=0.0), cmap="magma_r", aspect="auto")

    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{f:.0%}" for f in fracs])
    ax.set_yticks(range(len(wds)))
    ax.set_yticklabels([str(w) for w in wds])
    ax.set_xlabel("Train Fraction")
    ax.set_ylabel("Weight Decay")
    ax.set_title("Mean Grokking Delay τ_g")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean τ_g")

    for i in range(len(wds)):
        for j in range(len(fracs)):
            txt = annotations[i][j]
            val = phase[i, j]
            color = "white" if (np.isfinite(val) and val > max_steps / 4) else "black"
            if "N/M" in txt or "N/G" in txt:
                color = "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    # Right: regime map
    ax = axes[1]
    cmap = mcolors.ListedColormap(["#9e9e9e", "#4caf50", "#ff9800", "#d32f2f"])
    ax.imshow(regime, cmap=cmap, aspect="auto", vmin=0, vmax=3)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{f:.0%}" for f in fracs])
    ax.set_yticks(range(len(wds)))
    ax.set_yticklabels([str(w) for w in wds])
    ax.set_xlabel("Train Fraction")
    ax.set_ylabel("Weight Decay")
    ax.set_title("Majority Regime")

    short_status = {
        "no_memorization": "N/M",
        "immediate_generalization": "IMM",
        "delayed_generalization": "DELAY",
        "no_generalization": "N/G",
    }

    for i, wd in enumerate(wds):
        for j, frac in enumerate(fracs):
            row = grouped_df[
                (grouped_df["train_frac"] == frac) &
                (grouped_df["weight_decay"] == wd)
            ]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            succ = int(row["grokking_successes"])
            n_runs = int(row["n_runs"])
            label = short_status.get(row["majority_status"], "?")
            ax.text(j, i, f"{label}\n{succ}/{n_runs}", ha="center", va="center", fontsize=9, color="black")

    ax.legend(
        handles=[
            Patch(facecolor="#9e9e9e", label="No memorization"),
            Patch(facecolor="#4caf50", label="Immediate gen."),
            Patch(facecolor="#ff9800", label="Delayed gen."),
            Patch(facecolor="#d32f2f", label="No generalization"),
        ],
        loc="upper left",
        fontsize=9,
    )

    fig.suptitle("Local Ablation Around the Grokking Regime", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "phase_diagram.png"))