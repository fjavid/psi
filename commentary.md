# Grokking in Neural Networks: Commentary

## Experimental Setup

We study grokking on modular addition mod 97, where each input is the four-token sequence `a + b =` and the target is `(a + b) mod 97`. The dataset enumerates all 9,409 pairs in Z_97^2, and we hold out a fraction for testing. This is a finite exhaustive universe: we evaluate generalization exactly on the entire held-out operation table, not on a sample.

The shared implementation (`grokking_common.py`) uses a small decoder-only transformer: learned token and position embeddings, two attention blocks, width 128, four heads, feedforward width 512, and a linear classifier over the 97 residue classes. Training uses AdamW with betas (0.9, 0.98) and gradient clipping at 1.0.

The experiment is organized as a three-stage pipeline:

1. **Pilot sweep** (`grokking_pilot.py`): searches a grid of train fractions [0.25, 0.30, 0.35, 0.40, 0.50] and weight decays [0.0, 0.01, 0.03, 0.1, 0.3, 1.0] across two seeds at 10,000 steps each. Configs are scored by a composite metric that rewards high train accuracy, nontrivial but not immediate test accuracy, and a positive test-accuracy trend. The pilot selected `frac=0.4, wd=0.03` as the most promising candidate.

2. **Main run** (`grokking_main.py`): a single long diagnostic run with dense metric and checkpoint logging. After the pilot, we increased weight decay to 0.3 based on evidence that 0.03 was too weak to force generalization within practical budgets. The final config: `train_frac=0.4, weight_decay=0.3, lr=3e-4, steps=100,000`.

3. **Ablation** (`grokking_ablation.py`): maps grokking delay across a grid of train fractions and weight decays to produce a phase diagram showing where the model generalizes immediately, groks with delay, or fails to generalize.

## Evidence of Grokking

The main run with `frac=0.4, wd=0.3` exhibits textbook grokking:

- **Rapid memorization**: Train accuracy reaches 99% at step 200.
- **Long plateau**: Test accuracy stays low for tens of thousands of steps while training loss continues to decrease to near zero.
- **Sudden generalization**: Test accuracy crosses 95% at step 43,200 and reaches 100% shortly after.
- **Grokking delay**: tau_g = 43,000 steps — the model trains for over 200x longer after memorization before it generalizes.

The training curves confirm this pattern clearly. Train loss drops to effectively zero within the first few hundred steps, while test loss initially rises (the model becomes confidently wrong on held-out data) before eventually dropping sharply around the same time test accuracy jumps.

The operation-table plots make this especially vivid: at the memorization checkpoint, the train-set panel is entirely green (all training pairs correct) while the held-out panel is almost entirely red. At the final checkpoint, both panels are fully green — the model has learned the complete addition table.

## Internal Dynamics

### Weight Norms

Total parameter norm rises during the memorization phase, then steadily declines as weight decay compresses the solution. Per-component norms show that the token embedding and transformer blocks all participate in this compression. The output head norm follows a similar trajectory. This is consistent with the view that weight decay gradually penalizes the high-norm memorizing solution, creating pressure toward a more efficient representation.

### Gradient Norm

Gradient norms are initially large during rapid memorization, then drop as the model saturates on training data. During the long plateau, gradients remain small but nonzero — the model is still changing internally even though external metrics appear stagnant. A slight increase in gradient activity is visible around the generalization transition.

### Embedding Effective Rank

The effective rank of the number-embedding matrix, measured via the entropy of its singular values, remains relatively stable throughout training. This suggests that the transition to generalization is not primarily about dimensionality reduction in the embeddings, but rather about how the existing dimensions are used.

### Fourier Concentration

Because modular arithmetic over Z_97 has natural cyclic structure, Fourier modes provide a principled basis for probing whether the embeddings are aligning with the algebraic structure of the task.

The Fourier structured ratio (fraction of total embedding energy in the top-5 frequencies) increases from about 0.117 at initialization to about 0.234 by the end of training. This is a meaningful doubling, and importantly, the increase is gradual throughout the plateau phase — it does not happen only at the moment of the test-accuracy jump.

The Fourier entropy decreases correspondingly, from about 3.87 to 3.64, indicating that energy is concentrating into fewer frequencies over time.

The overlay of Fourier structured ratio against test accuracy shows that Fourier concentration begins increasing well before the sharp test-accuracy jump. This is consistent with the hypothesis that the model gradually builds a structured representation of modular arithmetic during the plateau, and the external jump in test performance occurs only when the structured circuit becomes strong enough to dominate predictions on held-out data.

### Embedding PCA

PCA snapshots of the number embeddings at three checkpoints (initialization, post-memorization, post-grokking) show a progression from unstructured to organized geometry. The post-grokking embeddings exhibit more regular spatial arrangement, consistent with the Fourier analysis showing increasing alignment with the cyclic structure of Z_97.

## Pilot Sweep Results

The pilot heatmaps reveal a clear landscape:

- **No weight decay (wd=0.0)**: The model memorizes perfectly but test accuracy stays very low across all train fractions. Without regularization pressure, there is no incentive to find a generalizing solution.
- **Weak weight decay (wd=0.01-0.03)**: Memorization is still fast, and test accuracy shows a slow upward trend, but generalization is incomplete within 10,000 pilot steps.
- **Moderate weight decay (wd=0.1-0.3)**: The most interesting regime. The model memorizes and shows stronger test-accuracy trends, suggesting it is in the zone where grokking can occur given sufficient training budget.
- **Strong weight decay (wd=1.0)**: With 50% training data, the model generalizes almost immediately (test accuracy reaches 100% within 10,000 steps). With smaller train fractions, strong decay can slow memorization itself.

The pilot scoring function was designed to prefer configs in the intermediate zone: high train accuracy, nontrivial but not yet saturated test accuracy, and a positive trend. This successfully identified `frac=0.4` as the most promising train fraction.

## Hypothesis

The model first finds a memorizing solution — an unstructured mapping that achieves perfect training accuracy by essentially storing the training pairs. This solution has high parameter norm. Weight decay then gradually penalizes this high-norm solution, creating optimization pressure toward lower-norm alternatives. Simultaneously, the model builds a structured Fourier representation of modular arithmetic in its embeddings. The test-accuracy jump occurs when the structured circuit becomes strong enough, and the memorizing components have been sufficiently suppressed, for the generalizing solution to dominate predictions on unseen pairs.

This is supported by:

- The steady decline in parameter norms throughout the plateau (weight decay compressing the memorizing solution).
- The gradual increase in Fourier concentration during the plateau (structured representation forming before the external jump).
- The sharp test-accuracy transition occurring long after memorization but coinciding with substantial Fourier reorganization.
- The pilot sweep showing that weight decay is the critical control parameter: without it, the model never generalizes; with too much, it generalizes immediately.

This picture is consistent with the circuit-formation-and-cleanup view (Nanda et al.), the lazy-to-rich transition framework, and the Goldilocks-zone perspective on grokking. Fourier concentration appears to be a more informative predictor of impending generalization than raw norm decay alone, since the structured ratio moves gradually throughout the plateau while norms decline smoothly without a clear inflection aligned to the generalization transition.
