# `gmdhtorch` — Group Method of Data Handling in PyTorch

**`gmdhtorch`** is a **GMDH (Group Method of Data Handling)** library in **Python & PyTorch** (`src/gmdhtorch/`), supporting both classical constructive closed-form polynomial networks (O-GMDH / C-GMDH) and differentiable hybrid/solver-free PyTorch models.

Companion benchmarks, reinforcement-learning scripts, and fixed-point Verilog HDL workflows live in [alsbraga/gmdh_experiments](https://github.com/alsbraga/gmdh_experiments).

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/alsbraga/gmdhtorch.git"
```

> **PyTorch / GPU note:** `gmdhtorch` depends on `torch>=2.0`. If you need a specific CUDA build of PyTorch, install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/) and then install `gmdhtorch`.

For local development (including test dependencies):

```bash
git clone https://github.com/alsbraga/gmdhtorch.git
cd gmdhtorch
pip install -e ".[dev]"
pytest
```

## Quickstart

### Scikit-Learn API (`GMDHRegressor`)

```python
import numpy as np
from gmdhtorch import GMDHRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-1.0, 1.0, size=(300, 4))
y = X[:, 0] ** 2 + 0.5 * X[:, 1] * X[:, 2] - 0.3 * X[:, 3]

reg = GMDHRegressor(max_layers=5, max_layer_neurons=20, max_degree=2, shuffle_data=True, random_state=0)
reg.fit(X, y)
preds = reg.predict(X[:5])
```

### PyTorch API (`GMDHModel` & `ModelIncrementer`)

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from gmdhtorch import GMDHModel, ModelIncrementer, default_layers

torch.manual_seed(0)
X = torch.rand(300, 4, dtype=torch.float64) * 2 - 1
y = (X[:, 0] ** 2 + 0.5 * X[:, 1] * X[:, 2] - 0.3 * X[:, 3]).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X[:200], y[:200]), batch_size=64)
select_loader = DataLoader(TensorDataset(X[200:], y[200:]), batch_size=64)

model = GMDHModel(dtype=torch.float64)
inc = ModelIncrementer(model, layers=default_layers(inputs=2, degree=2), max_layers=5)
while inc.growing:
    for xb, yb in train_loader:
        inc.observe(yb, x=xb)
    inc.step(select_loader)

model.trim()
preds = model(X[:5])
```

## License

BSD 3-Clause License — see `LICENSE` for details.

---

## Architecture & Technical Reference

The sections below document the historical context, mathematical foundations, module design, constructive growth lifecycle, and ongoing architectural objectives.

## 1. History, Context & Motivation

### 1.1 Origin: The GMDH Box
The Group Method of Data Handling was introduced by Alexey G. Ivakhnenko (1968–1971) as an inductive self-organizing modeling technique. Inspired by the Gabor-Kolmogorov polynomial representation of complex systems, Ivakhnenko proposed generating networks layer-by-layer:
1. Combining pairs of inputs into quadratic candidate neurons.
2. Estimating weights using closed-form linear regression on a training partition.
3. Selecting survivor neurons using an external criterion on a separate validation/selection partition.
4. Repeating until the selection error stops decreasing, followed by backward trimming to prune unneeded neurons.

This implementation originated as a doctoral research library in MATLAB/GNU Octave (`GMDH Box`, documented in [this report](http://repositorio.unb.br/handle/10482/13365)). The original codebase explored several departures from classic GMDH:
- **Forward nodes (*expedidores*)**: Passing inputs and preceding layer activations directly into deeper layers to preserve raw features and bypass vanishing signals.
- **Higher combination arities ($k > 2$)**: Grouping inputs in tuples larger than pairs.
- **Higher-degree polynomials ($d \ge 1$)**: Supporting quadratic, cubic, or arbitrary polynomial expansions.
- **Modern numerical solvers**: Mitigating the numerical ill-conditioning of normal equations through Singular Value Decomposition (SVD) and QR factorizations.

### 1.2 Evolution to Pure PyTorch (`torchonly`)
The codebase was migrated to a pure PyTorch package (`src/gmdhtorch/`) with the following core architectural milestones:
1. **Differentiable Layers as First-Class PyTorch Modules**: Every `GMDHLayer` is an `nn.Module` containing registered parameter tensors (`W`) and connectivity buffers (`idx0`), enabling standard PyTorch backpropagation (`loss.backward()` and `optimizer.step()`).
2. **Decoupled Polynomial Geometry**: Decoupling the input arity ($k$), polynomial degree ($d$), and interaction degree ($q$).
3. **Numerically Stabilized Closed-Form Solvers**: Float64-accumulated QR factorizations, column-pivoted QR, SVD with condition thresholds (`rcond`), Tikhonov regularization (Ridge with GCV), and column $L_2$ equilibration.
4. **Inter-Layer Normalization & Activations**: Inserting frozen/affine standardization (`GMDHNorm`) and bounded non-linearities (`SoftClip`, `ReLU`, `Tanh`, `GELU`) between layers, preventing numerical explosion across deep polynomial compositions.
5. **Hybrid Training**: Constructive layer-by-layer growth interleaved with simultaneous gradient fine-tuning (`gradient_during_growth=True`) via `ModelIncrementer`.

---

## 2. Mathematical Foundations

### 2.1 The Volterra / Kolmogorov-Gabor Polynomial
The Kolmogorov-Gabor polynomial describes an arbitrary continuous non-linear relationship between input vector $\mathbf{x} \in \mathbb{R}^D$ and scalar target $y$:

$$y = w_0 + \sum_{i=1}^D w_i x_i + \sum_{i=1}^D \sum_{j=i}^D w_{ij} x_i x_j + \sum_{i=1}^D \sum_{j=i}^D \sum_{k=j}^D w_{ijk} x_i x_j x_k + \dots$$

Rather than fitting a single intractable high-order polynomial over all $D$ inputs, GMDH decomposes the mapping into a cascade of sparse low-arity nodes.

### 2.2 Decoupled Polynomial Formulation
For any neuron receiving $k$ inputs $\mathbf{u} = (u_1, \dots, u_k) \in \mathbb{R}^k$, the polynomial expansion is defined by three decoupled parameters:
1. **$k$ (`inputs` / `input_count`)**: Receptive field arity (number of input features combined).
2. **$d$ (`degree`)**: Maximum power for univariate terms ($u_i, u_i^2, \dots, u_i^d$).
3. **$q$ (`interaction`)**: Maximum order of multi-variable cross-terms ($u_i u_j$, $u_i u_j u_m$, etc.), where $0 \le q \le k$.

The total number of coefficients $`P = \text{coeff\_count}(k, d, q)`$ is computed exactly by:

$$P = 1 + k \cdot d + \sum_{m=2}^{\min(k, q)} \binom{k}{m}$$

- If $q = 0$: Pure additive powers (no cross-terms): $P = 1 + k \cdot d$.
- If $d = 1, q = 2$: Pairwise interaction model (affine + cross-products): $P = 1 + k + \binom{k}{2}$.
- If $k = 2, d = 2, q = 2$: Classic Ivakhnenko quadratic neuron: $P = 1 + 2(2) + 1 = 6$ terms ($w_0 + w_1 u_1 + w_2 u_2 + w_3 u_1^2 + w_4 u_2^2 + w_5 u_1 u_2$).

### 2.3 3D Batched Polynomial Preprocessing
To evaluate or fit $M$ candidate neurons across $N$ samples on GPU/CPU without per-neuron Python loops:
1. `precompute_layer_powers_torch(X, d)` evaluates all powers $X^1, X^2, \dots, X^d$ in a single tensor of shape $(N, \text{features}, d)$.
2. Gathered powers are indexed by candidate indices $\text{idx0} \in \mathbb{Z}^{M \times k}$.
3. Pairwise cross-terms are assembled using cached index buffers (`_comb_2`, `_comb_3`, etc.).
4. The output design tensor $\mathbf{X}_{\text{batch}}$ has shape $(M, N, P)$, where:
   - $M$: Number of candidate neurons in the layer.
   - $N$: Number of samples in the batch.
   - $P$: Number of polynomial coefficients per neuron.

During forward evaluation, einsum operations compute all neuron activations simultaneously:
$$\mathbf{H}_{\text{raw}} = \text{einsum}("mnc, mc \to nm", \mathbf{X}_{\text{batch}}, \mathbf{W}) \in \mathbb{R}^{N \times M}$$

### 2.4 Inter-Layer Activations, Stability & Affine GMDH for Reinforcement Learning
In purely polynomial GMDH, composing degree-$d$ layers across depth $L$ grows the effective polynomial degree exponentially (up to $d^L$). Even moderate input excursions or iterative gradient updates can cause intermediate values and gradients to explode across layers.

To address this, the library extends the polynomial cascade with optional inter-layer activation functions $\phi(\cdot)$ (`GMDHActivation` wrapping bounded or piecewise-linear non-linearities such as `SoftClip`, `nn.Hardtanh`, `nn.ReLU`, `nn.Tanh`, or `nn.GELU`):

$$\mathbf{H}^{(l)} = \phi\left(\mathbf{H}_{\text{raw}}^{(l)}\right)$$

(with locked forwarders bypassing $\phi$ via `passthrough_mask`). Bounded activations such as `SoftClip` and `nn.Hardtanh` directly cap the dynamic range between layers, stabilizing both forward propagation and backpropagation—even when `GMDHNorm` is omitted.

With non-linearity provided by $\phi$, GMDH can also be configured in a **sparse affine regime** ($k = 2, d = 1, q = 0$; $P = 3$ weights per neuron: $w_0 + w_1 u_1 + w_2 u_2$). In reinforcement learning (e.g., DQN on `CartPole-v1` in [`gmdh_experiments/reinforcement.py`](https://github.com/alsbraga/gmdh_experiments)), pairing `degree=1, interaction=0` with bounded activations such as `nn.Hardtanh` avoids the numerical sensitivity of quadratic/cross-product terms under non-stationary Bellman targets while retaining GMDH's constructive layer growth, uniform fan-in $k = 2$, and cascade pruning (`trim()`). This produces highly compact Q-networks that also map cleanly to fixed-point quantization and hardware (HDL) synthesis.

### 2.5 Solver-Free Weight Initialization
Classic GMDH fits every candidate neuron via closed-form least-squares solvers (SVD, QR, or Ridge) directly against supervised targets. In settings where per-neuron least-squares targets are unavailable or non-stationary—such as reinforcement learning (TD/Bellman losses), policy gradients, or end-to-end composite neural networks—the library also supports **solver-free weight initialization** (`solver=None`):

- Instead of invoking a closed-form solver, `GMDHLayer` materializes sparse candidate connections (`idx0`) and initializes polynomial weights $\mathbf{W}$ using standard neural-network initialization schemes (`init="kaiming" | "xavier" | "normal" | "zeros"`, with zeroed intercepts $w_0 = 0$ and optional `init_std`).
- Weights are then optimized end-to-end via backpropagation (`AdamW`, `SGD`, etc.), while `ModelIncrementer.grow(...)` preserves constructive GMDH topology discovery through importance-based neuron selection (`GMDHOutput.importance_scores()`) and backward cascade pruning (`trim()`).

---

## 3. Core Package Architecture (`src/gmdhtorch/`)

```
src/gmdhtorch/
├── polynomial.py    # Combinatorial indexing & 3D batched polynomial tensor generation
├── network.py       # PyTorch modules: GMDHModel, GMDHLayer, GMDHNorm, GMDHActivation, SoftClip, GMDHOutput
├── solvers.py       # Batched least-squares solvers (SVD, QR, Ridge, GCV) & StreamedQRAccumulator
├── selection.py     # Layer selection strategies (SelectMedian, SelectBest, SelectThreshold, Gauss)
├── train.py         # ModelIncrementer lifecycle, constructive growth, rollback, and tuning
├── metrics.py       # Evaluation metrics: MSE, MAE, NRMSE, R2
├── sklearn_api.py   # Scikit-learn estimators: GMDHRegressor, GMDHClassifier
└── experimental.py # Isolated research extensions (e.g., PSO weight perturbation)
```

### 3.1 Network Modules (`src/gmdhtorch/network.py`)

#### `GMDHModel(nn.Sequential)`
The top-level container inheriting from `nn.Sequential`. It orchestrates:
- **Closed Validated Module Set**: Explicitly validates modules against `_ALLOWED_GMDH_MODULES` (`GMDHLayer`, `GMDHNorm`, `GMDHActivation`, `SoftClip`, `GMDHOutput`, `nn.Linear`, and standard PyTorch activations such as `nn.ReLU`, `nn.Tanh`, `nn.GELU`, `nn.SiLU`, `nn.LeakyReLU`, `nn.Sigmoid`, `nn.ELU`, `nn.Identity`).
- **Flexible Output Head (`output="auto" | "none" | GMDHOutput`)**: Supports selector `GMDHOutput(dense=False)`, linear-aggregator `GMDHOutput(dense=True)`, manual `nn.Linear`, or headless termination directly in an eager `GMDHLayer` (e.g., for latent encoders).
- **Lazy Input Inference**: Discovers input dimensionality $D$ from the first observed batch when `input_count=None`.
- **Structural Pruning (`prune(position, keep)`)**: Slices the layer at `position`, updates downstream connection indices `idx0`, and resizes intervening normalization/activation modules as well as terminal `GMDHOutput(dense=True)` / `nn.Linear` heads.
- **Cascade Trimming (`trim()`)**: Traces active connectivity backward from `GMDHOutput.index` (or from all active neurons of the final `GMDHLayer` for `dense=True` / headless models), recursively pruning unreferenced neurons back to the input layer.
- **Research Instrumentation**:
  - `parameter_count(trainable_only=True)`: Active polynomial weights (excluding locked forwarders).
  - `mac_count()`: Multiply-accumulate operations per inference pass.
  - `connectivity_stats()`: Layer dimensions, active input indices, fan-in distributions.
  - `gradient_norms()`: L2 gradient norm per layer during fine-tuning.
  - `describe()`: Formatted ASCII diagram of connectivity and weights.

#### `GMDHLayer(nn.Module)`
Represents an individual sparse polynomial perceptron layer, supporting three creation paths:
- **Template State** (`neuron_count == 0`): Stores hyperparameter specifications (`inputs`, `degree`, `interaction`, `forward`, `max_neurons`, `init`) for `ModelIncrementer`.
- **Eager Solver-Free Materialization** (`in_features=...`): Resolves `idx0` immediately (capped by `out_features` / `max_neurons`, optionally ranked by `input_scores`) and initializes `W` with zeroed intercept (`W[:, 0] = 0.0`) via `"kaiming"`, `"xavier"`, `"normal"`, or `"zeros"`, ready for backpropagation on Step 0.
- **Materialized State** (`neuron_count > 0`): Owns parameter tensor $`W \in \mathbb{R}^{M \times P}`$ and integer buffer $`idx0 \in \mathbb{Z}^{M \times k}`$.
- **Locked Forwarders (*expedidores*)**: Channels where `lock_mask[i] == True`. A registered backward hook zeroes out gradients for locked forwarders, ensuring they remain strict identity pass-throughs.
- **Multi-Output Target Tagging**: $`\text{target\_id} \in \mathbb{Z}^M`$ associates each neuron with its assigned target channel $`j \in \{0, \dots, C-1\}`$.

#### `GMDHNorm(nn.Module)`
Channel-wise standardization placed between polynomial layers:
$$z_{\text{norm}} = \frac{z - \mu}{\sigma}$$
- Stores running or closed-form statistics in `mean` and `std` buffers.
- Optional learnable affine parameters (`affine=True`: $\gamma z_{\text{norm}} + \beta$).
- Channels flagged by `passthrough_mask` (forwarders) bypass normalization to preserve raw signals.

#### `GMDHActivation(nn.Module)`
Wraps an arbitrary activation module (`SoftClip`, `nn.ReLU`, `nn.Tanh`, `nn.GELU`). Channels flagged by `passthrough_mask` bypass activation to prevent forwarder saturation.

#### `SoftClip(nn.Module)`
Smooth symmetric saturating activation:
$$\text{SoftClip}(z; c) = c \cdot \tanh(z / c)$$
Linear around the origin ($z \approx 0$) with smooth asymptotic saturation at $\pm c$, preventing explosive dynamic range across polynomial compositions without vanishing gradients.

#### `GMDHOutput(nn.Module)`
Terminal output head supporting two modes:
- **`dense=False` (Selector Mode)**: Parameter-free index routing ($`\text{index} \in \mathbb{Z}^C`$) picking the best-performing neuron(s) to output ($y = H[:, \text{index}]$).
- **`dense=True` (Linear Combiner Mode)**: Wraps an `nn.Linear(in_features, count)` projection aggregating all active neurons of the final `GMDHLayer`, with `adapt_width()`, `resize_warm()` (function-preserving weight copy onto forwarded channels + zero-init on new candidates), `prune()`, and scale-invariant `importance_scores()` ($`\|W_{:, j}\|_2 \cdot \text{RMS}(H_{:, j})`$).
- Before Layer 1 (or linear head width) is materialized, outputs a registered `fallback` buffer (zeros or target mean).

---

## 4. Constructive Growth & Fine-Tuning Lifecycle (`src/gmdhtorch/train.py`)

### 4.1 Generic `ModelIncrementer` & State Machine
`ModelIncrementer` accepts any `nn.Module`:
- **No-Op Mode**: If no `GMDHModel` exists inside `model` (e.g., a standard MLP), `ModelIncrementer` acts as a safe pass-through (`growing=False`, `stopped=True`), eliminating custom null wrappers.
- **Composite Auto-Discovery**: If `model` wraps a single `GMDHModel` (e.g., `nn.Sequential(upstream, gmdh_head)`), `ModelIncrementer` discovers it via `model.modules()` and captures local inputs via forward pre-hooks so raw `(x, y)` loaders work directly.
- **`SelectionStrategy` Placement**: Neuron selection policies (`select`, `final_select`, including `SelectRelativeThreshold` and `higher_is_better=True` importance ranking) live on `ModelIncrementer` (overridable per step via `plan_next(select=...)`).
- **Pre-Materialization `max_neurons` Capping**: When $`\binom{n_{\text{prev}}}{k} > \text{max\_neurons}`$, candidate combinations are capped **before** solving or allocating `W` via lazy $O(M \log M)$ performance-ranked generation (`resolve_candidate_indices()`), with a 50% forwarder budget cap when `forward=True`.

Four discrete states manage the growth lifecycle:

```
                  ┌──────────────┐
                  │   "stats"    │ (Cycle 0: input normalization pass, optional)
                  └──────┬───────┘
                         │
                         ▼
┌──────────────┐ step()/ ┌──────────────┐  stop()   ┌────────────┐
│   "tuning"   │◄─grow()─┤ "accumulate" ├──────────►│ "stopped"  │
└──────┬───────┘ (done)  └──────┬───────┘ (manual)  └────────────┘
       │                        │
       │                        │ remove_last_layer()
       ▼                        ▼
  End Training             Rollback Layer
```

### 4.2 Closed-Form vs. Solver-Free Piecewise Growth
- **Closed-Form Growth (`solver="svd" | "qr" | "ridge"`)**:
  1. `observe(yb, x=xb)` accumulates intermediate representations (or streamed QR factors).
  2. `step(select_loader)` solves candidate polynomials, selects survivors, inserts `[GMDHNorm, GMDHActivation, GMDHLayer]`, and registers trainable parameters.
  3. `update_layer(select_loader, train_loader)` re-selects `GMDHOutput.index` and refreshes `GMDHNorm` statistics after inter-layer gradient tuning.
  4. `remove_last_layer()` / `stop(remove_last=True)` reverts the last layer without full snapshots.
- **Solver-Free Piecewise Growth (`solver=None`, `inc.grow(...)`)**:
  1. Scores top-layer neurons via `GMDHOutput.importance_scores()` on `select_loader`.
  2. Lightly prunes weak neurons via `select` (and enforces the 50% forwarder cap).
  3. Inserts `[GMDHNorm (unlabeled x stats), GMDHActivation, GMDHLayer(in_features=..., init=...)]` and calls `GMDHOutput.resize_warm()` when `forward=True` for function-preserving expansion.
  4. Once `max_layers` (or `final=True`) is reached, applies `final_select` + `model.trim()` and transitions to `"tuning"`.

---

## 5. Solvers and Numerics (`src/gmdhtorch/solvers.py`)

Closed-form least-squares estimation fits weights $\mathbf{w}_m \in \mathbb{R}^P$ for each candidate neuron $m$:
$$\mathbf{X}_m \mathbf{w}_m \approx \mathbf{y}$$

### 5.1 Supported Solvers
- **`svd`**: Batched singular value decomposition via `torch.linalg.pinv(..., rcond=rcond)`. Handles rank deficiency and collinear polynomial cross-terms gracefully.
- **`qr`**: Batched column-pivoted QR factorization (`torch.linalg.qr`). Fast, direct solution on well-conditioned feature sets.
- **`ridge`**: Tikhonov regularized normal equations $(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$. Supports scalar $\alpha$ or Generalized Cross-Validation (`alpha="gcv"`).
- **Column Equilibration (`equilibrate=True`)**: Pre-scales columns of $\mathbf{X}_m$ by their inverse $L_2$ norm before solving, dramatically lowering condition numbers for high-degree polynomial powers.

### 5.2 Out-of-Core Streamed QR (`StreamedQRAccumulator`)
When observed training batches exceed `max_observe_bytes` (defaulting to `DEFAULT_MAX_OBSERVE_BYTES` = 64 MiB; set `max_observe_bytes=0` to stream immediately or `None` to retain batches in memory) and candidate chunks stay bounded by `chunk_size` / `max_chunk_bytes` (default `DEFAULT_MAX_CHUNK_BYTES` = 512 MiB; configurable in `ModelIncrementer`, `GMDHLayer`, `fit_layer_neurons`, and `StreamedQRAccumulator.solve`):
1. `StreamedQRAccumulator` processes mini-batches incrementally in `float64`.
2. Accumulates upper-triangular factor $\mathbf{R}_m \in \mathbb{R}^{P \times P}$ and projected target vector $\mathbf{c}_m \in \mathbb{R}^P$ across mini-batches using Givens / Householder updates:


$$
\begin{bmatrix}
  \mathbf{R}_{\text{old}} \\
  \mathbf{X}_{\text{batch}}
\end{bmatrix}
\to
\begin{bmatrix}
  \mathbf{R}_{\text{new}} \\
  \mathbf{0}
\end{bmatrix}
$$

3. Computes exact closed-form output mean and variance from $\mathbf{R}$ and column sums without a second data pass (`compute_closed_form_norm_stats`).
4. Solves all candidate systems $\mathbf{R}_m \mathbf{w}_m = \mathbf{c}_m$ in a single batched back-substitution at the end of the epoch.

---

## 6. Open Points & Active Research Directions

1. **Model Serialization (`state_dict` / `torch.save` / `torch.load`)** *(partially resolved)*:
   - **Implemented**: Portable `weights_only=True`-safe checkpointing via `GMDHModel.architecture_config()` / `GMDHModel.from_config()` (with post-load hook refresh for locked forwarders and `GMDHNorm` affine channels) and `ModelIncrementer.state_dict()` / `load_state_dict()`.
   - **Remaining**: Detaching `ModelIncrementer` forward hooks upon training completion to support full-object pickling cleanly, and optionally supporting implicit dynamic layer reconstruction directly inside `GMDHModel._load_from_state_dict()` without an explicit `from_config()` step.
2. **`min_layers` & Best-Layer Rollback on Stop**:
   - Investigate the stopping/removal behavior when `min_layers` is set (or when forcing growth past a temporary performance dip reveals a deeper optimum): currently, if `min_layers` forces keeping a layer that worsened performance and training later stops, the network can retain the degraded final layer instead of reverting to the best-performing layer depth observed so far. Explore tracking/restoring the best layer depth (multi-layer rollback or patience-based rollback to best checkpoint) when stopping.
3. **Adaptive Forwarded vs. New-Neuron Budget Balancing**:
   - Explore adaptive/intelligent strategies for splitting `max_neurons` between forwarded neurons (stability) and newly generated sparse candidates (innovation), replacing the fixed 50% cap rule.
