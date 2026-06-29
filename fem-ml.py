"""
Surrogate Model (ML-based Reduced-Order Model) for an Expensive FEM/CFD Simulation
=================================================================================

This single-file script demonstrates how a small neural network can act as a
"surrogate" (a.k.a. ML-based Reduced-Order Model / ROM) that approximates an
expensive finite-element / CFD simulation at a tiny fraction of the cost.

Pipeline:
    1. Generate synthetic training data from a known analytical function that
       mimics a structural simulation (max von Mises stress as a function of
       4 design parameters). Samples are drawn with Latin Hypercube Sampling
       (LHS) for uniform coverage of the input space.
    2. Build a small fully-connected ReLU network in PyTorch.
    3. Train with normalized I/O, an 80/20 split, MSE loss and Adam.
    4. Evaluate with R^2 and MAE, and produce a parity plot + loss curve.
    5. Demonstrate the inference speedup of the ROM vs. the "expensive" sim.

Run with:  python fem-ml.py
Requires:  numpy, scipy, torch, scikit-learn, matplotlib
"""

import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc                      # Latin Hypercube Sampling
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------- #
# 1. Synthetic "expensive simulation"
# --------------------------------------------------------------------------- #
# In reality this would be a full FEM/CFD solve taking minutes to hours. Here we
# use a smooth, non-linear analytical surrogate of the true physics so that the
# script is self-contained and fast to run. The function maps 4 design
# parameters to a single scalar response: the maximum von Mises stress [MPa].
#
# Design parameters (with physically motivated ranges):
#   P  - applied load                [kN]      range [10, 100]
#   E  - material (Young's) modulus  [GPa]     range [70, 210]
#   L  - beam length / geometry dim  [m]       range [0.5, 3.0]
#   t  - cross-section thickness     [m]       range [0.01, 0.10]
#
# The closed form below is heuristic but captures realistic trends:
#   - stress grows with load P and length L (bending moment ~ P * L),
#   - stress drops as thickness t increases (stiffer section),
#   - a mild non-linear coupling on stiffness E to mimic material effects.

PARAM_NAMES = ["Load P [kN]", "Modulus E [GPa]", "Length L [m]", "Thickness t [m]"]
PARAM_BOUNDS = np.array(
    [
        [10.0, 100.0],   # P
        [70.0, 210.0],   # E
        [0.5, 3.0],      # L
        [0.01, 0.10],    # t
    ]
)


def expensive_simulation(X):
    """Analytical stand-in for an expensive FEM/CFD solve.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 4)
        Columns are [P, E, L, t].

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Maximum von Mises stress [MPa].
    """
    P, E, L, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    # Bending-moment-like driver: load times lever arm.
    bending = (P * L) / (t ** 2)

    # Stiffness relief: stiffer / thicker sections reduce stress, with a mild
    # non-linear dependence on the modulus.
    stiffness_relief = 1.0 / (1.0 + 0.002 * E * np.sqrt(t))

    # Geometric non-linearity to make the surface harder to fit than a plane.
    geom_nonlinear = 1.0 + 0.15 * np.sin(2.0 * L) * np.cos(0.5 * t * 100.0)

    stress = 0.05 * bending * stiffness_relief * geom_nonlinear
    return stress


def sample_inputs(n_samples, bounds, seed=SEED):
    """Draw `n_samples` points in the parameter space using Latin Hypercube
    Sampling, then scale each dimension to its physical bounds."""
    sampler = qmc.LatinHypercube(d=bounds.shape[0], seed=seed)
    unit_samples = sampler.random(n=n_samples)        # in [0, 1]^d
    return qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])


# --------------------------------------------------------------------------- #
# 2. Neural network surrogate
# --------------------------------------------------------------------------- #
class SurrogateNet(nn.Module):
    """Small fully-connected ReLU network: 4 inputs -> 1 output."""

    def __init__(self, in_dim=4, hidden=64, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# 3. Build dataset, normalize, split
# --------------------------------------------------------------------------- #
def main():
    N_SAMPLES = 500
    EPOCHS = 200
    LR = 1e-3

    print("Generating synthetic FEM/CFD data via Latin Hypercube Sampling...")
    X = sample_inputs(N_SAMPLES, PARAM_BOUNDS)
    y = expensive_simulation(X).reshape(-1, 1)

    # 80/20 train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # Standardize inputs and outputs (fit ONLY on training data to avoid leakage).
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train_s = x_scaler.transform(X_train)
    X_test_s = x_scaler.transform(X_test)
    y_train_s = y_scaler.transform(y_train)
    y_test_s = y_scaler.transform(y_test)

    # Convert to tensors.
    Xtr = torch.tensor(X_train_s, dtype=torch.float32)
    ytr = torch.tensor(y_train_s, dtype=torch.float32)
    Xte = torch.tensor(X_test_s, dtype=torch.float32)
    yte = torch.tensor(y_test_s, dtype=torch.float32)

    # --------------------------------------------------------------------- #
    # 4. Training loop
    # --------------------------------------------------------------------- #
    model = SurrogateNet(in_dim=X.shape[1], hidden=64, out_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_losses, test_losses = [], []

    print(f"\nTraining surrogate network for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        # --- training step ---
        model.train()
        optimizer.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        optimizer.step()

        # --- evaluation step ---
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(Xte), yte)

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{EPOCHS} | "
                f"train MSE: {loss.item():.5f} | test MSE: {test_loss.item():.5f}"
            )

    # --------------------------------------------------------------------- #
    # 5. Evaluation (in physical units)
    # --------------------------------------------------------------------- #
    model.eval()
    with torch.no_grad():
        y_pred_s = model(Xte).numpy()

    # Invert the output scaling to get stress back in MPa.
    y_pred = y_scaler.inverse_transform(y_pred_s)
    y_true = y_test  # already in physical units

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("\n" + "=" * 55)
    print("Test-set performance (physical units, MPa):")
    print(f"  R^2 : {r2:.4f}")
    print(f"  MAE : {mae:.4f} MPa")
    print("=" * 55)

    # --------------------------------------------------------------------- #
    # 6. Inference-speed demonstration (the ROM payoff)
    # --------------------------------------------------------------------- #
    # A single new design point.
    new_design = np.array([[55.0, 120.0, 1.8, 0.04]])  # P, E, L, t

    # "Expensive" reference value (here cheap, but in practice a full solve).
    t0 = time.perf_counter()
    true_val = expensive_simulation(new_design)[0]
    t_sim = time.perf_counter() - t0

    # Surrogate inference.
    t0 = time.perf_counter()
    with torch.no_grad():
        new_s = x_scaler.transform(new_design)
        rom_s = model(torch.tensor(new_s, dtype=torch.float32)).numpy()
    rom_val = y_scaler.inverse_transform(rom_s)[0, 0]
    t_rom = time.perf_counter() - t0

    print("\nInference demo on a NEW design point:")
    print(f"  Parameters: {dict(zip(['P', 'E', 'L', 't'], new_design[0]))}")
    print(f"  'Expensive' simulation : {true_val:9.3f} MPa  ({t_sim * 1e3:.3f} ms)")
    print(f"  Surrogate (ROM)        : {rom_val:9.3f} MPa  ({t_rom * 1e3:.3f} ms)")
    print(
        f"  Relative error         : "
        f"{abs(rom_val - true_val) / abs(true_val) * 100:.2f}%"
    )
    print(
        "  NOTE: a real FEM/CFD solve takes seconds-to-hours, so the surrogate\n"
        "        typically delivers a 10^3-10^6x speedup at inference time."
    )

    # --------------------------------------------------------------------- #
    # 7. Plots: parity plot + loss curve
    # --------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity plot (predicted vs. true).
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", label="ideal (y = x)")
    ax.set_xlabel("True max von Mises stress [MPa]")
    ax.set_ylabel("Predicted stress [MPa]")
    ax.set_title(f"Parity plot (R$^2$ = {r2:.3f}, MAE = {mae:.2f} MPa)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss curve.
    ax = axes[1]
    ax.plot(train_losses, label="train MSE")
    ax.plot(test_losses, label="test MSE")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalized units)")
    ax.set_title("Training / test loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = "surrogate_results.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plots to '{out_path}'.")
    plt.show()


if __name__ == "__main__":
    main()
