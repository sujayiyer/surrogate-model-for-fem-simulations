# Surrogate Model for Simulation (ML-based ROM)

A neural-network surrogate model that approximates an expensive FEM/CFD simulation, predicting outputs (e.g. max von Mises stress) directly from input parameters. Once trained, it replaces costly solver runs with millisecond-scale predictions.

## Motivation

High-fidelity FEM/CFD simulations (Ansys, etc.) are accurate but slow, making design exploration, optimization, and uncertainty quantification expensive. A Reduced-Order Model (ROM) trained on simulation data acts as a fast surrogate, enabling rapid what-if studies and integration into optimization loops.

## How It Works

1. **Sampling** — Input parameters (e.g. load, stiffness, geometry, boundary condition) are sampled via Latin Hypercube Sampling for even coverage of the design space.
2. **Data generation** — An analytical function stands in for the expensive solver, producing the target output for each sample.
3. **Training** — A small fully-connected PyTorch network learns the input → output mapping on normalized data.
4. **Evaluation** — Accuracy is measured with R² and MAE, plus a parity plot and loss curve.
5. **Inference** — New parameter sets are predicted near-instantly, demonstrating the ROM speedup.

## Requirements

```bash
pip install numpy scipy torch scikit-learn matplotlib
```

## Usage

```bash
python surrogate_rom.py
```

This trains the model, prints train/test metrics, saves the parity and loss plots, and runs an inference demo.

## Outputs

- Train/test loss printed per epoch
- Test-set R² and MAE
- `parity_plot.png` — predicted vs. true values
- `loss_curve.png` — training/validation loss
- Example prediction with timing comparison

## Project Structure
