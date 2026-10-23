<div align="center">

# PIDG-Mamba

**Short-term solar flare forecasting from Aditya-L1 X-ray data**

Takes the last hour of soft and hard X-ray light curves from ISRO's Aditya-L1 spacecraft, predicts the next 10 minutes of solar X-ray flux, and classifies the coming window as Quiet, C, M or X-class.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Astropy](https://img.shields.io/badge/Astropy-FITS-FF8C00)](https://www.astropy.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com)

</div>

---

## The problem

Solar flares can disrupt satellites, GPS and radio communication. They show up first as a sharp rise in X-ray flux, so a model that sees the rise early can give a few minutes of warning.

Aditya-L1 carries two X-ray instruments that look at the same Sun in different energy bands:

| Instrument | Band | What it tells us |
|---|---|---|
| **SoLEXS** | Soft X-rays | Thermal emission: the "brightness" of the flare, and the quantity we forecast |
| **HEL1OS** | Hard X-rays (CdTe and CZT detectors) | Non-thermal emission from accelerated particles, which usually spikes *before* the soft X-ray peak |

This project fuses both instruments and uses the physics linking them to forecast flares.

---

## Approach

```
SoLEXS light curve ─┐
                    ├─► 10 s resampling ─► feature engineering ─► 60-min window
HEL1OS (3 detectors)┘         + GTI mask (drops bad telemetry)
                                                   │
                                                   ▼
                              ┌──────────────────────────────────┐
                              │  Channel graph attention         │  which detectors matter right now
                              │  Multi-scale dilated causal TCN  │  local flare shapes, no future leakage
                              │  Mamba selective state space     │  long-range context at linear cost
                              └──────────────┬───────────────────┘
                                   ┌─────────┴─────────┐
                                   ▼                   ▼
                       10-min flux forecast     Flare class (Quiet / C / M / X)
```

### Features
All computed in `log1p` space, since flux spans four or more orders of magnitude:
- SoLEXS and per-detector HEL1OS flux, EMA-smoothed to suppress photon shot noise
- First derivatives (rate of thermal heating, non-thermal impulses)
- SoLEXS second derivative (acceleration of thermal emission)
- Per-detector hard-to-soft ratio
- A Good Time Interval (GTI) mask channel so the model knows which samples to trust

### Physics-informed loss
Plain MSE learns to predict a flat line, because quiet periods dominate the data. The final model is trained with a combined loss instead:

| Term | Purpose |
|---|---|
| **Wing loss** (asymmetric) | Regression that penalises *under*-predicting a flare twice as hard as over-predicting |
| **Focal loss** | 4-class flare classification with class weights (Quiet 1, C 4, M 20, X 60) for heavy imbalance |
| **Peak loss** | Gets the flare's maximum right |
| **Delta and curvature losses** | Match the shape of the rise and decay, not just the level |
| **Neupert-effect term** | Encourages consistency with the Neupert effect: the time derivative of soft X-rays tracks hard X-ray flux |
| **GTI masking** | Ignores data gaps and bad telemetry |

---

## Repository

| Notebook | What's inside |
|---|---|
| [`chunking-clean.ipynb`](chunking-clean.ipynb) | Indexes raw SoLEXS `.lc` and HEL1OS `.fits` files, matches observation days, resamples to a common grid and builds the training dataset (`X_all.npy`) |
| [`TCN.ipynb`](TCN.ipynb) | Baseline 1: residual dilated causal TCN with a 20-min input window |
| [`NHiTS.ipynb`](NHiTS.ipynb) | Baseline 2: N-HiTS with coarse, mid and fine stacks and a Fourier basis, trained with focal loss |
| [`TCN_Mamba.ipynb`](TCN_Mamba.ipynb) | **Final model.** Channel attention, TCN and Mamba with the physics-informed loss, a 60-min input window, evaluation, and an interactive inference widget |

### Model progression

| | TCN | N-HiTS | **TCN-Mamba** |
|---|---|---|---|
| Input window | 20 min | 20 min | **60 min** |
| Forecast horizon | 10 min | 10 min | 10 min |
| HEL1OS detectors | combined | combined | **per-detector (CdTe1, CdTe2, CZT1)** |
| Classification loss | cross-entropy | focal | focal + class weights |
| Physics constraints | none | none | **Neupert term and GTI mask** |

### Evaluation
Each notebook reports:
- A confusion matrix over Quiet / C / M / X
- Per-class precision, recall and **True Skill Statistic (TSS)**, the standard metric in flare forecasting
- Forecast-versus-actual plots, including views filtered to missed flares, false alarms and worst predictions

---

## Running it

The notebooks were built to run on Kaggle with a GPU.

1. Add the Aditya-L1 SoLEXS (Level-1 light curves) and HEL1OS (Level-1 CdTe/CZT spectra) data as Kaggle inputs.
2. Set `SOLEXS_DIR`, `HELIOS_DIR` and `GTI_DIR` in the config cell.
3. Run all cells. If no FITS data is found, the notebooks fall back to a calibrated synthetic dataset so the pipeline still runs end to end.

```bash
pip install torch numpy pandas matplotlib astropy sunpy scikit-learn ipywidgets
```

---

## Tech stack
PyTorch (mixed precision, multi-GPU) · a Mamba selective-scan implemented from scratch in PyTorch · Astropy / SunPy for FITS handling · NumPy · pandas · scikit-learn · Matplotlib

---

## 📈 Results

| Metric | Final model (TCN-Mamba) |
|---|---|
| **True Skill Statistic (TSS)** | **0.9642** |
| **F1 score** (flare classification) | **0.9254** |

Measured on held-out test data, with flares classified as Quiet, C, M or X from a 60-minute input window and a 10-minute forecast horizon.
