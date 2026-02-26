# EV Charging Probabilistic Forecasting (Jiaxing + Palo Alto)

This repository contains the code for our probabilistic multi-horizon EV charging demand forecasting experiments on the Jiaxing and Palo Alto datasets.

## Files
- `DLinear_Res_Prob_Jiaxing_PaloAlto.py`  
  Main proposed model training script for Jiaxing and Palo Alto datasets (DLinear-Res probabilistic model).
- `Calibration Diagnostics and Conformal Adjustments.py`  
  Calibration curves, PIT, and Conformalized Quantile Regression (CQR) diagnostics.
- `Cost Sensitivity Analysis.py`  
  Cost sensitivity analysis with τ* scheduling (asymmetric costs).
- `Comparison with Baselines.py`  
  Baseline comparison (MLP, TCN, BiLSTM, BiGRU, TFT, PatchTST, Crossformer, TimesNet).
- `Statistical Analysis.py`  
  Friedman test, Nemenyi post-hoc (optional), and Wilcoxon + Holm correction across seeds/horizons.

## Requirements
Python 3.9+ recommended.

Install dependencies:
```bash
pip install numpy pandas scikit-learn torch matplotlib scipy
pip install scikit-posthocs   # optional (for Nemenyi test)
