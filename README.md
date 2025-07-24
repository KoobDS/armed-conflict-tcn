# Armed‑Conflict‑Forecast (TCN vs. Random Forest)
Temporal forecasting of weekly conflict events & fatalities, plus benchmark comparison  
*(ACLED 1997 – 2025 · admin‑1 level · 26‑week horizon)*

## Headline
A **Temporal Convolutional Network (TCN)** trained on 52‑week ACLED sequences outperforms / complements in an ensemble approach a feature‑engineered **Random Forest** baseline.  
TCN wins on four of six conflict types; RF excels on the remaining two, therefore final deployment is an ensemble.

### Evaluation

| Conflict type | MAE (RF) | MAE (TCN) | RMSE (RF) | RMSE (TCN) | R² (RF) | R² (TCN) |
|---------------|---------:|----------:|----------:|-----------:|--------:|---------:|
| Fatalities    | **1.12** | 0.80 | 16.30 | **16.49** | 0.38 | **0.43** |
| Battles       | **0.17** | 0.25 | **1.42** | 4.57 | **0.90** | 0.22 |
| Protests      | 0.62 | **0.55** | 2.18 | **1.79** | 0.64 | **0.74** |
| Riots         | 0.11 | **0.12** | 0.65 | **0.48** | 0.80 | **0.90** |
| Explosions    | **0.18** | 0.23 | **1.96** | 3.10 | **0.93** | 0.87 |
| Civ. Violence | 0.16 | **0.16** | 0.62 | **0.64** | 0.66 | **0.68** |

---

## Tech stack
- **Python 3.10**   ·  PyTorch 2.3
- scikit‑learn 1.5 (Random Forest baseline)  
- pandas 2.2 · NumPy 2.2  
- YAML‑driven configs (`config.yaml`)  

---

## Repository layout
| Path | Description |
|------|-------------|
| `metrics/` | Model‑level CSVs & helper evaluation functions |
| `ACLED_TCN.ipynb` | Notebook for TCN training and inference |
| `Baseline_RF.ipynb` | Random‑Forest baseline notebook |
| `train_TCN.py` | Script version of the TCN training loop |
| `curve.py` | Training‑loss / validation‑loss plot generator |
| `Processing.ipynb` | Data cleaning & tensor‑building pipeline |
| `config.yaml` | Hyper‑parameters and file paths |
| `comparison_metrics.xlsx` | Full metric comparison table (RF vs TCN) |

---

## Quick start (TCN)
```bash
git clone https://github.com/KoobDS/armed-conflict-tcn.git
cd armed-conflict-tcn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
python train_TCN.py --config config.yaml

My contribution (Benjamin Koob): End‑to‑end prediction pipeline
- Merged & cleaned ACLED sequences.
- Engineered 52 -> 26‑week tensors.
- Authored all RF & TCN training scripts, inference, and evaluation plots.
Teammates (see commit log): Assisted in modeling or focused on socio‑economic impact analysis (panel regression / PVAR) and dashboarding.

This README summarizes the forecasting component; impact‑analysis notebooks were outside my scope of interest.
Note: The full technical report cannot be released due to a restriction.
