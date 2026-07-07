# TRB Traffic Speed Analysis — BGCP Imputation & Causal Analysis

This repository contains code for a TRB paper on traffic speed imputation and causal analysis using **Bayesian Gaussian CP decomposition (BGCP)** on the **G20 expressway** (upstream direction, 294.385–321.885 km).

## Repository Structure

| File | Description |
|---|---|
| `bgcp.py` | BGCP model — Bayesian Gaussian CP decomposition for tensor completion |
| `bgcp_gpu.py` | GPU-accelerated BGCP implementation |
| `causal_bgcp.py` | Causal analysis extension of BGCP |
| `run_bgcp.py` | Script to run BGCP imputation |
| `run_casual_bgcp.py` | Script to run causal BGCP analysis |
| `creat_missingpoint.py` | Generate synthetic missing points for evaluation |
| `data_check.py` | Data integrity and quality checks |
| `data_create.py` | Data loading and preprocessing pipeline |
| `data_process.py` | Feature engineering and data transformation |
| `data_summary.md` | Detailed data description (English) |

## Dataset

The raw data is **not included** in this repository (excluded via `.gitignore` due to file size). See `data_summary.md` for a full description.

**Key facts:**
- **Road:** G20 expressway, upstream, 294.385–321.885 km (56 segments at 500 m intervals)
- **Time span:** 2021-10-20 to 2021-12-31 (48 days, 5-minute resolution)
- **Tensor shape:** (56 segments, 73 intraday time slots, 48 days) → 196,224 entries
- **Speed range:** 18.24 – 113.64 km/h (mean: 94.12, std: 10.33)
- **Missing ratio:** 0.30% (589/196,224)

## Quick Start

```bash
# Data preprocessing
python data_process.py

# Run BGCP imputation
python run_bgcp.py

# Run causal analysis
python run_casual_bgcp.py
```

## Requirements

- Python 3.8+
- NumPy, Pandas
- Tensorly (for tensor decomposition)
- CuPy (optional, for GPU acceleration)

## Notes

- Large data files (`*.csv`, `*.npz`) are excluded from version control.
- The repository was pushed directly to GitHub using a Personal Access Token. If you encounter a `403` error involving `local_proxy@127.0.0.1:41729`, that is a local proxy authentication issue — not a GitHub permission problem. To fix it, run:

```bash
git remote set-url origin https://github.com/zxhouge123/trb-paper-code.git
```
