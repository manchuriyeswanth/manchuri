# Transformer Low-Rank QKᵀ Projection

This repository implements a Transformer variant that projects Q and K into a lower-dimensional basis to accelerate the QKᵀ cost. It supports:

- Learned basis (trainable) or precomputed basis via SVD/PCA,
- MLflow logging,
- Quick demo training via GitHub Actions.

## Quickstart (local)

1. Create and activate a python env:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
