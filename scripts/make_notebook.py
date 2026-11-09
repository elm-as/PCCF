"""Generate demonstration Jupyter notebook for PCCF (Kaggle S6E11)."""

import json
from pathlib import Path

nb_dir = Path(__file__).resolve().parents[1] / "notebooks"
nb_dir.mkdir(parents=True, exist_ok=True)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PCCF — Predicting Credit Card Fraud (Kaggle S6E11)\n",
            "## End-to-End Leak-Free Cross-Validation, Digit Feature Engineering & Rank-Logit Blending\n",
            "\n",
            "> **ElmasCore Competitive ML System** • Target Metric: **ROC-AUC**\n",
            "\n",
            "---\n",
            "\n",
            "### Architectural Highlights\n",
            "1. **Decimal & Digit Decomposition**: Synthetic tabular artifacts (CTGAN / Copula spikes) captured via cents, tenths, and Benford leading digits.\n",
            "2. **Strict Cross-Fitted Bayesian Target Encoding**: 5 inner splits with Bayesian m-smoothing ($m=20.0$), eliminating out-of-fold target leakage.\n",
            "3. **Multi-Model Diversity**: Spline-GLM, LightGBM Shallow, CatBoost Shallow, and XGBoost Shallow.\n",
            "4. **Nelder-Mead Rank-Logit Optimization**: Bounded simplex optimization directly maximizing non-parametric ROC-AUC."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "PROJECT_ROOT = Path('..').resolve()\n",
            "if str(PROJECT_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            "from src.core.config import PipelineConfig\n",
            "from src.data_pipeline.memory_optimizer import build_optimized_dataset_bundle\n",
            "from src.feature_engineering.digit_extractor import extract_digit_key_features\n",
            "from src.feature_engineering.interaction_builder import build_domain_interaction_features\n",
            "from src.validation.fold_manager import generate_stratified_folds\n",
            "\n",
            "print('PCCF modules successfully imported!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Data & Memory Optimization\n",
            "\n",
            "Loading 700k train and 300k test records, downcasting 64-bit integers and floats to 32-bit and 8-bit representations."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "cfg = PipelineConfig.create_default(PROJECT_ROOT)\n",
            "raw_train = pd.read_csv(cfg.paths.raw_data_dir / 'train.csv')\n",
            "raw_test = pd.read_csv(cfg.paths.raw_data_dir / 'test.csv')\n",
            "bundle = build_optimized_dataset_bundle(raw_train, raw_test, cfg.validation)\n",
            "print(f'Train shape: {bundle.train_features.shape}, Test shape: {bundle.test_features.shape}')\n",
            "print(f'Fraud prevalence: {bundle.train_target.mean()*100:.3f}%')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Granular Digit Feature Engineering\n",
            "\n",
            "Extracting discrete cents modulo keys (`.00` to `.99`), tenths keys, and leading digits."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "tr_digit, digit_cols = extract_digit_key_features(bundle.train_features, bundle.numerical_columns)\n",
            "print(f'Generated {len(digit_cols)} discrete digit keys.')\n",
            "tr_digit[['amount_usd', 'amount_usd__cents_key', 'amount_usd__lead_key']].head(5)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Stratified Folds Generation & Leak-Free Target Encoding\n",
            "\n",
            "Generating 5 deterministic stratified splits with target distribution preservation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "folds = generate_stratified_folds(bundle, cfg.validation.n_splits, cfg.validation.random_seed)\n",
            "print(f'Successfully built {len(folds)} stratified folds.')\n",
            "for f in folds:\n",
            "    print(f'Fold {f.fold_index + 1}: Train {len(f.train_indices):,}, Val {len(f.val_indices):,}')"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_file = nb_dir / "pccf_pipeline_walkthrough.ipynb"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"PCCF notebook written: {output_file}")
