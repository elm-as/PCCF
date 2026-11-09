# PCCF — Predicting Credit Card Fraud (Kaggle Playground Series S6E11)

> **Signature ElmasCore** — *"From Data to Systems"*
> Pipeline d'apprentissage statistique et d'assemblage Rank-Logit conçu pour viser la **1ère place du leaderboard** de la compétition [Kaggle Playground Series - Season 6, Episode 11](https://www.kaggle.com/competitions/playground-series-s6e11) tout en sécurisant le **Private Leaderboard (80% des données de test)**.

---

## 1. Justification Technique & Décisions d'Architecture

Conformément à la charte [ELMAS.md](./ELMAS.md), chaque brique de cette stack répond à une propriété mathématique démontrée du jeu de données S6E11 :

1. **Modèles Shallow & Additifs (Anti-Overfitting du Bruit Synthétique)** :
   - Le générateur synthétique de la compétition S6E11 suit un processus de risque quasi-additif en log-odds. Les arbres profonds (`depth > 5`) mémorisent les artefacts de génération et perdent jusqu'à `-0.0020` ROC-AUC en validation.
   - Notre zoo combine 4 familles complémentaires à faible variance :
     - `spline_glm` : Régression logistique régularisée sur expansions B-Splines (`n_knots=6, degree=2`).
     - `lightgbm_shallow` : Arbres à feuilles restreintes (`num_leaves=8, max_depth=3`).
     - `catboost_shallow` : Arbres symétriques ("oblivious trees", `depth=4, l2_leaf_reg=12.0`).
     - `xgboost_shallow` : Boosting par histogrammes contraint (`max_depth=3, min_child_weight=60`).
2. **Extraction de Clés "Digit-Level" (`digit_extractor.py`)** :
   - Décomposition des variables continues en clés catégorielles discrètes : centimes (`__cents_key`), dixièmes (`__tenth_key`), chiffre des unités (`__unit_key`), dizaines (`__tens_key`) et premier chiffre significatif de Benford (`__lead_key`).
3. **Cross-Fitted Bayesian Target Encoding (`cross_fitted_encoder.py`)** :
   - Calcul strictement hors-pli (*nested StratifiedKFold*) avec lissage bayésien (*m-estimate*, $m = 20$). Aucune observation d'entraînement ne voit jamais sa propre cible $y_i$, éliminant le piège de fuite de données (`-0.030` AUC) documenté sur S6E11.
4. **Rank-Logit Fusion & Optimisation Nelder-Mead (`rank_transformer.py`, `nelder_mead_blender.py`)** :
   - Le ROC-AUC étant une métrique d'ordre pur, les probabilités de chaque modèle sont projetées en centiles empiriques $R \in (0, 1)$ puis dans l'espace log-odds $z = \ln(R / (1 - R))$, où les poids $\mathbf{w}$ sont optimisés par simplexe de Nelder-Mead sur le score OOF.

---

## 2. Organisation par Domaine Métier

```text
PCCF/
├── ELMAS.md                                  # Charte de développement ElmasCore
├── README.md                                 # Documentation complète du système
├── requirements.txt                          # Dépendances Python strictement nécessaires
├── main.py                                   # Point d'entrée d'orchestration
├── src/
│   ├── core/
│   │   ├── config.py                         # Dataclasses immuables de configuration
│   │   └── types.py                          # Structures de données typées du domaine
│   ├── data_pipeline/
│   │   ├── dataset_downloader.py             # Acquisition Kaggle / local / benchmark synthétique
│   │   ├── schema_validator.py               # Vérification d'intégrité (IDs, binaire, colonnes)
│   │   └── memory_optimizer.py               # Downcasting RAM (float32, int8/16/32)
│   ├── feature_engineering/
│   │   ├── digit_extractor.py                # Extraction des centimes et chiffres de Benford
│   │   ├── interaction_builder.py            # Croisements catégoriels et ratios financiers
│   │   └── cross_fitted_encoder.py           # Target Encoding OOF imbriqué sans fuite
│   ├── modeling/
│   │   ├── linear_baseline.py                # Modèle additif B-Spline GLM
│   │   ├── lightgbm_shallow.py               # Modèle LightGBM shallow (8 feuilles)
│   │   ├── catboost_shallow.py               # Modèle CatBoost symétrique (profondeur 4)
│   │   └── xgboost_shallow.py                # Modèle XGBoost hist (profondeur 3)
│   ├── validation/
│   │   ├── metric_calculator.py              # Calcul pur et sécurisé du ROC-AUC
│   │   └── fold_manager.py                   # Orchestrateur Stratified 5-Fold OOF
│   ├── ensembling/
│   │   ├── rank_transformer.py               # Transformations centiles et Rank-Logit
│   │   └── nelder_mead_blender.py            # Optimiseur Nelder-Mead sur ROC-AUC hors-pli
│   └── submission/
│       ├── submission_formatter.py           # Validation stricte du schéma Kaggle
│       └── submission_exporter.py            # Export et audit statistique des soumissions
└── tests/
    └── test_pipeline_integrity.py            # Suite de tests unitaires et anti-fuite
```

---

## 3. Installation & Exécution

### Prérequis
```bash
pip install -r requirements.txt
```

### Lancer la Suite de Tests Automatisés
```bash
pytest tests/test_pipeline_integrity.py -v
```

### Exécuter le Pipeline sur les Données Officielles Kaggle
1. Placez `train.csv` et `test.csv` dans `data/raw/` (ou configurez `~/.kaggle/kaggle.json` pour le téléchargement automatique via `kagglehub`).
2. Lancez l'entraînement complet :
```bash
python main.py
```

### Vérification Hors-Ligne (Benchmark Synthétique S6E11)
Pour tester l'intégralité de la chaîne de bout en bout sans connexion Kaggle :
```bash
python main.py --verify-synthetic --fast-smoke
```

---

## 4. Double Stratégie de Soumission Finale

Chaque exécution génère deux fichiers vérifiés dans `submissions/` :
- `submissions/submission_cv_conservative.csv` : Assemblage en log-odds calibrés, conçu pour résister au *shakeup* sur les **80% du Private Leaderboard**.
- `submissions/submission_leaderboard_blend.csv` : Assemblage **Rank-Logit Fusion** optimisé par Nelder-Mead pour maximiser le classement sur le **Public & Private Leaderboard**.

---

— **Elmas**, ElmasCore

