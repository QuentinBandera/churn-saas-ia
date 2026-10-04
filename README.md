# Prédiction du churn d'un éditeur SaaS B2B

[![CI/CD](https://github.com/QuentinBandera/churn-saas-ia/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/QuentinBandera/churn-saas-ia/actions/workflows/ci-cd.yml)

Projet réalisé dans le cadre de la certification « Concevoir et implémenter une solution d'intelligence artificielle ».

Le modèle estime, pour chaque compte client, la **probabilité de résiliation à la prochaine échéance**. Il fournit aussi les 3 principaux facteurs de risque et le revenu menacé, pour aider les équipes Customer Success à prioriser leurs actions de rétention.

| | |
|---|---|
| Modèle retenu | Régression logistique régularisée (scikit-learn) |
| Performance (jeu de test, 1 000 comptes) | ROC-AUC 0,88 · PR-AUC 0,76 · rappel 0,89 au seuil métier 0,18 |
| Mise à disposition | Scoring batch mensuel + API FastAPI dans un conteneur Docker |

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `Churn_SaaS_IA.ipynb` | Le notebook complet : cadrage, données, éthique, EDA, préparation, modélisation, déploiement, suivi |
| `churn_saas_complet.csv`, `churn_saas_echantillon.csv`, `catalogue_plans.csv` | Données du cas d'usage |
| `Enonce cas usage_churn_saas.*` | Énoncé du cas d'usage |
| `models/` | Modèle entraîné (`churn_model_v1.0.0.joblib`), ses métadonnées et les résultats des recherches d'hyperparamètres |
| `api/` | API de scoring (FastAPI), générée depuis le notebook, et ses dépendances figées |
| `tests/` | Tests automatisés (fonctions de préparation et API) |
| `Dockerfile` | Image de l'API, avec le modèle embarqué |
| `.github/workflows/ci-cd.yml` | Pipeline d'intégration et de livraison continues |

Les dossiers `data/` (couches Bronze, Silver, Gold), `mlruns/`, `mlflow.db` et `carbone/` ne sont pas versionnés : ils sont recréés à chaque exécution du notebook.

## Démarrage

Prérequis : Python 3.12.

```bash
python -m venv .venv
.venv/bin/pip install -r models/requirements.txt uvicorn httpx pytest jupyter
```

**Exécuter le notebook** (environ 4 minutes) :

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Churn_SaaS_IA.ipynb
```

**Consulter les expériences MLflow** :

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Puis ouvrir http://127.0.0.1:5001.

**Lancer l'API** :

```bash
CHURN_API_KEY=ma-cle .venv/bin/uvicorn api.app:app --port 8000
```

Documentation interactive : http://127.0.0.1:8000/docs (bouton *Authorize*, clé `ma-cle`).

**Lancer les tests** :

```bash
.venv/bin/python -m pytest -q
```

**Avec Docker** :

```bash
docker build -t churn-api:1.0.0 .
docker run --rm -p 8000:8000 -e CHURN_API_KEY=ma-cle churn-api:1.0.0
```

## Intégration et livraison continues

| Déclencheur | Étapes |
|---|---|
| Push sur `main` ou pull request | Tests → construction de l'image Docker → démarrage du conteneur et vérification de `/health`, `/ready` et de l'utilisateur non administrateur |
| Étiquette `vX.Y.Z` | Étapes ci-dessus → vérification que l'étiquette correspond à la version du modèle embarqué → publication de l'image sur `ghcr.io` → déploiement en production après approbation manuelle |

Le détail et la justification du pipeline sont au §10.6 du notebook.
