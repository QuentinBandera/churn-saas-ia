# Configuration commune des tests : racine du projet, clé d'API de test, données d'exemple.
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                                  # l'API lit models/ et catalogue_plans.csv en chemins relatifs
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CHURN_API_KEY", "cle-de-test")


@pytest.fixture(scope="session")
def sample_raw() -> pd.DataFrame:
    return pd.read_csv(ROOT / "churn_saas_echantillon.csv", dtype=str, encoding="utf-8-sig")
