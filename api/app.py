# API de scoring churn (FastAPI).
# Lancer depuis la racine du projet :
#     CHURN_API_KEY=ma-cle uvicorn api.app:app --port 8000
# Documentation interactive : http://127.0.0.1:8000/docs
import hashlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.churn_scoring import normalize_cat, score_accounts

MODEL_DIR = Path(os.getenv("CHURN_MODEL_DIR", "models"))
MODEL_VERSION = os.getenv("CHURN_MODEL_VERSION", "1.0.0")
CATALOGUE_PATH = Path(os.getenv("CHURN_CATALOGUE", "catalogue_plans.csv"))
API_KEY = os.getenv("CHURN_API_KEY", "dev-key")  # à fournir en production, jamais dans le code


@dataclass
class ModelBundle:
    # Tout ce qu'une prédiction nécessite, chargé une fois au démarrage
    model: Any
    meta: dict
    catalogue: pd.DataFrame


_BUNDLE: Optional[ModelBundle] = None


def load_bundle() -> ModelBundle:
    meta = json.loads((MODEL_DIR / f"churn_model_v{MODEL_VERSION}.json").read_text())
    path = MODEL_DIR / f"churn_model_v{MODEL_VERSION}.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("checksum du modèle invalide")
    catalogue = pd.read_csv(CATALOGUE_PATH, encoding="utf-8-sig")
    catalogue["plan"] = normalize_cat(catalogue["plan"])
    catalogue["support_dedie"] = (catalogue["support_dedie"].str.strip().str.lower() == "oui").astype(int)
    return ModelBundle(model=joblib.load(path), meta=meta, catalogue=catalogue)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _BUNDLE
    try:
        _BUNDLE = load_bundle()
    except (FileNotFoundError, ValueError, KeyError):
        _BUNDLE = None  # l'API démarre quand même : /ready répondra 503
    yield


app = FastAPI(title="Churn scoring API", version=MODEL_VERSION, lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def get_bundle() -> Optional[ModelBundle]:
    return _BUNDLE


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clé API absente ou invalide")


class Account(BaseModel):
    client_id: str = Field(..., examples=["CLI-000123"])
    plan: str = Field(..., examples=["Pro"])
    secteur: Optional[str] = None
    pays: Optional[str] = None
    taille_entreprise: str = Field(..., examples=["PME"])
    anciennete_mois: int = Field(..., ge=0, le=240)
    sieges_souscrits: int = Field(..., ge=1)
    utilisateurs_actifs: int = Field(..., ge=0)
    taux_adoption_pct: Optional[float] = Field(None, ge=0, le=100)
    connexions_30j: int = Field(..., ge=0)
    heures_usage_30j: Optional[float] = Field(None, ge=0)
    fonctionnalites_utilisees: int = Field(..., ge=0)
    nb_integrations: Optional[int] = Field(None, ge=0)
    derniere_connexion_jours: int = Field(..., ge=0)
    tickets_support_90j: int = Field(..., ge=0)
    delai_reponse_support_h: Optional[float] = Field(None, ge=0)
    csat: Optional[int] = Field(None, ge=1, le=5)
    retards_paiement_12m: Optional[int] = Field(None, ge=0)
    revenu_mensuel_recurrent_eur: Optional[float] = Field(None, ge=0)


class ScoreRequest(BaseModel):
    accounts: list[Account] = Field(..., min_length=1, max_length=1000)


class AccountScore(BaseModel):
    client_id: str
    proba_churn: float = Field(..., ge=0, le=1)
    a_risque: bool
    niveau: str
    facteurs_principaux: str
    mrr_eur: Optional[float]
    mrr_a_risque_eur: Optional[float]


class ScoreResponse(BaseModel):
    model_version: str
    threshold: float
    results: list[AccountScore]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready(bundle: Optional[ModelBundle] = Depends(get_bundle)) -> dict:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ready", "model_version": bundle.meta["version"]}


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score(payload: ScoreRequest, bundle: Optional[ModelBundle] = Depends(get_bundle)) -> ScoreResponse:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    df = pd.DataFrame([a.model_dump() for a in payload.accounts]).astype("string")
    unknown = set(normalize_cat(df["plan"])) - set(bundle.catalogue["plan"])
    if unknown:
        raise HTTPException(status_code=422, detail=f"Plan inconnu du catalogue : {sorted(unknown)}")
    res = score_accounts(df, bundle.model, bundle.meta, bundle.catalogue)
    res = res.astype({"niveau": str}).astype(object).where(res.notna(), None)
    return ScoreResponse(model_version=bundle.meta["version"], threshold=bundle.meta["decision_threshold"],
                         results=res.to_dict(orient="records"))
