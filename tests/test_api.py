# Tests de l'API de scoring : mêmes comportements que ceux vérifiés au §10.4 du notebook.
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.app as api_app
from api.churn_scoring import parse_raw, score_accounts

KEY = {"X-API-Key": os.environ["CHURN_API_KEY"]}


@pytest.fixture(scope="module")
def client():
    with TestClient(api_app.app) as c:        # le bloc `with` déclenche le chargement du modèle (lifespan)
        yield c


@pytest.fixture(scope="module")
def accounts(sample_raw):
    df = parse_raw(sample_raw.head(3)).drop(columns=["date_souscription"])
    df = df[list(api_app.Account.model_fields)]
    return df.astype(object).where(df.notna(), None).to_dict("records")


def test_health(client):
    assert client.get("/health").status_code == 200


def test_ready_avec_modele_charge(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_version"] == api_app.MODEL_VERSION


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "faux"}])
def test_score_refuse_sans_cle_valide(client, accounts, headers):
    assert client.post("/score", json={"accounts": accounts}, headers=headers).status_code == 401


def test_score_refuse_une_entree_hors_bornes(client, accounts):
    r = client.post("/score", json={"accounts": [dict(accounts[0], csat=9)]}, headers=KEY)
    assert r.status_code == 422


def test_score_refuse_un_plan_inconnu(client, accounts):
    r = client.post("/score", json={"accounts": [dict(accounts[0], plan="Platinum")]}, headers=KEY)
    assert r.status_code == 422


def test_score_accepte_les_champs_optionnels_absents(client, accounts):
    required = {k: v for k, v in accounts[0].items() if api_app.Account.model_fields[k].is_required()}
    assert client.post("/score", json={"accounts": [required]}, headers=KEY).status_code == 200


def test_scores_api_identiques_au_batch(client, accounts, sample_raw):
    r = client.post("/score", json={"accounts": accounts}, headers=KEY)
    assert r.status_code == 200
    api_scores = {a["client_id"]: a["proba_churn"] for a in r.json()["results"]}
    bundle = api_app.load_bundle()
    batch = score_accounts(sample_raw.head(3), bundle.model, bundle.meta, bundle.catalogue)
    assert api_scores == dict(zip(batch["client_id"], batch["proba_churn"]))


def test_request_id_renvoye(client):
    assert client.get("/health", headers={"X-Request-ID": "demo-42"}).headers["X-Request-ID"] == "demo-42"


def test_503_si_modele_absent(client, accounts):
    api_app.app.dependency_overrides[api_app.get_bundle] = lambda: None
    try:
        assert client.get("/ready").status_code == 503
        assert client.post("/score", json={"accounts": accounts}, headers=KEY).status_code == 503
    finally:
        api_app.app.dependency_overrides.clear()
