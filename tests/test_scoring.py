# Tests unitaires des fonctions de préparation partagées par le notebook, le batch et l'API.
import numpy as np
import pandas as pd

from api.churn_scoring import add_features, normalize_cat, parse_dates, parse_raw, to_number


def test_to_number_gere_tous_les_formats_du_jeu():
    s = pd.Series(["11.37 €", "20.0%", "33,3", "7.9 h", "1\u00a0200", None, "abc"])  # espace insécable = séparateur de milliers
    out = to_number(s)
    np.testing.assert_allclose(out[:5], [11.37, 20.0, 33.3, 7.9, 1200.0])
    assert out[5:].isna().all()                # manquant et texte non numérique -> NaN, sans planter


def test_normalize_cat_casse_espaces_et_sigles():
    out = normalize_cat(pd.Series([" PRO ", "pro", "tpe", "ÉDUCATION", None]))
    assert out[:4].tolist() == ["Pro", "Pro", "TPE", "Éducation"]
    assert pd.isna(out[4])


def test_parse_dates_trois_formats_sans_inverser_jour_et_mois():
    out = parse_dates(pd.Series(["2024-02-06", "31/01/2024", "23 Dec 2024"]))
    assert out.tolist() == [pd.Timestamp("2024-02-06"), pd.Timestamp("2024-01-31"), pd.Timestamp("2024-12-23")]


def test_parse_raw_ne_perd_aucune_valeur_non_vide(sample_raw):
    parsed = parse_raw(sample_raw)
    for col in ["revenu_mensuel_recurrent_eur", "taux_adoption_pct", "delai_reponse_support_h"]:
        assert parsed[col].notna().sum() == sample_raw[col].notna().sum()


def test_add_features_imputations_metier_et_variables_derivees():
    d = pd.DataFrame({
        "sieges_souscrits": [10.0], "utilisateurs_actifs": [4.0], "taux_adoption_pct": [np.nan],
        "revenu_mensuel_recurrent_eur": [np.nan], "prix_mensuel_par_siege_eur": [25.0],
        "fonctionnalites_utilisees": [8.0], "fonctionnalites_incluses": [16.0], "heures_usage_30j": [20.0],
        "delai_reponse_support_h": [15.0], "sla_reponse_h": [12.0],
    })
    out = add_features(d).iloc[0]
    assert out["taux_adoption_pct"] == 40.0                # actifs / sièges
    assert out["revenu_mensuel_recurrent_eur"] == 250.0    # sièges × prix catalogue
    assert out["ratio_fonctionnalites"] == 0.5
    assert out["heures_par_utilisateur"] == 5.0
    assert out["ecart_sla_support_h"] == 3.0
    assert out["sieges_inactifs"] == 6.0
    assert np.isclose(out["log_mrr"], np.log1p(250.0))
