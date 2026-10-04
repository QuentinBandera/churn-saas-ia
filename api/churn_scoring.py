# Fonctions de préparation et de scoring — générées depuis Churn_SaaS_IA.ipynb (ne pas éditer à la main).
import numpy as np
import pandas as pd

NUM_COLS = ['anciennete_mois', 'sieges_souscrits', 'utilisateurs_actifs', 'taux_adoption_pct', 'connexions_30j', 'heures_usage_30j', 'fonctionnalites_total', 'fonctionnalites_utilisees', 'nb_integrations', 'derniere_connexion_jours', 'tickets_support_90j', 'delai_reponse_support_h', 'csat', 'retards_paiement_12m', 'revenu_mensuel_recurrent_eur', 'sante_compte_fin_periode', 'valeur_vie_client_eur']
CAT_NORMALIZE = ['secteur', 'pays', 'taille_entreprise', 'plan', 'jour_souscription', 'couleur_theme_interface', 'code_datacenter', 'groupe_experimentation']
DATE_FORMATS = ['%Y-%m-%d', '%d/%m/%Y', '%d %b %Y']
FACTOR_LABELS = {'anciennete_mois': 'compte récent', 'nb_integrations': "peu d'intégrations", 'retards_paiement_12m': 'retards de paiement', 'derniere_connexion_jours': 'inactif depuis longtemps', 'tickets_support_90j': 'nombreux tickets support', 'delai_reponse_support_h': 'réponses support lentes', 'ecart_sla_support_h': 'SLA support dépassé', 'csat': 'satisfaction faible', 'taux_adoption_pct': 'faible adoption des sièges', 'connexions_30j': 'peu de connexions', 'heures_usage_30j': 'usage faible', 'fonctionnalites_utilisees': 'peu de fonctionnalités utilisées', 'ratio_fonctionnalites': "faible profondeur d'usage", 'heures_par_utilisateur': 'usage par utilisateur faible', 'utilisateurs_actifs': "peu d'utilisateurs actifs", 'sieges_inactifs': 'licences inutilisées', 'sieges_souscrits': 'nombre de sièges', 'log_mrr': 'niveau de MRR'}


def to_number(s: pd.Series) -> pd.Series:
    # "11.37 €" / "20.0%" / "33,3" / "7.9 h"  ->  float
    cleaned = (s.astype("string")
                 .str.replace(r"[€%]|\bh\b", "", regex=True)
                 .str.replace(" ", "", regex=False)
                 .str.strip()
                 .str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")  # NaN numpy (pas pd.NA) pour scikit-learn


def normalize_cat(s: pd.Series) -> pd.Series:
    # " PRO " / "pro" / "Pro"  ->  "Pro"  (TPE/PME/ETI/GE restent en majuscules)
    s = s.astype("string").str.strip().str.lower()
    s = s.map(lambda v: v if pd.isna(v) else (v.upper() if v in {"tpe", "pme", "eti", "ge"} else v.capitalize()))
    return s.astype(object).where(s.notna(), np.nan)  # NaN numpy (pas pd.NA) : scikit-learn ne gère pas pd.NA


def parse_dates(s: pd.Series) -> pd.Series:
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:  # chaque format est essayé explicitement, sans deviner
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    return out


def parse_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Conversions de format uniquement (aucune imputation, aucune suppression d'information)
    df = df_raw.copy()
    for c in NUM_COLS:
        if c in df:
            df[c] = to_number(df[c])
    for c in CAT_NORMALIZE:
        if c in df:
            df[c] = normalize_cat(df[c])
    if "date_souscription" in df:
        df["date_souscription"] = parse_dates(df["date_souscription"])
    if "churn" in df:
        df["churn"] = df["churn"].astype(int)
    return df


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    # Règles déterministes : imputations métier + variables dérivées (aucun apprentissage -> pas de fuite)
    d = d.copy()
    d["taux_adoption_pct"] = d["taux_adoption_pct"].fillna(
        (d["utilisateurs_actifs"] / d["sieges_souscrits"] * 100).round(1))
    d["revenu_mensuel_recurrent_eur"] = d["revenu_mensuel_recurrent_eur"].fillna(
        d["sieges_souscrits"] * d["prix_mensuel_par_siege_eur"])
    d["ratio_fonctionnalites"] = d["fonctionnalites_utilisees"] / d["fonctionnalites_incluses"]
    d["heures_par_utilisateur"] = np.where(d["utilisateurs_actifs"] > 0,
                                           d["heures_usage_30j"] / d["utilisateurs_actifs"].clip(lower=1), 0.0)
    d.loc[d["heures_usage_30j"].isna(), "heures_par_utilisateur"] = np.nan
    d["ecart_sla_support_h"] = d["delai_reponse_support_h"] - d["sla_reponse_h"]
    d["log_mrr"] = np.log1p(d["revenu_mensuel_recurrent_eur"])
    d["sieges_inactifs"] = d["sieges_souscrits"] - d["utilisateurs_actifs"]
    return d


def score_accounts(raw_accounts: pd.DataFrame, model, meta, catalogue: pd.DataFrame, top_k=3) -> pd.DataFrame:
    d = parse_raw(raw_accounts).drop_duplicates()
    d = add_features(d.merge(catalogue, on="plan", how="left"))
    Xs = d[meta["features_numeric"] + meta["features_categorical"]]
    proba = model.predict_proba(Xs)[:, 1]

    # Contributions individuelles : coefficient × valeur transformée (lecture locale du modèle linéaire)
    prep, lr = model.named_steps["prep"], model.named_steps["model"]
    names = prep.get_feature_names_out()
    contrib = pd.DataFrame(prep.transform(Xs) * lr.coef_[0], columns=names, index=d.index)
    num_contrib = contrib[[n for n in names if n in FACTOR_LABELS]]
    reasons = num_contrib.apply(lambda r: ", ".join(FACTOR_LABELS[c] for c in r.nlargest(top_k).index if r[c] > 0), axis=1)

    t = meta["decision_threshold"]
    out = pd.DataFrame({
        "client_id": d["client_id"].values,
        "proba_churn": proba.round(3),
        "a_risque": proba >= t,
        "niveau": pd.cut(proba, [-0.01, t, max(0.6, t + 0.2), 1.0], labels=["faible", "modéré", "élevé"]),
        "facteurs_principaux": reasons.values,
        "mrr_eur": d["revenu_mensuel_recurrent_eur"].round(0).values,
    })
    out["mrr_a_risque_eur"] = (out["proba_churn"] * out["mrr_eur"]).round(0)
    return out.sort_values(["a_risque", "mrr_a_risque_eur"], ascending=False).reset_index(drop=True)
