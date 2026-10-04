## Cas d'usage 01 : Résiliation client SaaS (churn)

Certification « Concevoir et implémenter une solution d'intelligence artificielle » — Énoncé candidat

| Intitulé du projet | Résiliation client SaaS (churn) |
| --- | --- |
| Type de problème | Apprentissage supervisé — classification binaire (cible principale) + régression (cible secondaire) |
| Jeux de données fournis | churn_saas_complet.csv (~5000 lignes) · churn_saas_echantillon.csv (50 lignes) · catalogue_plans.csv |
| Durée de production | 3 semaines (notebook + support de présentation) |
| Soutenance orale | 1 heure (30 min présentation + 30 min échanges), à distance ou en présentiel |
| Compétences évaluées | Référentiel complet C1 → C9 |

## 1. Contexte général

Vous intervenez auprès d'un éditeur de logiciel SaaS B2B qui commercialise plusieurs formules d'abonnement à destination d'entreprises clientes. Son modèle économique repose sur des revenus récurrents : chaque mois, les clients paient un abonnement en fonction du plan souscrit, du nombre de licences utilisées et de leur niveau d'usage de la plateforme.

Comme beaucoup d'entreprises SaaS, cette organisation est confrontée à un enjeu majeur : la résiliation client, aussi appelée churn. Lorsqu'un client résilie son abonnement à l'échéance, l'entreprise perd non seulement un revenu mensuel, mais aussi une partie de sa valeur client future. Le churn constitue donc un indicateur critique pour la rentabilité, la croissance commerciale et la qualité de la relation client.

L'entreprise souhaite mieux anticiper les comptes susceptibles de résilier afin de permettre aux équipes Customer Success de prioriser leurs actions de rétention. L'objectif n'est pas de remplacer la décision humaine par un algorithme, mais de fournir un outil d'aide à la décision permettant d'identifier les signaux faibles, de hiérarchiser les comptes à risque et de proposer des actions ciblées avant la prochaine échéance contractuelle.

Dans ce contexte, vous devez concevoir une solution d'intelligence artificielle capable d'estimer la probabilité de churn d'un compte client à partir de données d'usage, de facturation, de support et de caractéristiques contractuelles. La cible principale du projet est binaire : le client résilie ou ne résilie pas à l'échéance. Une cible secondaire est également proposée : l'estimation de la valeur vie client, afin d'enrichir l'analyse métier sans l'utiliser comme variable explicative du modèle principal.

Le jeu de données mis à votre disposition contient des informations réalistes, volontairement imparfaites. Il comporte des valeurs manquantes, des formats hétérogènes, des dates à parser, des données numériques stockées sous forme de texte, des doublons, des variables potentiellement inutiles et une ou plusieurs variables présentant un risque de fuite de données. Ces défauts ne sont pas des anomalies du sujet : ils font partie intégrante de l'évaluation. Vous devez les identifier, les traiter, les justifier et en tirer les conséquences méthodologiques.

Votre travail devra démontrer une démarche complète de conception et d'implémentation d'une solution d'IA : compréhension du besoin métier, analyse des données disponibles, préparation des données, choix argumenté d'un modèle, entraînement, évaluation, interprétation des résultats, réflexion éthique, proposition d'architecture cible, conditions de déploiement et amélioration continue.

L'enjeu principal n'est pas d'obtenir le score le plus élevé possible, mais de construire une solution robuste, explicable, pertinente pour le métier et compatible avec une utilisation réelle par des équipes opérationnelles. Un modèle affichant une performance anormalement élevée devra être questionné, car il peut révéler une fuite de données ou une mauvaise séparation entre les informations disponibles avant la décision et les informations connues après coup.


## 2. Données du projet

Le jeu de données principal comporte environ 5000 lignes et 29 colonnes. Il reflète des données réelles : valeurs manquantes, nombres stockés en texte (%, €, unités, virgules décimales), casse et espaces hétérogènes, dates en formats multiples et quelques doublons. Ces défauts sont volontaires et font partie du travail de préparation (C3). Certaines variables sont des LEURRES (sans pouvoir prédictif) et une variable constitue un PIÈGE DE FUITE (résultat connu a posteriori) : à identifier et à exclure.

## 2.1 Dictionnaire de données

| Variable | Type | Description | Exemple / Plage réelle |
| --- | --- | --- | --- |
| client_id | texte | Identifiant unique du compte client | ex. CLI-000001 |
| date_souscription | date | Date de souscription (formats mêlés) | formats mêlés (AAAA-MM-JJ, JJ/MM/AAAA, JJ mois AAAA) |
| jour_souscription | catégoriel | Jour de semaine de la souscription | mardi, jeudi, samedi, dimanche, vendredi, mercredi, lundi |
| secteur | catégoriel | Secteur d'activité du client | Tech, Finance, Commerce, Santé, Industrie, Public, Éducation |
| pays | catégoriel | Pays de facturation | France, Espagne, Canada, Allemagne, Suisse, Belgique |
| taille_entreprise | catégoriel | Segment de taille (TPE/PME/ETI/GE) | TPE, PME, ETI, GE |
| plan | catégoriel | Formule d'abonnement (référentiel catalogue) | Pro, Business, Starter, Enterprise |
| anciennete_mois | entier | Ancienneté du compte en mois | 1 – 36 |
| sieges_souscrits | entier | Nombre de licences souscrites | 1 – 898 |
| utilisateurs_actifs | entier | Utilisateurs actifs (<= sièges) | 0 – 829 |
| taux_adoption_pct | décimal | Taux d'adoption (%) = actifs/sièges | 0.0 – 100.0 |
| connexions_30j | entier | Connexions sur 30 jours | 0 – 156 |
| heures_usage_30j | décimal | Heures d'usage cumulées 30 j | 0.0 – 170.4 |
| fonctionnalites_total | entier | Fonctionnalités offertes par le plan | 8 – 40 |
| fonctionnalites_utilisees | entier | Fonctionnalités effectivement utilisées | 0 – 40 |
| nb_integrations | entier | Intégrations tierces connectées | 0 – 16 |
| derniere_connexion_jours | entier | Jours depuis la dernière connexion | 0 – 200 |
| tickets_support_90j | entier | Tickets support sur 90 j | 0 – 17 |
| delai_reponse_support_h | décimal | Délai moyen de réponse support (h) | 0.5 – 56.5 |
| csat | entier | Satisfaction client (1-5) | 1 – 5 |
| retards_paiement_12m | entier | Retards de paiement sur 12 mois | 0 – 7 |
| revenu_mensuel_recurrent_eur | décimal | MRR — revenu mensuel récurrent (€) | 9.1 – 89402.9 |
| couleur_theme_interface | catégoriel | Thème d'interface choisi | clair, vert, bleu, violet, sombre |
| code_datacenter | catégoriel | Datacenter d'hébergement | eu-w3, us-e1, ap-s1, eu-w1 |
| groupe_experimentation | catégoriel | Bucket A/B test | B, A, control |
| commentaire_csm | texte libre | Note libre du Customer Success Manager | champ libre (souvent vide) |
| sante_compte_fin_periode | entier | Score santé calculé en FIN de période | 0 – 100 |
| valeur_vie_client_eur | numérique (cible régr.) | Valeur vie client estimée (€) | 300 – 2000000 |
| churn | binaire | Résiliation à l'échéance (1) ou non (0) | {0, 1} |


## 3. Travail attendu : Notebook certifiant

Vous devez produire un notebook Jupyter unique, exécutable et lisible sans explication orale, intégrant texte, code, résultats et un journal de bord à chaque grande étape. Le notebook suit le plan imposé par le règlement de l'épreuve et doit démontrer explicitement les neuf compétences du référentiel.

## 3.1 Plan imposé du notebook

0. Page de garde (titre, certification, nom/prénom, date, version, environnement d'exécution) 1. Résumé exécutif (problème, objectif, approche, résultats clés, limites) 2. Cadrage métier et cas d'usage — journal de bord [C1] 3. Données : disponibilité, gouvernance et alternatives — journal de bord [C1] 4. Enjeux éthiques, sociétaux et conformité — journal de bord [C2] 5. Chargement et compréhension des données [C3] 6. Analyse exploratoire (EDA) : visualisations et interprétation — journal de bord [C3] 7. Préparation des données (nettoyage, manquants, transformations, features) — journal de bord [C3] 8. Choix du modèle et démarche scientifique (baseline, modèles, comparaison) — journal de bord [C4] 9. Entraînement, validation et ajustement (sélection du modèle final) — journal de bord [C5] 10. Implémentation et mise en exploitation (déploiement, exemple d'usage) — journal de bord [C6] 11. Architecture cible et contraintes [C7] 12. Mesure de performance et impacts (métriques techniques + métier) [C8] 13. Amélioration continue (ré-entraînement, suivi, versioning) [C9] 14. Conclusion (synthèse et recommandations)

15. Annexes (versions, paramètres, dépendances, fonctions utilitaires)

## 3.2 Exigences techniques minimales

Détecter et supprimer les doublons, documenter le taux de valeurs manquantes et la stratégie d'imputation retenue.

Convertir proprement les variables numériques stockées en texte (%, €, unités, virgules) et parser les dates multi- formats, traiter les valeurs manquantes et les incohérences résiduelles.

Établir une baseline, puis comparer au moins deux familles de modèles ; prévenir toute fuite de données en excluant les identifiants et toute variable de résultat connue seulement a posteriori (mesurée après la décision). Repérer et écarter la ou les variables constituant le piège de fuite.

Traiter la cible principale « churn » en classification : présenter la courbe ROC et l'AUC (exigée par C4), la matrice de confusion et la courbe précision-rappel (PR-AUC, particulièrement informative en classe déséquilibrée), puis justifier le choix du seuil au regard du coût métier d'un faux négatif.

Fournir une lecture explicable des facteurs (importance des variables / permutation) et identifier les variables leurres (sans pouvoir prédictif) sans les sur-interpréter.

Traiter la cible secondaire « valeur_vie_client_eur » en régression (métrique RMSE/MAE, R²) et ne PAS l'utiliser comme variable explicative du modèle principal.


## 4. Organisation de l'évaluation

| Comp. | Intitulé (abrégé) | Attendus concrets sur ce cas d'usage |
| --- | --- | --- |
| C1 | Identifier un jeu de données pertinent | cadrage besoin rétention + gouvernance données d'usage (+ jointure catalogue). |
| C2 | Risques éthiques, sociétaux & conformité | éthique : biais, RGPD, conséquences FP/FN, variables à écarter. |
| C3 | Préparer les données | nettoyage (parsing %/€, casse, doublons, NaN), anti-fuite (exclure santé_fin + id), feature engineering (ratios). |
| C4 | Choisir un modèle (démarche scientifique) | baseline vs modèles, courbe ROC + AUC, seuil justifié (rappel vs précision). |
| C5 | Entraîner le modèle | entraînement/validation croisée, choix du seuil métier. |
| C6 | Implémenter la solution | sérialisation du modèle, esquisse d'API de scoring. |
| C7 | Architecture cible & contraintes | architecture cible (batch scoring mensuel, intégration CRM). |
| C8 | Mesurer performance & impacts | performance technique (AUC/PR) + métier (comptes sauvés) + régression CLV (R²/MAE). |
| C9 | Amélioration continue | MLOps : dérive d'usage, ré-entraînement, monitoring. |

## 5. Livrables obligatoires

Notebook .ipynb exécutable (texte + code + résultats + journal de bord).

Support de présentation (PPT ou PDF) pour la soutenance orale.

Le(s) jeu(x) de données utilisé(s) et, le cas échéant, le modèle sérialisé.

## 6. Points clés à retenir

Aucune variable ne « donne » seule la réponse : la performance vient de la démarche (EDA + préparation + feature engineering).

Un modèle qui atteint une AUC quasi parfaite cache presque toujours une fuite de données : cherchez-la et excluez-la.

Justifiez chaque choix (seuil, métrique, exclusion) au regard du besoin métier et des enjeux éthiques.
