# Finance Data Lake — Détection d'anomalies sur les marchés

Projet final du cours **Data Lakes & Data Integration** — EFREI 2025-2026.

Data lake structuré en trois zones **Raw → Staging → Curated**, alimenté par deux sources
hétérogènes (505 fichiers CSV et une API REST), orchestré par **Apache Airflow**, enrichi par
un **autoencodeur** de détection d'anomalies, et exposé via une **API Gateway FastAPI**.

**Volumétrie réelle** : 684 objets en zone Raw, 619 217 cotations en Staging (505 valeurs du
S&P 500 sur 2013-2026), 609 017 documents enrichis en Curated, **5 794 journées de marché
flaguées comme anormales**.

---

## Sommaire

1. [Architecture](#1-architecture)
2. [Sources de données](#2-sources-de-données)
3. [Pipelines de transformation](#3-pipelines-de-transformation)
4. [Le modèle de détection d'anomalies](#4-le-modèle-de-détection-danomalies)
5. [Orchestration Airflow](#5-orchestration-airflow)
6. [API Gateway](#6-api-gateway)
7. [Niveau avancé : /ingest vs /ingest_fast](#7-niveau-avancé--ingest-vs-ingest_fast)
8. [Installation et build](#8-installation-et-build)
9. [Utilisation](#9-utilisation)
10. [Difficultés rencontrées et enseignements](#10-difficultés-rencontrées-et-enseignements)
11. [Limites connues](#11-limites-connues)
12. [Structure du dépôt](#12-structure-du-dépôt)

---

## 1. Architecture

```
                 ┌──────────────┐         ┌──────────────┐
   Kaggle CSV ──►│              │         │              │
   (505 fichiers)│   ZONE RAW   │         │  ZONE STAGING│
                 │    MinIO     │────────►│    MySQL     │
   yfinance API ►│  (objet S3)  │ nettoyage│  (tabulaire) │
                 └──────────────┘ typage   └──────┬───────┘
                                  dédup            │
                                                   │ indicateurs
                                                   │ techniques
                                                   ▼
   ┌───────────────┐          ┌──────────────────────────┐
   │  API GATEWAY  │◄─────────│      ZONE CURATED        │
   │    FastAPI    │          │        MongoDB           │
   └───────────────┘          │  features + anomalies    │
                              └──────────────────────────┘
                                          ▲
                          autoencodeur ───┘
                          (MLPRegressor)

              Orchestration : Apache Airflow (3 DAG)
```

| Zone | Technologie | Rôle | Justification |
|---|---|---|---|
| **Raw** | MinIO (compatible S3) | Données brutes, immuables, telles qu'ingérées | Le sujet impose S3 ou Elasticsearch. MinIO expose la même API S3 que le TP3, pour une empreinte disque cinq fois moindre (voir §10). |
| **Staging** | MySQL 8.0 | Cotations nettoyées, typées, dédupliquées | Les cotations OHLCV sont régulières et tabulaires. Un SGBD relationnel force le typage et permet une contrainte d'unicité — clé de l'idempotence. |
| **Curated** | MongoDB | Indicateurs techniques + scores d'anomalie | Le document `{ohlcv, features, metadata}` a accueilli un bloc `{anomaly}` **sans migration de schéma**. En SQL, il aurait fallu un `ALTER TABLE`. |

**Modèle sérialisé** : le fichier `models/autoencoder.pkl` est stocké **dans la zone Raw**, aux
côtés des données. Un artefact de ML est une donnée comme une autre : il a sa place dans le lac,
et sa production est reproductible depuis les sources.

---

## 2. Sources de données

Le sujet exige **deux sources** : un dataset fichier et une API.

| Source | Type | Contenu | Volume |
|---|---|---|---|
| **Kaggle `camnugent/sandp500`** | 505 fichiers CSV | Cotations journalières des 505 valeurs du S&P 500, 2013-2018 | 619 040 lignes |
| **Yahoo Finance (`yfinance`)** | API REST | Cotations récentes, ré-ingérées toutes les heures par Airflow | flux continu |

Les deux sources convergent sur le **même schéma** (`date, open, high, low, close, volume, ticker`),
ce qui permet de les faire atterrir dans une **table Staging unique**. C'est précisément le rôle
de cette zone : unifier l'hétérogène.

---

## 3. Pipelines de transformation

### Raw → Staging (`src/load_to_staging.py`)

Quatre responsabilités :

1. **Unifier** — 505 CSV + N payloads JSON → une table `quotes`
2. **Typer** — prix en `DECIMAL(14,4)`, dates en `DATETIME`, volume en `BIGINT`
3. **Nettoyer** — six règles, chacune comptée et journalisée
4. **Dédupliquer** — contrainte `UNIQUE (ticker, quote_date, source)` + `INSERT IGNORE`

**Résultat du nettoyage sur le dataset Kaggle :**

```
lignes brutes            : 619040
apres valeurs manquantes : 619029   (-11)
apres typage             : 619029   (-0)
apres prix invalides     : 619029   (-0)
apres incoherences OHLC  : 619017   (-12)
apres doublons           : 619017   (-0)
```

Les **12 incohérences OHLC** sont des lignes où le plus haut du jour était inférieur au plus bas —
physiquement impossible. Ce sont de vrais défauts du jeu de données source, détectés par le pipeline.

**L'idempotence est démontrable** : relancer le script affiche `Lignes inserees ce run : 0`. C'est
indispensable, puisque Airflow le rejoue toutes les heures. Un pipeline qui duplique ses données à
chaque exécution transforme un data lake en *data swamp* en quelques jours.

### Staging → Curated (`src/staging_to_curated.py`)

Calcul de **huit indicateurs techniques** par valeur :

| Feature | Ce qu'elle capture |
|---|---|
| `daily_return` | variation quotidienne du cours de clôture |
| `ma_5`, `ma_20` | moyennes mobiles court et moyen terme |
| `volatility_20` | écart-type glissant des rendements — cœur de la détection |
| `rsi_14` | momentum : sur-achat (>70) / sur-vente (<30) |
| `volume_ratio` | volume du jour / sa moyenne mobile — un pic précède souvent un krach |
| `hl_range` | amplitude intra-journalière, normalisée par le prix |
| `gap` | écart entre l'ouverture et la clôture de la veille |

**Point de conception** : les indicateurs sont calculés **par couple (ticker, source)**, jamais par
ticker seul. Le dataset s'arrête en 2018 et l'API reprend en 2026 : une moyenne mobile calculée à
cheval sur ces deux périodes enjamberait un trou de huit ans et produirait des valeurs dénuées de sens.

**10 300 lignes sont écartées à l'amorçage** — les 20 premières de chaque groupe, qui n'ont pas
l'historique nécessaire à une fenêtre de 20 jours. C'est un choix assumé : une moyenne mobile
calculée sur 3 points ne serait pas comparable à une moyenne sur 20.

---

## 4. Le modèle de détection d'anomalies

### Principe

Un **autoencodeur** est un réseau de neurones entraîné à reproduire son entrée en sortie, en la
faisant transiter par une couche cachée plus étroite :

```
8 features ──► 6 ──► 3 ──► 6 ──► 8 features
                    ▲
              goulot d'étranglement
```

La compression le contraint à apprendre la structure du comportement **normal**. Sur une journée
ordinaire, il reconstruit fidèlement : l'erreur est faible. Sur un krach flash ou une bulle de
volume, il n'a jamais rien vu de tel : l'erreur explose. **L'erreur de reconstruction *est* le
score d'anomalie.**

Implémenté avec `MLPRegressor` de scikit-learn, entraîné avec `X` en entrée **et** en cible.

### Le piège du choix des features

Les indicateurs de la zone Curated ne sont **pas tous utilisables tels quels**. `ma_5` et `ma_20`
sont des **prix absolus** : Amazon cote plusieurs milliers de dollars, Coca-Cola quelques dizaines.
Un modèle global apprendrait surtout « Amazon est chère », pas « cette journée est anormale ».

Huit features **sans échelle** sont donc dérivées : rendements, ratios, indicateurs bornés
(`ma_ratio = ma_5/ma_20`, `price_to_ma20 = close/ma_20`, `rsi/100`, etc.). C'est la condition pour
qu'un modèle unique ait du sens sur les 505 valeurs.

Une **standardisation** (`StandardScaler`) précède l'entraînement : sans elle, la volatilité
(≈ 0,01) pèserait mille fois moins que le `volume_ratio` (≈ 1) dans l'erreur quadratique, et le
modèle serait aveugle au signal qui nous intéresse.

### Séparation entraînement / inférence

Deux scripts distincts, comme en production :

- `src/train_model.py` — entraîne sur un **échantillon** de 100 000 documents, calcule le seuil, sérialise vers MinIO. Tourne en `@weekly`.
- `src/apply_model.py` — recharge l'artefact et score **l'intégralité** des 609 017 documents. Tourne à chaque pipeline.

### Résultats

```
Convergence en 27 epoques
Erreur de reconstruction moyenne : 0.25344
Erreur mediane                   : 0.10206
Erreur maximale                  : 94.59465
Seuil (contamination 1 %)        : 2.44416
Anomalies detectees              : 5 794 / 609 017 (0.95 %)
```

Un facteur **900 entre l'erreur médiane et l'erreur maximale** : c'est la signature recherchée. La
grande majorité des journées se ressemblent, une poignée est radicalement hors norme.

### Validation par le bon sens — et une découverte

Les journées les plus anormales détectées :

| Ticker | Date | Score | Rendement | Volume/moy |
|---|---|---|---|---|
| XL | 2015-08-24 | 311,2 | −3,6 % | 1,6 |
| HCA | 2015-08-24 | 94,6 | −2,3 % | 2,1 |
| NI | 2015-07-02 | 92,9 | −62,6 % | 5,0 |
| EBAY | 2015-07-20 | 89,1 | −56,9 % | 2,7 |
| SYY | 2013-12-09 | 75,8 | +9,7 % | 11,9 |
| AOS | 2017-07-25 | 69,9 | −0,5 % | 14,1 |

**Deux valeurs, la même date : XL et HCA, le 24 août 2015.** C'est le *flash crash* du 24 août 2015 —
le Dow Jones a perdu 1 000 points à l'ouverture. Le modèle l'a retrouvé seul, sans qu'on lui ait
jamais indiqué qu'une telle journée existait. **C'est la validation du modèle.**

**Mais les rendements à −62 % et −57 % ne sont pas des anomalies de marché.** Vérification faite :

- `NI`, 2 juillet 2015 → NiSource **scinde** Columbia Pipeline Group
- `EBAY`, 20 juillet 2015 → eBay **détache** PayPal
- `BAX`, 1er juillet 2015 → Baxter **scinde** Baxalta
- `DISCA`/`DISCK`, 7 août 2014 → **split** d'actions

Ce sont des **opérations sur titres**. Le prix chute mécaniquement parce que l'action a été divisée
ou qu'une partie de l'entreprise en est sortie. **Le dataset Kaggle n'est pas ajusté des splits.**

Le pipeline a donc mis au jour un **défaut de qualité du jeu de données source** que sa description
ne mentionne pas. C'est un résultat en soi. Piste de correction : filtrer les variations supérieures
à 30 % non confirmées par un pic de volume, ou croiser avec un référentiel d'opérations sur titres.

Enfin, les valeurs les plus souvent anormales — `CHK` (Chesapeake Energy), `FCX` (Freeport-McMoRan),
`MRO` (Marathon Oil) — sont précisément les plus exposées à **l'effondrement pétrolier de 2014-2016**.
Cohérent.

---

## 5. Orchestration Airflow

Trois DAG, dont la séparation traduit **trois rythmes différents** :

| DAG | Cadence | Rôle |
|---|---|---|
| `finance_datalake_pipeline` | manuel | Pipeline complet : les 2 ingestions **en parallèle**, puis Staging → Curated → inférence |
| `finance_api_scheduled` | **`@hourly`** | Ré-ingestion périodique de l'API — exigence explicite du sujet |
| `finance_train_model` | `@weekly` | Ré-entraînement de l'autoencodeur |

**Pourquoi trois DAG et pas un ?** Parce que l'entraînement est coûteux et l'inférence légère.
Ré-entraîner un autoencodeur toutes les heures serait aussi absurde que de ne jamais le
ré-entraîner. Cette séparation entraînement / inférence est la norme en production.

**Airflow orchestre, il ne calcule pas** : chaque tâche se contente de lancer le script correspondant
via un `BashOperator`. La logique métier reste dans `src/`, testable et rejouable **en dehors**
d'Airflow. Un DAG qui contient de la logique métier est un DAG qu'on ne peut ni tester ni réutiliser.

Deux réglages qui comptent :

- **`catchup=False`** — sans lui, Airflow rejouerait *tous* les créneaux horaires depuis le `start_date`, soit des milliers d'exécutions d'un coup.
- **`--period 5d` dans le DAG horaire** (contre `1mo` dans le complet) — on ne re-télécharge pas un mois d'historique toutes les heures. La clé d'unicité en base écarte de toute façon ce qui est déjà connu.

---

## 6. API Gateway

FastAPI, sept endpoints. Documentation interactive auto-générée sur **`/docs`**.

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/` | GET | Carte des endpoints |
| `/health` | GET | État des trois zones + du modèle |
| `/stats` | GET | Volumétrie de remplissage des trois zones |
| `/raw` | GET | Inventaire des objets MinIO (filtrable par préfixe) |
| `/staging` | GET | Cotations MySQL (filtrable par ticker, source ; paginé) |
| `/curated` | GET | Documents enrichis MongoDB (filtrable, `anomalies_only`) |
| `/ingest` | POST | Ingestion à chaud — implémentation de référence |
| `/ingest_fast` | POST | Ingestion à chaud — implémentation optimisée |

Les filtres SQL passent par des **requêtes paramétrées** (`%s`), jamais par concaténation : une
concaténation ouvrirait la porte à l'injection SQL.

---

## 7. Niveau avancé : /ingest vs /ingest_fast

### Ce que font les deux endpoints

**Rigoureusement le même travail** :

1. valider le lot de cotations reçu (Pydantic) ;
2. archiver le payload brut dans la zone Raw ;
3. calculer les 8 features du modèle — ce qui suppose de récupérer **l'historique récent** de chaque valeur ;
4. scorer avec l'autoencodeur ;
5. écrire les documents enrichis dans la zone Curated.

Seule la **manière** diffère.

### Les optimisations retenues

| | `/ingest` | `/ingest_fast` |
|---|---|---|
| **Archivage** | bloquant, avant tout calcul | **concurrent** (`ThreadPoolExecutor`) |
| **Historique** | 1 requête Mongo par cotation | mêmes requêtes indexées, **en parallèle** |
| **Scoring** | 1 `predict()` par ligne | **1 `predict()`** sur la matrice complète |
| **Erreur de reconstruction** | boucle Python interprétée | **Numba** (`@njit(parallel=True)`, compilé) |
| **Écriture** | 1 `update_one` par document | **1 `bulk_write`** |

### Mesures

M�dianes sur **10 répétitions**, après *warm-up* (la première requête paie l'établissement des pools
de connexions et la compilation Numba — l'inclure fausserait la mesure).

| Lot | `/ingest` | `/ingest_fast` | **Gain** | Facteur |
|---|---|---|---|---|
| **1 élément** | 19,3 ms | 20,6 ms | ≈ 0 % | ×0,94 |
| **100 éléments** | 179,3 ms | **85,9 ms** | **52,1 %** | **×2,09** |

**Objectif du sujet (≥ 30 %) atteint sur le lot de 100.**

### Analyse

Le gain **croît avec la taille du lot**, ce qui est attendu : les optimisations amortissent des
**coûts fixes payés une fois par LOT**, non une fois par élément. Sur une cotation isolée, il n'y a
rien à paralléliser ni à grouper — les deux implémentations font strictement le même travail, et les
mesures le confirment (minima : 16,3 ms contre 16,7 ms, soit du bruit, pas un signal).

### Deux optimisations mesurées puis ABANDONNÉES

Ce sont les résultats les plus instructifs du projet.

#### Échec n°1 — l'agrégation MongoDB unique : **4,7× plus lente**

L'intuition de départ : remplacer les N requêtes par **une seule agrégation** (`$match` + `$sort` +
`$group`) couvrant tous les tickers du lot.

**Mesure : 849 ms contre 179 ms.** Presque cinq fois pire.

La raison : pour 100 tickers d'environ 1 200 cotations chacun, MongoDB devait **trier 120 000
documents** et empiler 120 000 valeurs en mémoire, pour n'en conserver ensuite que 20 par ticker.
On jetait 99 % du travail. La version de référence, elle, émet des `find().sort().limit(20)` que
l'index `(ticker, quote_date)` résout instantanément : elle ne lit **jamais que 20 documents**.

> **Enseignement : « une requête au lieu de cent » n'est pas une optimisation en soi. Ce qui compte
> est le VOLUME RÉELLEMENT TRAITÉ, pas le nombre d'appels.**

#### Échec n°2 — la parallélisation systématique : **−40 % sur le lot unitaire**

Après correction, la parallélisation par threads donnait +56 % sur 100 éléments… mais **−40 % sur
1 élément**.

La raison : créer un `ThreadPoolExecutor`, démarrer un thread, le synchroniser et le détruire coûte
plusieurs millisecondes — pour paralléliser **une seule** requête. Le coût de coordination dépassait
le travail à paralléliser. C'est la **loi d'Amdahl** appliquée à un cas concret : le surcoût du
parallélisme est fixe, il ne s'amortit qu'au-delà d'un certain volume.

Correction : un **seuil de bascule** — en dessous de 2 tickers, on appelle directement la fonction
séquentielle.

> **Enseignement : une optimisation doit toujours être MESURÉE, jamais supposée.**

---

## 8. Installation et build

### Prérequis

- Docker et Docker Compose
- Python ≥ 3.10 et [uv](https://docs.astral.sh/uv/)
- Un compte Kaggle (pour télécharger le dataset)

### Étapes

```bash
# 1. Cloner
git clone https://github.com/Akissi14/finance-datalake.git
cd finance-datalake

# 2. Environnement Python
uv sync
source .venv/bin/activate          # Windows : .venv\Scripts\activate

# 3. Démarrer les trois zones du lac
docker compose up -d minio mysql mongodb

# 4. Vérifier que tout répond (attendre ~30 s l'init de MySQL)
python src/test_connections.py
#   Raw     (MinIO)   : OK
#   Staging (MySQL)   : OK
#   Curated (MongoDB) : OK

# 5. Télécharger le dataset Kaggle
#    (nécessite ~/.kaggle/kaggle.json — voir kaggle.com > Settings > API)
kaggle datasets download camnugent/sandp500 -p data/ --unzip
rm -rf data/individual_stocks_5yr/__MACOSX
```

> **Note sur le dataset** : l'archive Kaggle contient un dossier **imbriqué**
> (`data/individual_stocks_5yr/individual_stocks_5yr/`) ainsi qu'un dossier résiduel `__MACOSX`.
> Les scripts pointent sur le sous-répertoire effectif.

### Exécuter le pipeline manuellement

```bash
python src/unpack_data.py                    # 505 CSV       → Raw
python src/ingest_api.py                     # yfinance      → Raw
python src/load_to_staging.py --source both  # Raw           → Staging
python src/staging_to_curated.py             # Staging       → Curated
python src/train_model.py                    # entraîne l'autoencodeur
python src/apply_model.py                    # score les 609 017 documents
```

### Lancer Airflow

```bash
docker compose up -d postgres airflow-init airflow-webserver airflow-scheduler
```

Interface : <http://localhost:8081> — identifiants `airflow` / `airflow`.

Activer le DAG `finance_api_scheduled` (toggle) pour l'ingestion horaire automatique.

### Lancer l'API Gateway

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Documentation interactive : <http://localhost:8000/docs>

### Interfaces disponibles

| Service | URL | Identifiants |
|---|---|---|
| API Gateway | <http://localhost:8000/docs> | — |
| Airflow | <http://localhost:8081> | `airflow` / `airflow` |
| Console MinIO | <http://localhost:9001> | `minioadmin` / `minioadmin` |

---

## 9. Utilisation

### Lecture

```bash
# État des trois zones
curl -s http://localhost:8000/health | python -m json.tool

# Volumétrie
curl -s http://localhost:8000/stats | python -m json.tool

# Objets de la zone Raw
curl -s "http://localhost:8000/raw?prefix=source_api/&limit=5" | python -m json.tool

# Cotations d'Apple en Staging
curl -s "http://localhost:8000/staging?ticker=AAPL&limit=5" | python -m json.tool

# Les 10 journées les plus anormales
curl -s "http://localhost:8000/curated?anomalies_only=true&limit=10" | python -m json.tool
```

### Ingestion à chaud

```bash
curl -s -X POST http://localhost:8000/ingest_fast \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "quotes": [
        {
          "ticker": "AAPL",
          "date": "2026-07-15T00:00:00",
          "open": 210.5,
          "high": 213.2,
          "low": 208.9,
          "close": 212.4,
          "volume": 55000000
        }
      ]
    }
  }' | python -m json.tool
```

### Reproduire le benchmark

```bash
python src/benchmark.py --repetitions 10
```

---

## 10. Difficultés rencontrées et enseignements

Cette section documente les problèmes **réels** rencontrés en construisant ce lac. Chacun a
entraîné une décision technique explicite.

### `localstack:latest` bascule en édition payante

Le TP utilise `localstack/localstack:latest` pour la zone Raw. Au moment de ce projet, ce tag pointe
désormais sur l'**édition Pro**, qui exige une licence : le conteneur démarre, réclame un token et
s'arrête (`exit code 55`).

**Décision** : passage à **MinIO**, également compatible S3 (l'exigence du sujet est « Elasticsearch
ou un S3 », pas LocalStack en particulier). Le code `boto3` est **rigoureusement identique** — seul
l'endpoint change. Bénéfice collatéral : MinIO pèse ~100 Mo contre 636 Mo pour LocalStack, qui
embarque un JDK complet pour émuler tous les services AWS.

> **Enseignement : épingler les versions d'images est une exigence de reproductibilité, pas une
> coquetterie.** Un `:latest` peut changer de licence du jour au lendemain.

### `torch` traîne 3 Go de CUDA inutile

L'installation de PyTorch dans l'image Airflow a saturé le disque : pip résout par défaut la version
**GPU**, qui embarque tous les paquets NVIDIA — dont un fichier de **731 Mo**. Pour un conteneur
sans carte graphique.

**Décision** : l'autoencodeur est implémenté avec **`MLPRegressor` de scikit-learn**, déjà présent.
Un autoencodeur est un réseau entraîné à reconstruire son entrée avec une couche cachée étroite —
`MLPRegressor(hidden_layer_sizes=(6,3,6))` entraîné avec `X` en entrée et en cible, c'est
exactement cela : un vrai réseau de neurones, rétropropagation comprise. Gain : ~2,5 Go d'image
en moins, build de 2 min au lieu de 15, et une stack qui tourne sur n'importe quelle machine.

### Trois images Airflow construites pour un seul Dockerfile

Le `docker-compose.yml` faisait construire une image **par service Airflow** (`init`, `webserver`,
`scheduler`) — trois fois ~2 Go pour un contenu identique.

**Décision** : un seul `build:` sur `airflow-init`, un `image:` partagé par les deux autres.

### Le modèle `pickle` est illisible dans le conteneur

```
ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator module.
```

Le modèle avait été entraîné dans le venv local (**NumPy 2.4**) et le conteneur Airflow tentait de
le dépickler avec **NumPy 1.26**. Or l'image Airflow 2.7.1 est figée sous **Python 3.10**, et
NumPy ≥ 2.4 exige Python ≥ 3.11 : **aligner le conteneur sur le venv est impossible**.

**Décision** : le modèle est désormais **entraîné dans l'environnement même qui le sert** (le
conteneur Airflow). C'est d'ailleurs la bonne pratique : un artefact de ML n'est valide qu'avec le
runtime qui l'a produit.

> **Enseignement : `pickle` couple le modèle à ses versions exactes de bibliothèques.** Une solution
> industrielle passerait par un format neutre (ONNX, PMML) ou un registre de modèles (MLflow)
> versionnant conjointement le modèle et son environnement.

### Divergence de schéma entre l'API et le pipeline batch

La zone Curated est alimentée par **deux chemins** : le pipeline batch et l'endpoint `/ingest`. Une
divergence entre les deux (`ma_5`/`ma_20` d'un côté, `ma_ratio`/`price_to_ma20` de l'autre) a fait
échouer le ré-entraînement — `train_model.py`, qui échantillonne au hasard, est tombé sur un
document incompatible (`KeyError: 'ma_20'`).

**Décision** : l'API reconstitue et écrit le **même schéma** que le pipeline batch.

> **Enseignement : l'absence de schéma imposé par MongoDB ne supprime pas le besoin de contrat —
> elle en déplace la responsabilité du moteur vers l'application.** Une base relationnelle aurait
> rejeté l'écriture ; ici, elle est passée silencieusement et n'a explosé que trois étapes plus loin.

### Saturation mémoire sur 600 000 documents

Construire les 608 917 dictionnaires imbriqués **avant** de les insérer faisait tuer le processus par
l'OS (`Terminated`).

**Décision** : traitement **par tranches** — construire 20 000 documents, les insérer, libérer,
recommencer. L'empreinte mémoire devient constante, quel que soit le volume total.

### Permissions sur les volumes Airflow

Airflow tourne sous l'UID **50000** dans son conteneur et ne pouvait pas écrire dans le dossier
`./logs` monté depuis l'hôte — les ACL POSIX de l'environnement écrasant même un `chown` explicite.

**Décision** : les logs ne sont plus montés sur l'hôte. Ils restent consultables dans l'interface
web et via `docker compose logs`.

---

## 11. Limites connues

| Limite | Description | Piste d'amélioration |
|---|---|---|
| **Traitement full-refresh** | À chaque exécution, `staging_to_curated` recalcule les indicateurs sur les 619 217 lignes et `apply_model` re-score les 609 017 documents — alors que 99,9 % n'ont pas changé. On refait tout le travail pour ~50 nouvelles cotations. | Traitement **incrémental** : ne retraiter que les lignes postérieures au dernier `loaded_at`. |
| **Dataset non ajusté des splits** | Les opérations sur titres (splits, spin-offs) apparaissent comme des chutes de 50-60 % et sont flaguées à tort comme anomalies (voir §4). | Filtrer les variations > 30 % non confirmées par un pic de volume, ou croiser avec un référentiel d'opérations sur titres. |
| **Secrets en clair** | Les identifiants (`root`/`root`, `minioadmin`) sont en dur dans le `docker-compose.yml`. Acceptable en local, inacceptable en production. | Gestionnaire de secrets, ou Connections/Variables d'Airflow. |
| **Modèle sérialisé par `pickle`** | Couplage fort aux versions de NumPy et scikit-learn (voir §10). | ONNX, ou un registre de modèles type MLflow. |
| **Seuil de contamination arbitraire** | Le taux de 1 % est une **hypothèse métier**, pas une vérité statistique. On ne connaît pas la « vraie » proportion d'anomalies. | Validation sur un jeu d'événements de marché étiquetés. |
| **Pas de tests automatisés** | Les scripts sont validés manuellement et par leurs contrôles internes (`validate_data`, `verify_mongodb`, comptages croisés). | `pytest` + fixtures sur conteneurs éphémères. |

---

## 12. Structure du dépôt

```
finance-datalake/
├── src/
│   ├── test_connections.py     Contrôle des trois zones (préalable à tout)
│   ├── unpack_data.py          Source fichier (505 CSV)  → Raw
│   ├── ingest_api.py           Source API (yfinance)     → Raw
│   ├── load_to_staging.py      Raw → Staging  (nettoyage, typage, idempotence)
│   ├── staging_to_curated.py   Staging → Curated  (8 indicateurs techniques)
│   ├── train_model.py          Entraîne l'autoencodeur, sérialise vers MinIO
│   ├── apply_model.py          Score les 609 017 documents
│   └── benchmark.py            Mesure /ingest vs /ingest_fast
├── api/
│   └── main.py                 API Gateway FastAPI (7 endpoints)
├── dags/
│   └── pipeline.py             Les 3 DAG Airflow
├── build/
│   └── reqs.txt                Dépendances de l'image Airflow
├── docker-compose.yml          MinIO, MySQL, MongoDB, Postgres, Airflow
├── dockerfile                  Image Airflow + dépendances du pipeline
├── pyproject.toml
└── README.md
```

Chaque script est autonome, paramétrable par `argparse`, et exécutable **en dehors d'Airflow** —
condition pour être testable et rejouable.

---

*Projet réalisé dans le cadre du cours Data Lakes & Data Integration — EFREI Paris, 2025-2026.*