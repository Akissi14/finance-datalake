# Finance Data Lake — Détection d'anomalies sur les marchés

Projet final du cours **Data Lakes & Data Integration** — EFREI Paris, 2025-2026.

Data lake structuré en trois zones **Raw → Staging → Curated**, alimenté par deux sources
hétérogènes (505 fichiers CSV et une API REST), orchestré par **Apache Airflow**, enrichi par
un **autoencodeur** de détection d'anomalies, et exposé via une **API Gateway FastAPI**.

> 📄 **L'analyse complète — architecture, choix techniques, résultats, benchmarks et
> limites — se trouve dans [`Rapport_technique.pdf`](Rapport_technique.pdf).**
> Ce README explique uniquement **comment lancer le projet**.

---

## 1. Architecture

| Zone | Technologie | Rôle |
|---|---|---|
| **Raw** | MinIO (compatible S3) | Données brutes, immuables, telles qu'ingérées |
| **Staging** | MySQL | Cotations nettoyées, typées, dédupliquées |
| **Curated** | MongoDB | Indicateurs techniques + scores d'anomalie |

**Orchestration** : Apache Airflow (3 DAG) — **API Gateway** : FastAPI (7 endpoints)
**Modèle** : autoencodeur `8 → 6 → 3 → 6 → 8`, sérialisé dans la zone Raw

---

## 2. Sources de données

| Source | Type | Contenu |
|---|---|---|
| S&P 500 (Kaggle) | 505 fichiers CSV | Cotations journalières du S&P 500, 2013-2018 |
| Yahoo Finance | API REST | Cotations récentes, ré-ingérées toutes les heures par Airflow |

**Références :**

- **Dataset** — *S&P 500 stock data*, Cam Nugent, licence **CC0 1.0** (domaine public) :
  <https://www.kaggle.com/datasets/camnugent/sandp500>
- **API** — Yahoo Finance, via la bibliothèque `yfinance` :
  <https://pypi.org/project/yfinance/>

---

## 3. Prérequis

- **Docker** et **Docker Compose**
- **Python ≥ 3.10** et [**uv**](https://docs.astral.sh/uv/)
- Un **compte Kaggle** avec un token API (`~/.kaggle/kaggle.json`, voir *kaggle.com → Settings → API*)

---

## 4. Installation

```bash
git clone https://github.com/Akissi14/finance-datalake.git
cd finance-datalake

uv sync
source .venv/bin/activate          # Windows : .venv\Scripts\activate
```

### Démarrer les trois zones du lac

```bash
docker compose up -d minio mysql mongodb
```

Attendre ~30 s (initialisation de MySQL), puis **valider l'environnement** :

```bash
python src/test_connections.py
```

Sortie attendue :

```
--- Resume ---
Raw     (MinIO)   : OK
Staging (MySQL)   : OK
Curated (MongoDB) : OK
```

### Télécharger le dataset

```bash
# Dataset : https://www.kaggle.com/datasets/camnugent/sandp500
kaggle datasets download camnugent/sandp500 -p data/ --unzip
rm -rf data/individual_stocks_5yr/__MACOSX
```

> ⚠️ L'archive Kaggle contient un **dossier imbriqué**
> (`data/individual_stocks_5yr/individual_stocks_5yr/`) ainsi qu'un dossier résiduel
> `__MACOSX`. Les scripts pointent sur le sous-répertoire effectif.

---

## 5. Exécuter le pipeline

### Option A — manuellement, étape par étape

```bash
python src/unpack_data.py                    # 505 CSV       → Raw
python src/ingest_api.py                     # yfinance      → Raw
python src/load_to_staging.py --source both  # Raw           → Staging
python src/staging_to_curated.py             # Staging       → Curated
python src/train_model.py                    # entraîne l'autoencodeur
python src/apply_model.py                    # score tous les documents
```

⏱️ Compter ~15 min au total (dont 3-5 min pour le chargement des 619 000 cotations).

### Option B — via Airflow

```bash
docker compose up -d postgres airflow-init airflow-webserver airflow-scheduler
```

Interface : <http://localhost:8081> — identifiants `airflow` / `airflow`

| DAG | Cadence | Rôle |
|---|---|---|
| `finance_datalake_pipeline` | manuel | Pipeline complet, de bout en bout |
| `finance_api_scheduled` | `@hourly` | Ré-ingestion périodique de l'API |
| `finance_train_model` | `@weekly` | Ré-entraînement du modèle |

Activer le toggle de **`finance_api_scheduled`** pour l'ingestion automatique.

---

## 6. Lancer l'API Gateway

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Documentation interactive : <http://localhost:8000/docs>

### Endpoints

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/` | GET | Carte des endpoints |
| `/health` | GET | État des trois zones + du modèle |
| `/stats` | GET | Volumétrie de remplissage |
| `/raw` | GET | Inventaire des objets MinIO |
| `/staging` | GET | Cotations MySQL (filtrable, paginé) |
| `/curated` | GET | Documents enrichis (filtre `anomalies_only`) |
| `/ingest` | POST | Ingestion à chaud — implémentation de référence |
| `/ingest_fast` | POST | Ingestion à chaud — implémentation optimisée |

### Exemples

```bash
# État du lac
curl -s http://localhost:8000/health | python -m json.tool

# Volumétrie des trois zones
curl -s http://localhost:8000/stats | python -m json.tool

# Les 10 journées les plus anormales
curl -s "http://localhost:8000/curated?anomalies_only=true&limit=10" | python -m json.tool

# Ingestion à chaud
curl -s -X POST http://localhost:8000/ingest_fast \
  -H "Content-Type: application/json" \
  -d '{"data":{"quotes":[{"ticker":"AAPL","date":"2026-07-15T00:00:00",
       "open":210.5,"high":213.2,"low":208.9,"close":212.4,"volume":55000000}]}}' \
  | python -m json.tool
```

---

## 7. Reproduire le benchmark

```bash
python src/benchmark.py --repetitions 10
```

Compare `/ingest` et `/ingest_fast` sur un lot d'**1 élément** et de **100 éléments**.
L'endpoint optimisé atteint **52 % de gain** sur le lot de 100 (×2,09).

📄 *Analyse détaillée — méthodologie de mesure, loi d'Amdahl, et les deux optimisations
mesurées puis abandonnées — au §8 du rapport technique.*

---

## 8. Interfaces

| Service | URL | Identifiants |
|---|---|---|
| API Gateway | <http://localhost:8000/docs> | — |
| Airflow | <http://localhost:8081> | `airflow` / `airflow` |
| Console MinIO | <http://localhost:9001> | `minioadmin` / `minioadmin` |

---

## 9. Résultats

Le pipeline détecte environ **1 % de journées anormales** sur les 609 000 documents de la
zone Curated — dont le **flash crash du 24 août 2015**, retrouvé par le modèle sur plusieurs
valeurs simultanément, sans qu'aucune information sur cet événement ne lui ait été fournie.

Il a par ailleurs mis au jour un **défaut du jeu de données source** : le dataset Kaggle
n'est pas ajusté des opérations sur titres (splits, spin-offs), qui apparaissent donc comme
des chutes de 50 à 60 %.

📄 *Chiffres précis, validation, limites et pistes d'amélioration : voir le rapport technique.*

> ⚠️ **Sur la reproductibilité** : l'échantillon d'entraînement est tiré par l'opérateur
> `$sample` de MongoDB, qui **n'accepte aucun germe aléatoire**. Une réexécution de
> `train_model.py` produit donc des valeurs *voisines* mais non identiques (seuil ~2,45,
> anomalies ~1 %). Cette limite est documentée au §9.4 du rapport.

---

## 10. Structure du dépôt

```
finance-datalake/
├── src/
│   ├── test_connections.py     Contrôle des trois zones (préalable à tout)
│   ├── unpack_data.py          Source fichier (505 CSV)  → Raw
│   ├── ingest_api.py           Source API (yfinance)     → Raw
│   ├── load_to_staging.py      Raw → Staging  (nettoyage, typage, idempotence)
│   ├── staging_to_curated.py   Staging → Curated  (8 indicateurs techniques)
│   ├── train_model.py          Entraîne l'autoencodeur, sérialise vers MinIO
│   ├── apply_model.py          Score les documents de la zone Curated
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
├── Rapport_technique.pdf       ← analyse complète
└── README.md
```

Chaque script est **autonome**, paramétrable par `argparse`, et exécutable **en dehors
d'Airflow** — condition pour être testable et rejouable.

---

*Projet réalisé dans le cadre du cours Data Lakes & Data Integration — EFREI Paris, 2025-2026.*