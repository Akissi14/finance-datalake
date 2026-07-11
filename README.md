# Finance Data Lake — Détection d'anomalies sur les marchés

Projet final du cours **Data Lakes & Data Integration** (EFREI 2025-2026).

Data lake structuré en trois zones **Raw → Staging → Curated**, alimenté par deux
sources (un dataset de fichiers et une API), orchestré par Apache Airflow, enrichi
par un autoencodeur de détection d'anomalies, et exposé via une API Gateway FastAPI.

---

## 1. Architecture

| Zone | Technologie | Rôle |
|---|---|---|
| **Raw** | LocalStack (S3) | Données brutes telles qu'ingérées, sans transformation |
| **Staging** | MySQL | Cotations nettoyées, typées, dédupliquées |
| **Curated** | MongoDB | Indicateurs techniques + scores d'anomalie |

**Orchestration** : Apache Airflow — **API Gateway** : FastAPI

### Sources de données

| Source | Type | Détail |
|---|---|---|
| S&P 500 (Kaggle) | Fichiers | 505 CSV, cotations journalières sur 5 ans |
| Yahoo Finance (yfinance) | API | Cotations récentes, ingérées périodiquement par Airflow |

---

## 2. Prérequis

- Docker et Docker Compose
- Python 3.10+ et [uv](https://docs.astral.sh/uv/)

---

## 3. Installation

```bash
uv sync
source .venv/bin/activate        # Windows : .venv\Scripts\activate

docker compose up -d             # démarre les 3 zones + Airflow
docker compose ps                # vérifier que tout tourne
```

Attendre ~30 s l'initialisation de MySQL, puis valider l'environnement :

```bash
python src/test_connections.py
```

Les trois zones doivent remonter `OK`.

Interfaces : Airflow sur <http://localhost:8081> (`airflow` / `airflow`).

---

## 4. Structure du dépôt

```
finance-datalake/
├── src/                    # Scripts du pipeline (une étape = un script)
│   └── test_connections.py # Contrôle de l'environnement (3 zones)
├── dags/                   # DAG Airflow
├── build/reqs.txt          # Dépendances de l'image Airflow
├── api/                    # API Gateway FastAPI
├── docker-compose.yml      # Les 3 zones + Airflow
├── dockerfile              # Image Airflow + dépendances du pipeline
└── pyproject.toml
```

---

*Projet réalisé dans le cadre du cours Data Lakes & Data Integration — EFREI 2025-2026.*