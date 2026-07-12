"""
Etape 7 : Orchestration du data lake avec Apache Airflow.

Trois DAG, dont la separation traduit trois rythmes differents :

    finance_datalake_pipeline  (manuel)   le pipeline complet, de bout en bout
    finance_api_scheduled      (@hourly)  la re-ingestion periodique de l'API
    finance_train_model        (@weekly)  le re-entrainement du modele

Pourquoi ne pas tout mettre dans un seul DAG ? Parce que l'entrainement est
couteux et l'inference legere. Re-entrainer un autoencodeur toutes les heures
serait aussi absurde que de ne jamais le re-entrainer. Cette separation
entrainement / inference est la norme en production.

Airflow ORCHESTRE, il ne calcule pas : chaque tache se contente de lancer le
script correspondant. La logique metier reste dans src/, testable et rejouable
en dehors d'Airflow. Un DAG qui contient de la logique metier est un DAG qu'on
ne peut ni tester ni reutiliser.

Les scripts tournent DANS le conteneur Airflow (montes en volume sur
/opt/airflow/scripts). Les services sont donc joints par leur nom de service
Docker (minio, mysql, mongodb), et non par localhost.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# ---------------------------------------------------------------------------
# Configuration commune
# ---------------------------------------------------------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

SCRIPTS = "/opt/airflow/scripts"
DATA_DIR = "/opt/airflow/data/individual_stocks_5yr/individual_stocks_5yr"

# Depuis le conteneur Airflow, les services sont joints par leur nom Docker.
S3 = "--s3-endpoint http://minio:9000 --s3-access-key minioadmin --s3-secret-key minioadmin"
MYSQL = "--db-host mysql --db-user root --db-password root --db-name staging"
MONGO = "--mongo-uri mongodb://mongodb:27017/"


# ---------------------------------------------------------------------------
# DAG 1 : le pipeline complet
# ---------------------------------------------------------------------------

with DAG(
    "finance_datalake_pipeline",
    default_args=default_args,
    description="Pipeline complet : Raw -> Staging -> Curated -> inference",
    schedule=None,  # Declenchement manuel : c'est un traitement de fond
    catchup=False,
    tags=["datalake"],
) as dag_full:

    # Les deux ingestions sont independantes : elles peuvent donc tourner
    # en parallele. Airflow le fait automatiquement des lors qu'aucune
    # dependance ne les relie.
    ingest_dataset = BashOperator(
        task_id="ingest_dataset",
        bash_command=(
            f"python {SCRIPTS}/unpack_data.py "
            f"--input-dir {DATA_DIR} --bucket raw --prefix source_dataset {S3}"
        ),
    )

    ingest_api = BashOperator(
        task_id="ingest_api",
        bash_command=(
            f"python {SCRIPTS}/ingest_api.py "
            f"--period 1mo --interval 1d --bucket raw --prefix source_api {S3}"
        ),
    )

    load_staging = BashOperator(
        task_id="load_to_staging",
        bash_command=(
            f"python {SCRIPTS}/load_to_staging.py --source both --bucket raw {S3} {MYSQL}"
        ),
    )

    load_curated = BashOperator(
        task_id="staging_to_curated",
        bash_command=(f"python {SCRIPTS}/staging_to_curated.py {MYSQL} {MONGO}"),
    )

    # INFERENCE seulement : ce DAG applique un modele existant, il ne
    # l'entraine pas. L'entrainement a son propre DAG, plus bas.
    score_anomalies = BashOperator(
        task_id="apply_model",
        bash_command=(
            f"python {SCRIPTS}/apply_model.py --bucket raw "
            f"--model-key models/autoencoder.pkl {S3} {MONGO}"
        ),
    )

    [ingest_dataset, ingest_api] >> load_staging >> load_curated >> score_anomalies


# ---------------------------------------------------------------------------
# DAG 2 : re-ingestion horaire de l'API
# ---------------------------------------------------------------------------

with DAG(
    "finance_api_scheduled",
    default_args=default_args,
    description="Re-ingestion horaire de l'API Yahoo Finance",
    schedule="@hourly",
    catchup=False,  # Sans cela, Airflow rejouerait tous les creneaux passes
    tags=["datalake", "scheduled"],
) as dag_api:

    # On ne recupere que les 5 derniers jours : inutile de re-telecharger un
    # mois d'historique toutes les heures. La cle unique en base ecarte de
    # toute facon les cotations deja connues.
    hourly_ingest = BashOperator(
        task_id="ingest_api",
        bash_command=(
            f"python {SCRIPTS}/ingest_api.py "
            f"--period 5d --interval 1d --bucket raw --prefix source_api {S3}"
        ),
    )

    # --source api : on ne relit pas les 505 CSV du dataset a chaque heure.
    hourly_staging = BashOperator(
        task_id="load_to_staging",
        bash_command=(
            f"python {SCRIPTS}/load_to_staging.py --source api --bucket raw {S3} {MYSQL}"
        ),
    )

    hourly_curated = BashOperator(
        task_id="staging_to_curated",
        bash_command=(f"python {SCRIPTS}/staging_to_curated.py {MYSQL} {MONGO}"),
    )

    hourly_scoring = BashOperator(
        task_id="apply_model",
        bash_command=(
            f"python {SCRIPTS}/apply_model.py --bucket raw "
            f"--model-key models/autoencoder.pkl {S3} {MONGO}"
        ),
    )

    hourly_ingest >> hourly_staging >> hourly_curated >> hourly_scoring


# ---------------------------------------------------------------------------
# DAG 3 : re-entrainement hebdomadaire du modele
# ---------------------------------------------------------------------------

with DAG(
    "finance_train_model",
    default_args=default_args,
    description="Re-entrainement hebdomadaire de l'autoencodeur",
    schedule="@weekly",
    catchup=False,
    tags=["datalake", "ml"],
) as dag_train:

    # Volontairement isole du pipeline de donnees : l'entrainement est long,
    # et un modele qui change a chaque execution rendrait les scores
    # incomparables d'un jour a l'autre.
    train = BashOperator(
        task_id="train_model",
        bash_command=(
            f"python {SCRIPTS}/train_model.py --bucket raw "
            f"--model-key models/autoencoder.pkl --sample-size 100000 "
            f"--contamination 0.01 {S3} {MONGO}"
        ),
    )