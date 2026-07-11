FROM apache/airflow:2.7.1-python3.10

USER root

# Dossiers utilises par les scripts du pipeline
RUN mkdir -p /opt/airflow/build /opt/airflow/scripts /opt/airflow/data/raw
RUN chown -R airflow:root /opt/airflow/build /opt/airflow/scripts /opt/airflow/data

USER airflow

# Installation des dependances du pipeline
COPY build/reqs.txt /opt/airflow/build/reqs.txt
RUN pip install --no-cache-dir -r /opt/airflow/build/reqs.txt

# Copie des scripts (ils sont aussi montés en volume pour le dev)
COPY src/*.py /opt/airflow/scripts/