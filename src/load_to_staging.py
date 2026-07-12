"""
Etape 4 : Transformation Raw (MinIO) -> Staging (MySQL).

La zone Staging est la premiere ou la donnee est reellement travaillee. Elle
remplit quatre roles :

    1. UNIFIER  : les 505 CSV du dataset et les payloads JSON de l'API
                  atterrissent dans une table unique, au meme schema.
    2. TYPER    : les prix deviennent des DECIMAL, les dates des DATETIME.
    3. NETTOYER : lignes incompletes, prix negatifs, incoherences OHLC.
    4. DEDUPLIQUER : une contrainte d'unicite en base garantit qu'un rejeu
                  du script n'insere aucun doublon (idempotence).

Ce dernier point est essentiel : Airflow relancera ce script periodiquement.
Un script qui duplique ses donnees a chaque execution transforme le data lake
en data swamp.

Usage:
    python src/load_to_staging.py --source both
    python src/load_to_staging.py --source api        # rejeu leger
"""
import argparse
import io
import json
from datetime import datetime

import boto3
import mysql.connector
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

# Colonnes attendues en sortie de nettoyage, dans l'ordre d'insertion
COLUMNS = ["ticker", "quote_date", "open", "high", "low", "close", "volume", "source"]


def get_s3_client(endpoint_url, access_key, secret_key):
    """Cree un client S3 pointant vers la zone Raw (MinIO)."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def list_keys(s3, bucket, prefix):
    """
    Liste les cles presentes sous un prefixe de la zone Raw.

    La pagination est necessaire : S3 ne renvoie que 1000 cles par appel, et
    nous en avons 505 rien que pour le dataset.

    Returns
    -------
    list of str
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys = []

    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    return keys


def read_dataset_from_raw(s3, bucket, prefix):
    """
    Lit les CSV du dataset S&P 500 depuis la zone Raw.

    Chaque CSV correspond a une valeur et contient les colonnes
    date, open, high, low, close, volume, Name.

    Returns
    -------
    pandas.DataFrame
        Concatenation brute de tous les CSV (avant nettoyage).
    """
    keys = list_keys(s3, bucket, prefix)
    print(f"  {len(keys)} objets trouves sous '{prefix}/'.")

    frames = []
    for i, key in enumerate(keys, start=1):
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            frames.append(pd.read_csv(io.BytesIO(body)))
        except (BotoCoreError, ClientError, pd.errors.ParserError) as e:
            # Un fichier corrompu ne doit pas faire echouer les 504 autres.
            print(f"  IGNORE {key} : {e}")

        if i % 100 == 0:
            print(f"  {i} objets lus...")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def read_api_from_raw(s3, bucket, prefix):
    """
    Lit les payloads JSON de l'API depuis la zone Raw.

    Chaque objet contient des metadonnees d'ingestion et une liste
    d'enregistrements deja alignes sur le schema du dataset.

    Returns
    -------
    pandas.DataFrame
    """
    keys = list_keys(s3, bucket, prefix)
    print(f"  {len(keys)} objets trouves sous '{prefix}/'.")

    frames = []
    for key in keys:
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            payload = json.loads(body)
            records = payload.get("records", [])

            if records:
                frames.append(pd.DataFrame(records))
            else:
                print(f"  IGNORE {key} : payload sans enregistrements.")

        except (BotoCoreError, ClientError, json.JSONDecodeError) as e:
            print(f"  IGNORE {key} : {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def clean(df, source):
    """
    Nettoie et type un lot de cotations brutes.

    Les regles appliquees, dans l'ordre :

        1. Renommage vers le schema cible (Name -> ticker, date -> quote_date).
        2. Suppression des lignes ou une valeur essentielle manque.
        3. Typage : dates en datetime, prix en float, volume en entier.
        4. Suppression des prix nuls ou negatifs (impossibles en bourse).
        5. Suppression des incoherences OHLC : le plus haut du jour ne peut pas
           etre inferieur au plus bas. Cela arrive dans les vraies donnees.
        6. Suppression des doublons internes au lot.

    Chaque regle est comptee et affichee : un nettoyage silencieux est un
    nettoyage qu'on ne peut pas auditer.

    Parameters
    ----------
    df : pandas.DataFrame
        Donnees brutes issues de la zone Raw.
    source : str
        Identifiant de la source ('dataset' ou 'api').

    Returns
    -------
    pandas.DataFrame
        Donnees propres, pretes a l'insertion.
    """
    if df.empty:
        return df

    initial = len(df)

    # 1. Schema cible
    df = df.rename(columns={"Name": "ticker", "date": "quote_date"})

    required = ["ticker", "quote_date", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes absentes dans la source '{source}': {missing_cols}")

    # 2. Lignes incompletes
    df = df.dropna(subset=required)
    after_na = len(df)

    # 3. Typage
    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # 'coerce' transforme l'invalide en NaT/NaN : on les retire ici
    df = df.dropna(subset=required)
    after_types = len(df)

    df["volume"] = df["volume"].astype("int64")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # 4. Prix impossibles
    price_cols = ["open", "high", "low", "close"]
    df = df[(df[price_cols] > 0).all(axis=1)]
    after_prices = len(df)

    # 5. Incoherences OHLC
    df = df[(df["high"] >= df["low"])]
    df = df[(df["high"] >= df["open"]) & (df["high"] >= df["close"])]
    df = df[(df["low"] <= df["open"]) & (df["low"] <= df["close"])]
    after_ohlc = len(df)

    # 6. Doublons internes
    df = df.drop_duplicates(subset=["ticker", "quote_date"])
    after_dups = len(df)

    df["source"] = source

    print(f"  Nettoyage de la source '{source}' :")
    print(f"    lignes brutes            : {initial}")
    print(f"    apres valeurs manquantes : {after_na}   (-{initial - after_na})")
    print(f"    apres typage             : {after_types}   (-{after_na - after_types})")
    print(f"    apres prix invalides     : {after_prices}   (-{after_types - after_prices})")
    print(f"    apres incoherences OHLC  : {after_ohlc}   (-{after_prices - after_ohlc})")
    print(f"    apres doublons           : {after_dups}   (-{after_ohlc - after_dups})")

    return df[COLUMNS]


def create_mysql_connection(host, user, password, database):
    """
    Cree une connexion MySQL.

    Returns
    -------
    mysql.connector.MySQLConnection or None
    """
    try:
        return mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
    except mysql.connector.Error as e:
        print(f"ERREUR : connexion MySQL impossible ({e}).")
        return None


def create_table(connection):
    """
    Cree la table 'quotes' si elle n'existe pas.

    La contrainte UNIQUE (ticker, quote_date, source) est le coeur de
    l'idempotence : combinee a INSERT IGNORE, elle garantit qu'un rejeu du
    script n'insere aucun doublon, quel que soit le nombre d'executions.
    """
    query = """
        CREATE TABLE IF NOT EXISTS quotes (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            ticker      VARCHAR(10)   NOT NULL,
            quote_date  DATETIME      NOT NULL,
            open        DECIMAL(14,4) NOT NULL,
            high        DECIMAL(14,4) NOT NULL,
            low         DECIMAL(14,4) NOT NULL,
            close       DECIMAL(14,4) NOT NULL,
            volume      BIGINT        NOT NULL,
            source      VARCHAR(20)   NOT NULL,
            loaded_at   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uk_quote (ticker, quote_date, source),
            INDEX idx_ticker_date (ticker, quote_date)
        )
    """
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()
    cursor.close()
    print("  Table 'quotes' prete.")


def insert_quotes(connection, df, batch_size=5000):
    """
    Insere les cotations par lots, sans creer de doublons.

    INSERT IGNORE s'appuie sur la contrainte UNIQUE : une ligne deja presente
    est silencieusement ecartee. C'est ce qui rend le script rejouable.

    Returns
    -------
    int
        Nombre de lignes reellement inserees (hors doublons ecartes).
    """
    if df.empty:
        return 0

    query = """
        INSERT IGNORE INTO quotes
            (ticker, quote_date, open, high, low, close, volume, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    rows = [
        (
            r.ticker,
            r.quote_date.to_pydatetime(),
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            int(r.volume),
            r.source,
        )
        for r in df.itertuples(index=False)
    ]

    cursor = connection.cursor()
    inserted = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        cursor.executemany(query, batch)
        inserted += cursor.rowcount
        connection.commit()
        print(f"    {min(i + batch_size, len(rows))}/{len(rows)} lignes traitees...")

    cursor.close()
    return inserted


def validate_data(connection):
    """
    Controle qualite de la zone Staging.

    Verifie ce qui est reellement en base, pas ce qu'on croit y avoir mis :
    volumetrie par source, couverture temporelle, absence de valeurs nulles.
    """
    cursor = connection.cursor()

    cursor.execute("SELECT source, COUNT(*) FROM quotes GROUP BY source")
    print("\n  Lignes par source :")
    for source, count in cursor.fetchall():
        print(f"    {source:<10} {count}")

    cursor.execute("SELECT COUNT(DISTINCT ticker) FROM quotes")
    print(f"\n  Tickers distincts : {cursor.fetchone()[0]}")

    cursor.execute("SELECT MIN(quote_date), MAX(quote_date) FROM quotes")
    start, end = cursor.fetchone()
    print(f"  Couverture        : {start} -> {end}")

    cursor.execute("SELECT COUNT(*) FROM quotes WHERE close IS NULL OR volume IS NULL")
    print(f"  Valeurs nulles    : {cursor.fetchone()[0]}")

    cursor.close()


def main():
    parser = argparse.ArgumentParser(
        description="Charge la zone Raw (MinIO) dans la zone Staging (MySQL)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="both",
        choices=["dataset", "api", "both"],
        help="Source(s) a charger.",
    )
    parser.add_argument("--bucket", type=str, default="raw")
    parser.add_argument("--dataset-prefix", type=str, default="source_dataset")
    parser.add_argument("--api-prefix", type=str, default="source_api")
    parser.add_argument("--s3-endpoint", type=str, default="http://localhost:9000")
    parser.add_argument("--s3-access-key", type=str, default="minioadmin")
    parser.add_argument("--s3-secret-key", type=str, default="minioadmin")
    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-user", type=str, default="root")
    parser.add_argument("--db-password", type=str, default="root")
    parser.add_argument("--db-name", type=str, default="staging")
    args = parser.parse_args()

    s3 = get_s3_client(args.s3_endpoint, args.s3_access_key, args.s3_secret_key)

    # 1. Lecture de la zone Raw
    frames = []

    if args.source in ("dataset", "both"):
        print("Lecture du dataset depuis la zone Raw...")
        raw_dataset = read_dataset_from_raw(s3, args.bucket, args.dataset_prefix)
        if not raw_dataset.empty:
            frames.append(clean(raw_dataset, "dataset"))

    if args.source in ("api", "both"):
        print("Lecture de l'API depuis la zone Raw...")
        raw_api = read_api_from_raw(s3, args.bucket, args.api_prefix)
        if not raw_api.empty:
            frames.append(clean(raw_api, "api"))

    if not frames:
        print("\nERREUR : aucune donnee lue depuis la zone Raw.")
        return

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal a charger : {len(df)} lignes.")

    # 2. Chargement dans Staging
    print("Connexion a la zone Staging (MySQL)...")
    connection = create_mysql_connection(
        args.db_host, args.db_user, args.db_password, args.db_name
    )
    if connection is None:
        return

    create_table(connection)

    print("  Insertion (les doublons sont ecartes par la cle unique)...")
    inserted = insert_quotes(connection, df)

    # 3. Controle qualite
    print("\n--- Validation de la zone Staging ---")
    print(f"  Lignes inserees ce run : {inserted}")
    validate_data(connection)

    connection.close()
    print("\nZone Staging alimentee. On peut passer a la zone Curated.")


if __name__ == "__main__":
    main()