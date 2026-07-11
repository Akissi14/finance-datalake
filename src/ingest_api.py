"""
Etape 3 : Ingestion de la source API (Yahoo Finance) vers la zone Raw.

Seconde source de donnees du data lake, exigee par le sujet. Contrairement au
dataset Kaggle qui est fige (2013-2018), cette source apporte des cotations
recentes et sera re-ingeree periodiquement par Airflow.

Le payload est depose **tel quel** dans MinIO, au format JSON, sous le prefixe
'source_api/'. Chaque fichier porte l'horodatage de son ingestion : deux runs
successifs ne s'ecrasent donc pas, et on garde la trace de ce que l'API a
repondu a un instant t.

Les enregistrements suivent le meme schema que le dataset fichier
(date, open, high, low, close, volume, Name), ce qui permettra de reunir les
deux sources dans une table Staging unique.

Usage:
    python src/ingest_api.py \
        --tickers AAPL,MSFT,GOOGL,AMZN,JPM \
        --period 1mo \
        --interval 1d
"""
import argparse
import json
from datetime import datetime, timezone

import boto3
import yfinance as yf
from botocore.exceptions import BotoCoreError, ClientError


def get_s3_client(endpoint_url, access_key, secret_key):
    """
    Cree un client S3 pointant vers MinIO (zone Raw).

    Returns
    -------
    botocore.client.S3
    """
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def ensure_bucket(s3, bucket):
    """
    Cree le bucket s'il n'existe pas encore (operation idempotente).
    """
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket)
        print(f"  Bucket '{bucket}' cree.")


def fetch_ticker(ticker, period, interval):
    """
    Interroge l'API Yahoo Finance pour une valeur donnee.

    Le schema de sortie est aligne sur celui du dataset Kaggle, afin que les
    deux sources puissent alimenter la meme table Staging.

    Parameters
    ----------
    ticker : str
        Symbole boursier (ex. 'AAPL').
    period : str
        Profondeur d'historique ('1d', '5d', '1mo', '1y', 'max'...).
    interval : str
        Granularite ('1m', '1h', '1d'...).

    Returns
    -------
    list of dict
        Cotations au format {date, open, high, low, close, volume, Name}.
        Liste vide si l'API ne renvoie rien (ticker inconnu, marche ferme...).
    """
    try:
        history = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=False
        )
    except Exception as e:
        # L'API est un service externe : elle peut tomber, changer, limiter.
        # On isole l'echec sur un ticker sans faire echouer toute l'ingestion.
        print(f"  ERREUR API sur {ticker} : {e}")
        return []

    if history.empty:
        print(f"  AUCUNE DONNEE pour {ticker} (periode {period}).")
        return []

    records = []
    for index, row in history.iterrows():
        records.append(
            {
                "date": index.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "Name": ticker,
            }
        )

    print(f"  {ticker} : {len(records)} cotations recuperees.")
    return records


def build_payload(records, tickers, period, interval):
    """
    Enveloppe les cotations dans un payload horodate.

    Les metadonnees d'ingestion (quand, quoi, comment) accompagnent la donnee :
    sans elles, un objet de la zone Raw devient rapidement intracable.

    Returns
    -------
    dict
    """
    return {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source": "yahoo_finance_api",
        "params": {
            "tickers": tickers,
            "period": period,
            "interval": interval,
        },
        "record_count": len(records),
        "records": records,
    }


def upload_payload(s3, bucket, prefix, payload):
    """
    Depose le payload JSON dans la zone Raw.

    La cle contient l'horodatage : chaque execution cree un nouvel objet plutot
    que d'ecraser le precedent. C'est indispensable pour une source qu'Airflow
    va re-interroger toutes les heures.

    Returns
    -------
    str
        La cle de l'objet cree.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"{prefix}/yfinance_{stamp}.json"

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return key


def main():
    parser = argparse.ArgumentParser(
        description="Ingere les cotations de l'API Yahoo Finance dans la zone Raw."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,GOOGL,AMZN,JPM,XOM,JNJ,PG,KO,DIS",
        help="Symboles separes par des virgules.",
    )
    parser.add_argument("--period", type=str, default="1mo")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--bucket", type=str, default="raw")
    parser.add_argument("--prefix", type=str, default="source_api")
    parser.add_argument("--s3-endpoint", type=str, default="http://localhost:9000")
    parser.add_argument("--s3-access-key", type=str, default="minioadmin")
    parser.add_argument("--s3-secret-key", type=str, default="minioadmin")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if not tickers:
        print("ERREUR : aucun ticker fourni.")
        return

    # 1. Interroger l'API, ticker par ticker
    print(f"Interrogation de l'API Yahoo Finance ({len(tickers)} tickers)...")
    all_records = []
    for ticker in tickers:
        all_records.extend(fetch_ticker(ticker, args.period, args.interval))

    if not all_records:
        print("\nERREUR : l'API n'a renvoye aucune donnee. Rien n'est ingere.")
        return

    # 2. Envelopper dans un payload horodate
    payload = build_payload(all_records, tickers, args.period, args.interval)

    # 3. Deposer dans la zone Raw
    print("Connexion a la zone Raw (MinIO)...")
    s3 = get_s3_client(args.s3_endpoint, args.s3_access_key, args.s3_secret_key)

    try:
        ensure_bucket(s3, args.bucket)
        key = upload_payload(s3, args.bucket, args.prefix, payload)
    except (BotoCoreError, ClientError) as e:
        print(f"ERREUR : depot dans la zone Raw impossible ({e}).")
        return

    print("\n--- Resume de l'ingestion API ---")
    print(f"Tickers interroges : {len(tickers)}")
    print(f"Cotations ingerees : {len(all_records)}")
    print(f"Objet cree         : s3://{args.bucket}/{key}")
    print("\nZone Raw alimentee par les deux sources.")


if __name__ == "__main__":
    main()