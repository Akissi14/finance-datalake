"""
Etape 6b : Scoring des anomalies sur la zone Curated.

Ce script est le pendant 'inference' de train_model.py. Il recharge l'artefact
depuis MinIO et score l'INTEGRALITE des documents de la zone Curated, la ou
l'entrainement ne portait que sur un echantillon.

Chaque document recoit un bloc 'anomaly' :

    {"score": 4.83, "is_anomaly": true, "threshold": 2.46, "scored_at": "..."}

L'ajout se fait par $set sur un document existant : aucune migration de schema
n'est necessaire. C'est precisement ce que permet une base documentaire, la ou
un SGBD relationnel imposerait un ALTER TABLE ou une table jointe.

Le traitement se fait par lots : charger 600 000 documents en memoire ferait
tomber le processus. On lit, on score, on met a jour, on libere.

Usage:
    python src/apply_model.py
    python src/apply_model.py --tickers AAPL,MSFT
"""
import argparse
import io
import pickle
from datetime import datetime, timezone

import boto3
import numpy as np
import pymongo
from botocore.exceptions import BotoCoreError, ClientError
from pymongo import UpdateOne
from pymongo.errors import PyMongoError


def load_model_from_raw(s3, bucket, key):
    """
    Recharge l'artefact de modele depuis la zone Raw.

    L'artefact contient le reseau, son scaler, son seuil et la liste ordonnee
    de ses features. Les trois sont indispensables : un modele sans son scaler
    scorerait des donnees a la mauvaise echelle et produirait n'importe quoi.

    Returns
    -------
    dict
    """
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pickle.load(io.BytesIO(body))


def build_matrix(documents):
    """
    Derive les features du modele a partir d'un lot de documents.

    La logique est rigoureusement celle de train_model.py : le meme calcul doit
    etre applique a l'entrainement et a l'inference, sinon le modele score des
    donnees qu'il n'a jamais apprises. C'est le piege classique du 'training /
    serving skew'.

    Returns
    -------
    tuple of (list, numpy.ndarray)
        Les identifiants MongoDB retenus, et la matrice correspondante.
        Les deux sont alignes : la i-eme ligne correspond au i-eme identifiant.
    """
    ids = []
    rows = []

    for doc in documents:
        f = doc["features"]
        close = doc["ohlcv"]["close"]
        ma_5 = f["ma_5"]
        ma_20 = f["ma_20"]

        if ma_20 == 0 or ma_5 == 0:
            continue

        ids.append(doc["_id"])
        rows.append(
            [
                f["daily_return"],
                f["volatility_20"],
                f["rsi_14"] / 100.0,
                f["volume_ratio"],
                f["hl_range"],
                f["gap"],
                ma_5 / ma_20,
                close / ma_20,
            ]
        )

    return ids, np.array(rows, dtype=np.float64)


def score_batch(artifact, X):
    """
    Calcule le score d'anomalie d'un lot d'observations.

    Le score est l'erreur quadratique moyenne entre l'entree et sa
    reconstruction par l'autoencodeur.

    Returns
    -------
    numpy.ndarray
    """
    X_scaled = artifact["scaler"].transform(X)
    X_reconstructed = artifact["model"].predict(X_scaled)

    return np.mean((X_scaled - X_reconstructed) ** 2, axis=1)


def apply_scores(coll, artifact, query, batch_size=20000):
    """
    Score les documents de la zone Curated et y ecrit le resultat.

    Le curseur est parcouru par lots afin de garder une empreinte memoire
    constante. Les mises a jour partent en bulk_write : 20 000 UpdateOne en un
    aller-retour reseau, au lieu de 20 000 aller-retours.

    Returns
    -------
    tuple of (int, int)
        Nombre de documents scores, nombre d'anomalies detectees.
    """
    threshold = artifact["threshold"]
    scored_at = datetime.now(timezone.utc).isoformat()

    cursor = coll.find(query, {"features": 1, "ohlcv.close": 1})

    total_scored = 0
    total_anomalies = 0
    buffer = []

    def flush(docs):
        """Score le tampon courant et ecrit les resultats."""
        ids, X = build_matrix(docs)

        if not ids:
            return 0, 0

        scores = score_batch(artifact, X)

        operations = []
        anomalies = 0

        for doc_id, score in zip(ids, scores):
            is_anomaly = bool(score > threshold)
            anomalies += int(is_anomaly)

            operations.append(
                UpdateOne(
                    {"_id": doc_id},
                    {
                        "$set": {
                            "anomaly": {
                                "score": float(score),
                                "is_anomaly": is_anomaly,
                                "threshold": float(threshold),
                                "scored_at": scored_at,
                            }
                        }
                    },
                )
            )

        coll.bulk_write(operations, ordered=False)
        return len(ids), anomalies

    for doc in cursor:
        buffer.append(doc)

        if len(buffer) >= batch_size:
            scored, anomalies = flush(buffer)
            total_scored += scored
            total_anomalies += anomalies
            buffer = []
            print(f"    {total_scored} documents scores...")

    if buffer:
        scored, anomalies = flush(buffer)
        total_scored += scored
        total_anomalies += anomalies

    return total_scored, total_anomalies


def report_top_anomalies(coll, limit=10):
    """
    Affiche les journees les plus anormales detectees.

    C'est le controle de bon sens final : si les anomalies remontees
    correspondent a des journees de marche connues pour leur turbulence, le
    modele fait son travail. Si elles semblent aleatoires, il y a un probleme.
    """
    top = coll.find({"anomaly.is_anomaly": True}).sort("anomaly.score", -1).limit(limit)

    print(f"\n  Les {limit} journees les plus anormales :")
    print(f"    {'TICKER':<8} {'DATE':<12} {'SCORE':>10} {'RENDEMENT':>11} {'VOL/MOY':>9}")

    for doc in top:
        f = doc["features"]
        print(
            f"    {doc['ticker']:<8} "
            f"{doc['quote_date'].strftime('%Y-%m-%d'):<12} "
            f"{doc['anomaly']['score']:>10.2f} "
            f"{f['daily_return']:>10.2%} "
            f"{f['volume_ratio']:>9.1f}"
        )


def report_by_ticker(coll, limit=10):
    """
    Classe les valeurs par nombre d'anomalies detectees.
    """
    pipeline = [
        {"$match": {"anomaly.is_anomaly": True}},
        {"$group": {"_id": "$ticker", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
    ]

    print(f"\n  Valeurs les plus souvent anormales :")
    for row in coll.aggregate(pipeline):
        print(f"    {row['_id']:<8} {row['count']} journees")


def main():
    parser = argparse.ArgumentParser(
        description="Score les anomalies sur la zone Curated."
    )
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/")
    parser.add_argument("--mongo-db", type=str, default="curated")
    parser.add_argument("--mongo-collection", type=str, default="quotes_features")
    parser.add_argument("--bucket", type=str, default="raw")
    parser.add_argument("--model-key", type=str, default="models/autoencoder.pkl")
    parser.add_argument("--s3-endpoint", type=str, default="http://localhost:9000")
    parser.add_argument("--s3-access-key", type=str, default="minioadmin")
    parser.add_argument("--s3-secret-key", type=str, default="minioadmin")
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=20000)
    args = parser.parse_args()

    # 1. Recharger le modele
    print("Chargement du modele depuis la zone Raw...")
    s3 = boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint,
        aws_access_key_id=args.s3_access_key,
        aws_secret_access_key=args.s3_secret_key,
        region_name="us-east-1",
    )

    try:
        artifact = load_model_from_raw(s3, args.bucket, args.model_key)
    except (BotoCoreError, ClientError) as e:
        print(f"ERREUR : modele introuvable ({e}).")
        print("Lancez d'abord train_model.py.")
        return

    print(f"  Modele entraine le : {artifact['trained_at']}")
    print(f"  Seuil d'anomalie   : {artifact['threshold']:.5f}")
    print(f"  Contamination      : {artifact['contamination']:.1%}")

    # 2. Scorer la zone Curated
    query = {}
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        query = {"ticker": {"$in": tickers}}
        print(f"  Restreint aux valeurs : {', '.join(tickers)}")

    try:
        client = pymongo.MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        coll = client[args.mongo_db][args.mongo_collection]

        to_score = coll.count_documents(query)
        if to_score == 0:
            print("ERREUR : aucun document a scorer.")
            client.close()
            return

        print(f"\nScoring de {to_score} documents...")
        scored, anomalies = apply_scores(coll, artifact, query,  args.batch_size)

    except PyMongoError as e:
        print(f"ERREUR : zone Curated injoignable ({e}).")
        return

    # 3. Rapport
    print("\n--- Resultats de la detection ---")
    print(f"  Documents scores   : {scored}")
    print(f"  Anomalies detectees: {anomalies} ({anomalies / scored:.2%})")

    report_top_anomalies(coll)
    report_by_ticker(coll)

    client.close()
    print("\nZone Curated enrichie des scores d'anomalie.")


if __name__ == "__main__":
    main()