"""
Etape 8 : API Gateway du data lake.

Expose une interface HTTP unique au-dessus des trois zones, afin de recuperer
les donnees ingerees sans avoir a manipuler directement MinIO, MySQL ou MongoDB.

    GET  /health        etat des trois zones
    GET  /stats         volumetrie de remplissage
    GET  /raw           inventaire des objets de la zone Raw
    GET  /staging       cotations de la zone Staging
    GET  /curated       documents enrichis et scores de la zone Curated
    POST /ingest        ingestion a chaud (implementation de reference)
    POST /ingest_fast   ingestion a chaud (implementation optimisee)

NIVEAU AVANCE : /ingest et /ingest_fast
---------------------------------------
Les deux endpoints font RIGOUREUSEMENT le meme travail :

    1. valider le lot de cotations recu ;
    2. archiver le payload brut dans la zone Raw ;
    3. calculer les 8 features du modele, ce qui suppose de recuperer
       l'historique recent de chaque valeur ;
    4. scorer avec l'autoencodeur ;
    5. ecrire les documents enrichis dans la zone Curated.

Seule la MANIERE differe :

                     /ingest                      /ingest_fast
    archivage    bloquant, avant tout         concurrent (ThreadPoolExecutor)
    historique   1 requete Mongo par cotation  memes requetes, en parallele
    scoring      1 predict() par ligne         1 predict() sur la matrice
    erreur       boucle Python interpretee     Numba (@njit, compile)
    ecriture     1 update_one par document     1 bulk_write

Voir le docstring de _fetch_histories pour une optimisation tentee, MESUREE
COMME CONTRE-PRODUCTIVE, puis abandonnee. C'est un resultat en soi.

Lancement :
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import io
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional

import boto3
import mysql.connector
import numpy as np
import pymongo
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException, Query
from numba import njit, prange
from pydantic import BaseModel, Field
from pymongo import UpdateOne

# ---------------------------------------------------------------------------
# Configuration (surchargeable par variables d'environnement en conteneur)
# ---------------------------------------------------------------------------

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("BUCKET", "raw")
MODEL_KEY = os.getenv("MODEL_KEY", "models/autoencoder.pkl")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_NAME = os.getenv("DB_NAME", "staging")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "curated")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "quotes_features")

HISTORY_WINDOW = 20  # Profondeur necessaire au calcul des moyennes mobiles

app = FastAPI(
    title="Finance Data Lake - API Gateway",
    description="Interface HTTP au-dessus des zones Raw, Staging et Curated.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Clients (crees une fois, reutilises : ouvrir une connexion par requete
# couterait plus cher que la requete elle-meme)
# ---------------------------------------------------------------------------


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="us-east-1",
    )


def get_mysql():
    return mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )


mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
curated = mongo_client[MONGO_DB][MONGO_COLLECTION]

# Pool dedie a l'archivage dans la zone Raw. Cette ecriture est un aller-retour
# reseau independant du calcul qui suit : la lancer en arriere-plan permet de
# recouvrir les deux travaux au lieu de les additionner.
ARCHIVE_POOL = ThreadPoolExecutor(max_workers=4)

# Le modele est charge une seule fois, au demarrage. Le recharger a chaque
# requete d'ingestion ajouterait plusieurs centaines de millisecondes.
MODEL = None


@app.on_event("startup")
def load_model():
    """
    Charge l'autoencodeur au demarrage, et prechauffe la fonction Numba.

    Numba compile a la premiere invocation. Sans prechauffage, cette
    compilation serait imputee a la premiere requete /ingest_fast et fausserait
    completement la mesure de performance.
    """
    global MODEL

    try:
        body = get_s3().get_object(Bucket=BUCKET, Key=MODEL_KEY)["Body"].read()
        MODEL = pickle.load(io.BytesIO(body))
        print(f"Modele charge (seuil = {MODEL['threshold']:.5f}).")
    except Exception as e:
        print(f"AVERTISSEMENT : modele indisponible ({e}). /ingest sera inactif.")

    # Prechauffage de la compilation Numba
    dummy = np.zeros((2, 8), dtype=np.float64)
    _mse_numba(dummy, dummy)
    print("Fonction Numba compilee.")


# ---------------------------------------------------------------------------
# Schemas d'entree
# ---------------------------------------------------------------------------


class Quote(BaseModel):
    """Une cotation soumise a l'ingestion."""

    ticker: str = Field(..., min_length=1, max_length=10)
    date: str
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)


class QuoteBatch(BaseModel):
    """Le lot de cotations."""

    quotes: List[Quote]


class IngestPayload(BaseModel):
    """
    Enveloppe attendue, calquee sur l'exemple du sujet :

        {"data": {"quotes": [ {...}, {...} ]}}
    """

    data: QuoteBatch


# ---------------------------------------------------------------------------
# Noyau de calcul
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _mse_numba(X, X_reconstructed):
    """
    Erreur quadratique moyenne, ligne par ligne, compilee par Numba.

    La boucle est ecrite explicitement, mais Numba la compile en code machine
    et la parallelise sur les coeurs disponibles (prange). C'est exactement ce
    que la version de reference fait en Python pur, interprete, ligne par ligne.
    """
    n, m = X.shape
    out = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        total = 0.0
        for j in range(m):
            diff = X[i, j] - X_reconstructed[i, j]
            total += diff * diff
        out[i] = total / m

    return out


def _features_from_history(quote, closes, volumes):
    """
    Calcule les 8 features du modele pour une cotation, en Python pur.

    'closes' et 'volumes' sont les 20 derniers points connus de la valeur, du
    plus ancien au plus recent. Sans eux, aucune moyenne mobile ni volatilite
    n'est calculable : une cotation isolee n'a pas de contexte.

    Returns
    -------
    list of float or None
        None si l'historique est insuffisant.
    """
    if len(closes) < HISTORY_WINDOW:
        return None

    prev_close = closes[-1]
    if prev_close <= 0:
        return None

    series = closes + [quote["close"]]

    returns = []
    for i in range(1, len(series)):
        if series[i - 1] > 0:
            returns.append((series[i] - series[i - 1]) / series[i - 1])

    if len(returns) < HISTORY_WINDOW:
        return None

    window_returns = returns[-HISTORY_WINDOW:]

    mean_r = sum(window_returns) / len(window_returns)
    variance = sum((r - mean_r) ** 2 for r in window_returns) / (
        len(window_returns) - 1
    )
    volatility = variance**0.5

    ma_5 = sum(series[-5:]) / 5
    ma_20 = sum(series[-HISTORY_WINDOW:]) / HISTORY_WINDOW

    if ma_5 == 0 or ma_20 == 0:
        return None

    # RSI sur 14 periodes
    gains = [r for r in returns[-14:] if r > 0]
    losses = [-r for r in returns[-14:] if r < 0]
    avg_gain = sum(gains) / 14 if gains else 0.0
    avg_loss = sum(losses) / 14 if losses else 0.0

    if avg_loss == 0:
        rsi = 100.0
    else:
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

    avg_volume = sum(volumes[-HISTORY_WINDOW:]) / HISTORY_WINDOW
    volume_ratio = quote["volume"] / avg_volume if avg_volume > 0 else 0.0

    daily_return = (quote["close"] - prev_close) / prev_close
    hl_range = (quote["high"] - quote["low"]) / quote["close"]
    gap = (quote["open"] - prev_close) / prev_close

    return [
        daily_return,
        volatility,
        rsi / 100.0,
        volume_ratio,
        hl_range,
        gap,
        ma_5 / ma_20,
        quote["close"] / ma_20,
    ]


def _fetch_history(ticker):
    """
    Recupere les 20 derniers points connus d'une valeur (une requete Mongo).

    L'index (ticker, quote_date) resout ce find().sort().limit(20) sans
    parcourir la collection : seuls 20 documents sont reellement lus.
    """
    docs = list(
        curated.find(
            {"ticker": ticker},
            {"ohlcv.close": 1, "ohlcv.volume": 1, "quote_date": 1},
        )
        .sort("quote_date", -1)
        .limit(HISTORY_WINDOW)
    )
    docs.reverse()  # du plus ancien au plus recent

    closes = [d["ohlcv"]["close"] for d in docs]
    volumes = [float(d["ohlcv"]["volume"]) for d in docs]

    return closes, volumes


def _fetch_histories(tickers, max_workers=16):
    """
    Recupere l'historique de plusieurs valeurs EN PARALLELE.

    PREMIERE TENTATIVE, MESUREE PUIS ABANDONNEE
    -------------------------------------------
    L'intuition de depart etait de remplacer les N requetes par UNE agregation
    ($match + $sort + $group) couvrant tous les tickers du lot. Mesure : 4,7
    fois PLUS LENTE que la version de reference (849 ms contre 179 ms sur 100
    elements).

    La raison : pour 100 tickers d'environ 1 200 cotations chacun, MongoDB
    devait trier 120 000 documents et empiler 120 000 valeurs en memoire, pour
    n'en conserver ensuite que 20 par ticker. On jetait 99 % du travail. La
    version de reference, elle, emet des find().sort().limit(20) que l'index
    (ticker, quote_date) resout instantanement : elle ne lit jamais que 20
    documents par appel.

    ENSEIGNEMENT : 'une requete au lieu de cent' n'est pas une optimisation en
    soi. Ce qui compte est le VOLUME REELLEMENT TRAITE, pas le nombre d'appels.

    VERSION RETENUE
    ---------------
    On conserve les memes requetes indexees que la baseline, mais on les execute
    en parallele. Elles sont I/O-bound : elles passent leur temps a attendre le
    reseau. Or le GIL est relache pendant les entrees-sorties, donc des threads
    Python les parallelisent efficacement. Le pool de connexions de pymongo est
    concu pour cet usage concurrent.

    Returns
    -------
    dict
        {ticker: (closes, volumes)}
    """
    tickers = list(tickers)

    if not tickers:
        return {}

    # Un seul ticker : creer un pool de threads pour paralleliser UNE requete
    # couterait plus cher que la requete elle-meme (creation, synchronisation
    # et destruction des threads). Le surcout de coordination du parallelisme
    # est fixe : il ne s'amortit qu'au-dela d'un certain volume. C'est la loi
    # d'Amdahl appliquee a un cas concret, et c'est pourquoi une optimisation
    # doit toujours etre MESUREE, jamais supposee.
    if len(tickers) == 1:
        return {tickers[0]: _fetch_history(tickers[0])}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tickers))) as pool:
        results = list(pool.map(_fetch_history, tickers))

    return dict(zip(tickers, results))


def _archive_payload(payload, suffix):
    """
    Archive le lot recu dans la zone Raw.

    Meme regle que pour les pipelines batch : ce qui entre dans le lac est
    conserve tel quel. Si la logique de calcul aval change, on peut rejouer.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    key = f"source_api/ingest_{suffix}_{stamp}.json"

    get_s3().put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )

    return key

def _build_document(quote, features, score, threshold, endpoint):
    """
    Construit le document Curated d'une cotation ingeree a chaud.

    Le pipeline batch ecrit ma_5 et ma_20 ; l'API doit ecrire le MEME schema,
    sans quoi train_model.py, qui echantillonne au hasard dans la collection,
    tombe sur un document incompatible (KeyError: 'ma_20'). Une base sans schema
    n'exempte pas d'un contrat : elle en deplace seulement la responsabilite du
    moteur vers l'application.

    Les moyennes mobiles sont reconstituees a partir des features, qui sont des
    ratios sans echelle : features[7] = close / ma_20 et features[6] = ma_5 / ma_20.
    """
    ma_20 = quote["close"] / features[7] if features[7] else 0.0
    ma_5 = ma_20 * features[6]

    return {
        "ticker": quote["ticker"],
        "quote_date": datetime.fromisoformat(quote["date"]),
        "source": "ingest",
        "ohlcv": {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        },
        "features": {
            "daily_return": features[0],
            "volatility_20": features[1],
            "rsi_14": features[2] * 100,
            "volume_ratio": features[3],
            "hl_range": features[4],
            "gap": features[5],
            "ma_5": ma_5,
            "ma_20": ma_20,
        },
        "anomaly": {
            "score": float(score),
            "is_anomaly": bool(score > threshold),
            "threshold": float(threshold),
            "scored_at": datetime.now(timezone.utc).isoformat(),
        },
        "metadata": {"origin": endpoint},
    }

# ---------------------------------------------------------------------------
# Endpoints de lecture
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Etat des trois zones du lac."""
    status = {}

    try:
        get_s3().list_buckets()
        status["raw_minio"] = "ok"
    except (BotoCoreError, ClientError) as e:
        status["raw_minio"] = f"erreur: {e}"

    try:
        conn = get_mysql()
        conn.close()
        status["staging_mysql"] = "ok"
    except mysql.connector.Error as e:
        status["staging_mysql"] = f"erreur: {e}"

    try:
        mongo_client.server_info()
        status["curated_mongodb"] = "ok"
    except Exception as e:
        status["curated_mongodb"] = f"erreur: {e}"

    status["model"] = "charge" if MODEL else "indisponible"

    zones = [v for k, v in status.items() if k.startswith(("raw", "staging", "curated"))]
    status["global"] = "ok" if all(v == "ok" for v in zones) else "degrade"

    return status


@app.get("/stats")
def stats():
    """Volumetrie de remplissage des trois zones."""
    result = {}

    # Zone Raw : nombre d'objets par prefixe
    try:
        s3 = get_s3()
        paginator = s3.get_paginator("list_objects_v2")
        counts = {}
        total_size = 0

        for page in paginator.paginate(Bucket=BUCKET):
            for obj in page.get("Contents", []):
                prefix = obj["Key"].split("/")[0]
                counts[prefix] = counts.get(prefix, 0) + 1
                total_size += obj["Size"]

        result["raw"] = {
            "objets_par_prefixe": counts,
            "objets_total": sum(counts.values()),
            "taille_octets": total_size,
        }
    except (BotoCoreError, ClientError) as e:
        result["raw"] = {"erreur": str(e)}

    # Zone Staging
    try:
        conn = get_mysql()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM quotes")
        rows, tickers = cursor.fetchone()
        cursor.execute("SELECT source, COUNT(*) FROM quotes GROUP BY source")
        by_source = {s: c for s, c in cursor.fetchall()}
        cursor.close()
        conn.close()

        result["staging"] = {
            "lignes": rows,
            "tickers": tickers,
            "lignes_par_source": by_source,
        }
    except mysql.connector.Error as e:
        result["staging"] = {"erreur": str(e)}

    # Zone Curated
    try:
        total = curated.count_documents({})
        anomalies = curated.count_documents({"anomaly.is_anomaly": True})

        result["curated"] = {
            "documents": total,
            "tickers": len(curated.distinct("ticker")),
            "anomalies": anomalies,
            "taux_anomalies": round(anomalies / total, 4) if total else 0,
        }
    except Exception as e:
        result["curated"] = {"erreur": str(e)}

    return result


@app.get("/raw")
def read_raw(
    prefix: Optional[str] = Query(None, description="Ex. source_api/"),
    limit: int = Query(50, ge=1, le=500),
):
    """Inventaire des objets de la zone Raw."""
    try:
        s3 = get_s3()
        params = {"Bucket": BUCKET, "MaxKeys": limit}
        if prefix:
            params["Prefix"] = prefix

        response = s3.list_objects_v2(**params)

        objets = [
            {
                "key": obj["Key"],
                "taille_octets": obj["Size"],
                "modifie_le": obj["LastModified"].isoformat(),
            }
            for obj in response.get("Contents", [])
        ]

        return {"prefixe": prefix, "nombre": len(objets), "objets": objets}

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=503, detail=f"Zone Raw injoignable : {e}")


@app.get("/staging")
def read_staging(
    ticker: Optional[str] = Query(None),
    source: Optional[str] = Query(None, pattern="^(dataset|api)$"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Cotations de la zone Staging.

    Les filtres sont passes en requete parametree (%s), jamais concatenes :
    une concatenation ouvrirait la porte a l'injection SQL.
    """
    query = (
        "SELECT ticker, quote_date, open, high, low, close, volume, source FROM quotes"
    )
    conditions = []
    params = []

    if ticker:
        conditions.append("ticker = %s")
        params.append(ticker.upper())
    if source:
        conditions.append("source = %s")
        params.append(source)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY quote_date DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    try:
        conn = get_mysql()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        for row in rows:
            row["quote_date"] = row["quote_date"].isoformat()
            for col in ["open", "high", "low", "close"]:
                row[col] = float(row[col])

        return {"nombre": len(rows), "offset": offset, "cotations": rows}

    except mysql.connector.Error as e:
        raise HTTPException(status_code=503, detail=f"Zone Staging injoignable : {e}")


@app.get("/curated")
def read_curated(
    ticker: Optional[str] = Query(None),
    anomalies_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
):
    """Documents enrichis et scores de la zone Curated."""
    query = {}

    if ticker:
        query["ticker"] = ticker.upper()
    if anomalies_only:
        query["anomaly.is_anomaly"] = True

    try:
        sort_key = "anomaly.score" if anomalies_only else "quote_date"
        docs = list(curated.find(query, {"_id": 0}).sort(sort_key, -1).limit(limit))

        for doc in docs:
            doc["quote_date"] = doc["quote_date"].isoformat()

        return {"nombre": len(docs), "documents": docs}

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Zone Curated injoignable : {e}")


# ---------------------------------------------------------------------------
# Endpoints d'ingestion (niveau avance)
# ---------------------------------------------------------------------------


@app.post("/ingest")
def ingest(payload: IngestPayload):
    """
    Ingestion a chaud : IMPLEMENTATION DE REFERENCE.

    Volontairement directe, mais correcte. Elle traite les cotations une par
    une : archivage bloquant, puis pour chaque cotation une requete Mongo pour
    l'historique, un calcul de features, un appel a predict(), une ecriture.
    C'est la facon la plus naturelle d'ecrire ce traitement, et c'est la base
    de comparaison de /ingest_fast.
    """
    start = time.perf_counter()

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modele indisponible.")

    quotes = [q.model_dump() for q in payload.data.quotes]

    if not quotes:
        raise HTTPException(status_code=400, detail="Lot vide.")

    # Archivage BLOQUANT : on attend qu'il soit termine avant de calculer.
    _archive_payload({"quotes": quotes}, "baseline")

    threshold = MODEL["threshold"]
    scaler = MODEL["scaler"]
    model = MODEL["model"]

    traites = 0
    ignores = 0
    anomalies = 0

    for quote in quotes:
        # 1 requete Mongo PAR COTATION
        closes, volumes = _fetch_history(quote["ticker"])

        features = _features_from_history(quote, closes, volumes)
        if features is None:
            ignores += 1
            continue

        # 1 predict() PAR LIGNE
        X = scaler.transform(np.array([features], dtype=np.float64))
        X_rec = model.predict(X)

        # Erreur calculee en Python pur, interprete
        score = 0.0
        for j in range(X.shape[1]):
            diff = X[0][j] - X_rec[0][j]
            score += diff * diff
        score /= X.shape[1]

        doc = _build_document(quote, features, score, threshold, "ingest")

        # 1 ecriture PAR DOCUMENT
        curated.update_one(
            {
                "ticker": doc["ticker"],
                "quote_date": doc["quote_date"],
                "source": "ingest",
            },
            {"$set": doc},
            upsert=True,
        )

        traites += 1
        anomalies += int(doc["anomaly"]["is_anomaly"])

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "endpoint": "/ingest",
        "recus": len(quotes),
        "traites": traites,
        "ignores_historique_insuffisant": ignores,
        "anomalies": anomalies,
        "duree_ms": round(elapsed, 2),
    }


@app.post("/ingest_fast")
def ingest_fast(payload: IngestPayload):
    """
    Ingestion a chaud : IMPLEMENTATION OPTIMISEE.

    Le resultat est rigoureusement identique a /ingest. Cinq optimisations :

      0. ARCHIVAGE CONCURRENT. L'ecriture du payload dans MinIO est un
         aller-retour reseau d'une quinzaine de millisecondes, totalement
         independant du calcul qui suit. On la lance dans un thread et on ne
         l'attend qu'a la toute fin : les deux travaux se recouvrent au lieu de
         s'additionner. Decisif sur les petits lots, ou les couts fixes dominent.

      1. HISTORIQUES EN PARALLELE. Memes requetes indexees que la baseline, mais
         emises concurremment (voir _fetch_histories, qui documente aussi une
         optimisation tentee puis abandonnee apres mesure).

      2. UN SEUL predict(). L'algebre lineaire de scikit-learn est vectorisee
         via BLAS : traiter 100 lignes d'un coup coute a peine plus que d'en
         traiter une seule.

      3. ERREUR COMPILEE. La reconstruction est evaluee par une fonction Numba
         compilee en code machine et parallelisee, au lieu d'une boucle Python.

      4. UNE SEULE ECRITURE. Un bulk_write groupe toutes les mises a jour, au
         lieu d'un update_one par document.
    """
    start = time.perf_counter()

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modele indisponible.")

    quotes = [q.model_dump() for q in payload.data.quotes]

    if not quotes:
        raise HTTPException(status_code=400, detail="Lot vide.")

    # OPTIMISATION 0 : l'archivage part en arriere-plan, on ne l'attend pas ici.
    archive_future = ARCHIVE_POOL.submit(_archive_payload, {"quotes": quotes}, "fast")

    threshold = MODEL["threshold"]
    scaler = MODEL["scaler"]
    model = MODEL["model"]

    # OPTIMISATION 1 : les historiques sont recuperes en parallele.
    tickers = {q["ticker"] for q in quotes}
    histories = _fetch_histories(tickers)

    rows = []
    kept = []
    ignores = 0

    for quote in quotes:
        closes, volumes = histories.get(quote["ticker"], ([], []))
        features = _features_from_history(quote, closes, volumes)

        if features is None:
            ignores += 1
            continue

        rows.append(features)
        kept.append((quote, features))

    if not rows:
        # Meme dans ce cas, on s'assure que l'archivage a abouti.
        archive_future.result(timeout=30)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "endpoint": "/ingest_fast",
            "recus": len(quotes),
            "traites": 0,
            "ignores_historique_insuffisant": ignores,
            "anomalies": 0,
            "duree_ms": round(elapsed, 2),
        }

    X_raw = np.array(rows, dtype=np.float64)

    # OPTIMISATION 2 : un seul predict() sur la matrice complete.
    X = scaler.transform(X_raw)
    X_rec = model.predict(X)

    # OPTIMISATION 3 : erreur de reconstruction compilee par Numba.
    scores = _mse_numba(np.ascontiguousarray(X), np.ascontiguousarray(X_rec))

    # OPTIMISATION 4 : une seule ecriture groupee.
    operations = []
    anomalies = 0

    for (quote, features), score in zip(kept, scores):
        doc = _build_document(quote, features, score, threshold, "ingest_fast")
        anomalies += int(doc["anomaly"]["is_anomaly"])

        operations.append(
            UpdateOne(
                {
                    "ticker": doc["ticker"],
                    "quote_date": doc["quote_date"],
                    "source": "ingest",
                },
                {"$set": doc},
                upsert=True,
            )
        )

    curated.bulk_write(operations, ordered=False)

    # Resynchronisation : le travail d'archivage est concurrent, PAS abandonne.
    # Si MinIO avait echoue, l'exception remonterait ici. On a simplement cesse
    # d'attendre inutilement pendant que le calcul se deroulait.
    archive_future.result(timeout=30)

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "endpoint": "/ingest_fast",
        "recus": len(quotes),
        "traites": len(kept),
        "ignores_historique_insuffisant": ignores,
        "anomalies": anomalies,
        "duree_ms": round(elapsed, 2),
    }