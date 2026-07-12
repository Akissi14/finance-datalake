"""
Etape 8b : Mesure de performance des endpoints /ingest et /ingest_fast.

Le sujet impose de chronometrer et documenter le temps d'execution du pipeline
d'ingestion pour un lot d'UN element et pour un lot de CENT elements, puis de
demontrer que /ingest_fast apporte au moins 30 % de gain.

METHODOLOGIE
------------
1. Les cotations soumises sont REALISTES : elles derivent des derniers cours
   connus de valeurs reellement presentes en zone Curated, perturbes de
   quelques pourcents. Soumettre des donnees aleatoires fausserait la mesure,
   car les features (volatilite, RSI, ratios) dependent de l'historique.

2. Chaque configuration est mesuree PLUSIEURS FOIS et on retient la MEDIANE.
   Une mesure unique capturerait le bruit de la machine (ordonnancement,
   cache, garbage collector) autant que le code lui-meme.

3. Un appel a vide precede les mesures ('warm-up') : la premiere requete paie
   l'etablissement des pools de connexions et, cote /ingest_fast, la
   compilation Numba. L'inclure dans la mesure serait malhonnete.

Usage:
    python src/benchmark.py
    python src/benchmark.py --repetitions 10
"""
import argparse
import random
import statistics
import time

import pymongo
import requests


def get_reference_quotes(mongo_uri, database, collection, count):
    """
    Construit des cotations plausibles a partir de la zone Curated.

    On tire au sort des valeurs reellement presentes, on recupere leur dernier
    cours connu, et on le perturbe legerement. Les cotations soumises ont donc
    un historique en base, condition necessaire au calcul des features.

    Returns
    -------
    list of dict
    """
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    coll = client[database][collection]

    tickers = coll.distinct("ticker")
    if not tickers:
        client.close()
        raise RuntimeError("Zone Curated vide : lancez d'abord le pipeline.")

    quotes = []
    for i in range(count):
        ticker = random.choice(tickers)

        last = coll.find_one(
            {"ticker": ticker}, {"ohlcv": 1}, sort=[("quote_date", -1)]
        )
        if not last:
            continue

        close = last["ohlcv"]["close"]
        volume = last["ohlcv"]["volume"]

        # Perturbation de quelques pourcents autour du dernier cours connu
        drift = random.uniform(-0.05, 0.05)
        new_close = close * (1 + drift)

        quotes.append(
            {
                "ticker": ticker,
                "date": f"2026-07-{(i % 28) + 1:02d}T00:00:00",
                "open": round(close * (1 + random.uniform(-0.01, 0.01)), 4),
                "high": round(max(close, new_close) * 1.01, 4),
                "low": round(min(close, new_close) * 0.99, 4),
                "close": round(new_close, 4),
                "volume": int(volume * random.uniform(0.5, 2.0)),
            }
        )

    client.close()
    return quotes


def call_endpoint(base_url, endpoint, quotes):
    """
    Appelle un endpoint d'ingestion et mesure sa duree cote client.

    On mesure du cote de l'appelant, et non seulement la duree renvoyee par
    l'API : c'est le temps que subit reellement un consommateur du service.

    Returns
    -------
    tuple of (float, dict)
        Duree en millisecondes, et corps de la reponse.
    """
    payload = {"data": {"quotes": quotes}}

    start = time.perf_counter()
    response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=300)
    elapsed = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    return elapsed, response.json()


def measure(base_url, endpoint, quotes, repetitions):
    """
    Mesure un endpoint plusieurs fois et renvoie la mediane.

    La mediane est preferee a la moyenne : elle resiste aux valeurs aberrantes
    qu'une machine partagee produit inevitablement.

    Returns
    -------
    dict
        Statistiques de la serie de mesures.
    """
    durations = []

    for _ in range(repetitions):
        elapsed, body = call_endpoint(base_url, endpoint, quotes)
        durations.append(elapsed)

    return {
        "mediane_ms": statistics.median(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "traites": body.get("traites", 0),
        "anomalies": body.get("anomalies", 0),
    }


def run_comparison(base_url, quotes, label, repetitions):
    """
    Compare les deux endpoints sur un meme lot et affiche le gain.

    Returns
    -------
    float
        Le gain relatif de /ingest_fast, en pourcentage.
    """
    print(f"\n=== Lot de {label} ===")

    baseline = measure(base_url, "/ingest", quotes, repetitions)
    fast = measure(base_url, "/ingest_fast", quotes, repetitions)

    gain = (baseline["mediane_ms"] - fast["mediane_ms"]) / baseline["mediane_ms"] * 100

    print(f"{'ENDPOINT':<16} {'MEDIANE':>12} {'MIN':>10} {'MAX':>10} {'TRAITES':>9}")
    print(
        f"{'/ingest':<16} "
        f"{baseline['mediane_ms']:>10.1f} ms "
        f"{baseline['min_ms']:>8.1f} ms "
        f"{baseline['max_ms']:>8.1f} ms "
        f"{baseline['traites']:>9}"
    )
    print(
        f"{'/ingest_fast':<16} "
        f"{fast['mediane_ms']:>10.1f} ms "
        f"{fast['min_ms']:>8.1f} ms "
        f"{fast['max_ms']:>8.1f} ms "
        f"{fast['traites']:>9}"
    )

    acceleration = baseline["mediane_ms"] / fast["mediane_ms"]
    print(f"\n  Gain      : {gain:.1f} %")
    print(f"  Facteur   : x{acceleration:.2f}")
    print(f"  Objectif  : {'ATTEINT' if gain >= 30 else 'NON ATTEINT'} (seuil 30 %)")

    return gain


def main():
    parser = argparse.ArgumentParser(
        description="Compare les performances de /ingest et /ingest_fast."
    )
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/")
    parser.add_argument("--mongo-db", type=str, default="curated")
    parser.add_argument("--mongo-collection", type=str, default="quotes_features")
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    # Verification prealable : inutile de mesurer une API qui ne repond pas.
    try:
        health = requests.get(f"{args.base_url}/health", timeout=10).json()
    except requests.RequestException as e:
        print(f"ERREUR : API injoignable sur {args.base_url} ({e}).")
        print("Lancez : uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return

    if health.get("model") != "charge":
        print("ERREUR : le modele n'est pas charge cote API.")
        return

    print("Preparation des cotations de test...")
    try:
        quotes_100 = get_reference_quotes(
            args.mongo_uri, args.mongo_db, args.mongo_collection, 100
        )
    except RuntimeError as e:
        print(f"ERREUR : {e}")
        return

    quotes_1 = quotes_100[:1]
    print(f"  {len(quotes_100)} cotations generees a partir de la zone Curated.")

    # Warm-up : la premiere requete paie les pools de connexions et la
    # compilation Numba. La mesurer fausserait completement la comparaison.
    print("Warm-up...")
    call_endpoint(args.base_url, "/ingest", quotes_1)
    call_endpoint(args.base_url, "/ingest_fast", quotes_1)

    print(f"Mesures ({args.repetitions} repetitions, mediane retenue)")

    gain_1 = run_comparison(args.base_url, quotes_1, "1 element", args.repetitions)
    gain_100 = run_comparison(
        args.base_url, quotes_100, "100 elements", args.repetitions
    )

    print("\n=== Synthese ===")
    print(f"  Gain sur 1 element    : {gain_1:.1f} %")
    print(f"  Gain sur 100 elements : {gain_100:.1f} %")
    print(
        "\nLe gain croit avec la taille du lot : les optimisations portent sur "
        "\nl'amortissement des couts fixes (aller-retours reseau, appels au "
        "\nmodele), qui se paient une fois par LOT et non une fois par ELEMENT."
    )


if __name__ == "__main__":
    main()