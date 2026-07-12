"""
Etape 5 : Transformation Staging (MySQL) -> Curated (MongoDB).

La zone Staging contient des cotations propres mais brutes de sens : un prix de
cloture isole ne dit rien. La zone Curated produit de la donnee **prete a
l'usage**, ici enrichie d'indicateurs techniques qui serviront de features au
modele de detection d'anomalies.

Indicateurs calcules par valeur :

    daily_return   variation quotidienne du cours de cloture
    ma_5, ma_20    moyennes mobiles court et moyen terme
    volatility_20  ecart-type glissant des rendements (coeur de la detection)
    rsi_14         Relative Strength Index : momentum sur-achat / sur-vente
    volume_ratio   volume du jour rapporte a sa moyenne mobile
    hl_range       amplitude intra-journaliere, normalisee par le prix
    gap            ecart entre l'ouverture et la cloture de la veille

Point de conception important : les indicateurs sont calculés par couple
(ticker, source) et non par ticker seul. Le dataset s'arrete en 2018 et l'API
reprend en 2026 : une moyenne mobile calculee a cheval sur ces deux periodes
enjamberait un trou de huit ans et produirait des valeurs denuees de sens.

Le choix de MongoDB pour cette zone tient a la structure du document :
{ohlcv, features, metadata} accueillera un bloc {anomaly} a l'etape suivante
sans aucune migration de schema, la ou SQL imposerait un ALTER TABLE.

Usage:
    python src/staging_to_curated.py
    python src/staging_to_curated.py --tickers AAPL,MSFT   # test rapide
"""
import argparse
from datetime import datetime, timezone

import mysql.connector
import numpy as np
import pandas as pd
import pymongo
from pymongo.errors import BulkWriteError, PyMongoError

# Fenetres des indicateurs. Regroupees ici pour etre modifiables en un endroit.
MA_SHORT = 5
MA_LONG = 20
VOL_WINDOW = 20
RSI_WINDOW = 14

FEATURE_COLS = [
    "daily_return",
    "ma_5",
    "ma_20",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "hl_range",
    "gap",
]


def get_staging_data(host, user, password, database, tickers=None):
    """
    Recupere les cotations de la zone Staging.

    La lecture se fait au curseur plutot qu'avec pandas.read_sql : ce dernier
    attend un moteur SQLAlchemy et emet un avertissement avec un connecteur brut.

    Returns
    -------
    pandas.DataFrame or None
        None si la connexion echoue.
    """
    query = """
        SELECT ticker, quote_date, open, high, low, close, volume, source
        FROM quotes
    """
    params = []

    if tickers:
        placeholders = ", ".join(["%s"] * len(tickers))
        query += f" WHERE ticker IN ({placeholders})"
        params = tickers

    query += " ORDER BY ticker, source, quote_date"

    try:
        conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        cursor = conn.cursor()
        cursor.execute(query, params or None)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return pd.DataFrame(rows, columns=columns)

    except mysql.connector.Error as e:
        print(f"ERREUR : lecture de la zone Staging impossible ({e}).")
        return None


def compute_rsi(close, window=RSI_WINDOW):
    """
    Calcule le Relative Strength Index.

    Le RSI compare l'ampleur moyenne des hausses a celle des baisses sur une
    fenetre glissante. Il oscille entre 0 et 100 : au-dela de 70 la valeur est
    consideree sur-achetee, en dessous de 30 sur-vendue.

    Parameters
    ----------
    close : pandas.Series
        Cours de cloture, tries chronologiquement.
    window : int
        Taille de la fenetre.

    Returns
    -------
    pandas.Series
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Une moyenne de pertes nulle donnerait une division par zero.
    # Dans ce cas la valeur n'a connu que des hausses : le RSI vaut 100.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)

    return rsi


def compute_features(group):
    """
    Calcule les indicateurs techniques sur un groupe (ticker, source).

    Le groupe doit etre trie par date : toutes les fenetres glissantes en
    dependent.

    Parameters
    ----------
    group : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Le groupe augmente des colonnes de FEATURE_COLS.
    """
    group = group.sort_values("quote_date").copy()

    close = group["close"].astype(float)
    volume = group["volume"].astype(float)

    group["daily_return"] = close.pct_change()
    group["ma_5"] = close.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
    group["ma_20"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()

    group["volatility_20"] = (
        group["daily_return"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    )

    group["rsi_14"] = compute_rsi(close)

    avg_volume = volume.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()
    group["volume_ratio"] = volume / avg_volume.replace(0, np.nan)

    group["hl_range"] = (
        group["high"].astype(float) - group["low"].astype(float)
    ) / close

    group["gap"] = (group["open"].astype(float) - close.shift(1)) / close.shift(1)

    return group


def build_features(df):
    """
    Applique le calcul des indicateurs a chaque couple (ticker, source).

    La boucle explicite est preferee a groupby().apply() : depuis pandas 3.0,
    ce dernier n'expose plus les colonnes de groupement au sous-DataFrame.

    Les premieres lignes de chaque groupe n'ont pas assez d'historique pour
    remplir une fenetre de 20 jours : elles sortent avec des NaN et sont
    ecartees. C'est un choix assume, pas un oubli — une moyenne mobile calculee
    sur 3 points ne serait pas comparable a une moyenne sur 20.

    Returns
    -------
    pandas.DataFrame
    """
    print(f"  Calcul des indicateurs sur {df['ticker'].nunique()} valeurs...")

    frames = []
    for (ticker, source), group in df.groupby(["ticker", "source"], sort=False):
        enriched = compute_features(group)
        enriched["ticker"] = ticker
        enriched["source"] = source
        frames.append(enriched)

    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    after = len(df)

    print(f"    lignes avec historique complet : {after}")
    print(f"    lignes ecartees (amorcage)     : {before - after}")

    # Les infinis viennent de divisions par des volumes ou des prix nuls
    # residuels. Ils casseraient l'entrainement du modele : on les retire.
    df = df[np.isfinite(df[FEATURE_COLS]).all(axis=1)]
    print(f"    lignes apres retrait des infinis : {len(df)}")

    return df


def build_documents(df):
    """
    Transforme les lignes en documents MongoDB.

    La structure imbriquee separe explicitement les trois natures d'information :
    la cotation d'origine (ohlcv), ce qu'on en a derive (features), et la
    tracabilite (metadata). Le bloc 'anomaly' viendra s'y greffer a l'etape ML.

    Returns
    -------
    list of dict
    """
    computed_at = datetime.now(timezone.utc).isoformat()

    documents = []
    for row in df.itertuples(index=False):
        documents.append(
            {
                "ticker": row.ticker,
                "quote_date": row.quote_date.to_pydatetime(),
                "source": row.source,
                "ohlcv": {
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": int(row.volume),
                },
                "features": {
                    "daily_return": float(row.daily_return),
                    "ma_5": float(row.ma_5),
                    "ma_20": float(row.ma_20),
                    "volatility_20": float(row.volatility_20),
                    "rsi_14": float(row.rsi_14),
                    "volume_ratio": float(row.volume_ratio),
                    "hl_range": float(row.hl_range),
                    "gap": float(row.gap),
                },
                "metadata": {
                    "origin": "mysql_staging",
                    "computed_at": computed_at,
                },
            }
        )

    return documents


def load_to_curated(df, mongo_uri, database, collection, chunk_size=20000):
    """
    Construit et insere les documents par tranches.

    Materialiser 600 000 dictionnaires imbriques avant de les inserer saturait
    la memoire (le processus etait tue par l'OS). On traite donc le DataFrame
    par tranches : construction, insertion, liberation. La memoire consommee
    devient constante, quel que soit le volume total.

    L'index unique (ticker, quote_date, source) rejette les documents deja
    presents ; avec ordered=False, l'insertion se poursuit malgre ces rejets,
    ce qui rend le script rejouable.

    Returns
    -------
    int
        Nombre de documents reellement inseres.
    """
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    coll = client[database][collection]

    coll.create_index(
        [("ticker", 1), ("quote_date", 1), ("source", 1)],
        unique=True,
        name="uk_quote",
    )
    coll.create_index([("ticker", 1)], name="idx_ticker")

    total = len(df)
    inserted = 0
    duplicates = 0

    for start in range(0, total, chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        documents = build_documents(chunk)

        try:
            result = coll.insert_many(documents, ordered=False)
            inserted += len(result.inserted_ids)

        except BulkWriteError as e:
            # Le code 11000 signale un doublon : comportement attendu lors
            # d'un rejeu, pas un echec.
            errors = e.details.get("writeErrors", [])
            duplicates += len([err for err in errors if err.get("code") == 11000])
            inserted += e.details.get("nInserted", 0)

            others = [err for err in errors if err.get("code") != 11000]
            if others:
                print(f"    ERREURS non liees aux doublons : {len(others)}")

        print(f"    {min(start + chunk_size, total)}/{total} traites...")

    client.close()

    if duplicates:
        print(f"  {duplicates} documents deja presents (ecartes).")

    return inserted


def verify_mongodb(mongo_uri, database, collection):
    """
    Controle qualite de la zone Curated.

    Affiche la volumetrie, un document temoin, et les statistiques de
    volatilite : c'est cette derniere qui alimentera le modele.
    """
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    coll = client[database][collection]

    total = coll.count_documents({})
    print(f"\n  Documents dans Curated : {total}")

    print(f"  Tickers distincts      : {len(coll.distinct('ticker'))}")

    sample = coll.find_one()
    if sample:
        print("\n  Document temoin :")
        print(f"    {sample['ticker']} au {sample['quote_date'].date()}")
        print(f"    cloture       : {sample['ohlcv']['close']}")
        print(f"    volatilite 20j: {sample['features']['volatility_20']:.5f}")
        print(f"    RSI 14        : {sample['features']['rsi_14']:.2f}")

    pipeline = [
        {
            "$group": {
                "_id": None,
                "avg_vol": {"$avg": "$features.volatility_20"},
                "max_vol": {"$max": "$features.volatility_20"},
                "avg_rsi": {"$avg": "$features.rsi_14"},
            }
        }
    ]
    stats = list(coll.aggregate(pipeline))

    if stats:
        s = stats[0]
        print("\n  Statistiques des indicateurs :")
        print(f"    volatilite moyenne : {s['avg_vol']:.5f}")
        print(f"    volatilite max     : {s['max_vol']:.5f}")
        print(f"    RSI moyen          : {s['avg_rsi']:.2f}")

    client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Enrichit la zone Staging (MySQL) vers la zone Curated (MongoDB)."
    )
    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-user", type=str, default="root")
    parser.add_argument("--db-password", type=str, default="root")
    parser.add_argument("--db-name", type=str, default="staging")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/")
    parser.add_argument("--mongo-db", type=str, default="curated")
    parser.add_argument("--mongo-collection", type=str, default="quotes_features")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Restreint le traitement a certaines valeurs (ex. AAPL,MSFT).",
    )
    args = parser.parse_args()

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    # 1. Lecture de la zone Staging
    print("Lecture de la zone Staging (MySQL)...")
    df = get_staging_data(
        args.db_host, args.db_user, args.db_password, args.db_name, tickers
    )

    if df is None:
        return

    if df.empty:
        print("ERREUR : la zone Staging est vide. Lancez d'abord load_to_staging.py.")
        return

    print(f"  {len(df)} cotations lues.")

    # 2. Enrichissement
    print("Enrichissement...")
    df = build_features(df)

    if df.empty:
        print("ERREUR : aucune ligne ne dispose d'un historique suffisant.")
        return

   # 3. Chargement dans la zone Curated
    print(f"Chargement dans MongoDB ({len(df)} documents)...")
    try:
        inserted = load_to_curated(
            df, args.mongo_uri, args.mongo_db, args.mongo_collection
        )
    except PyMongoError as e:
        print(f"ERREUR : zone Curated injoignable ({e}).")
        return

    # 4. Controle qualite
    print("\n--- Validation de la zone Curated ---")
    print(f"  Documents inseres ce run : {inserted}")
    verify_mongodb(args.mongo_uri, args.mongo_db, args.mongo_collection)

    print("\nZone Curated alimentee. On peut entrainer le modele.")


if __name__ == "__main__":
    main()