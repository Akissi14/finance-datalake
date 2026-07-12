"""
Etape 6a : Entrainement de l'autoencodeur de detection d'anomalies.

PRINCIPE
--------
Un autoencodeur est un reseau de neurones entraine a reproduire son entree en
sortie, en la faisant transiter par une couche cachee plus etroite. Cette
compression le contraint a apprendre la structure du comportement 'normal'.

    8 features -> 6 -> 3 -> 6 -> 8 features

Sur une journee ordinaire, il reconstruit fidelement : l'erreur est faible.
Sur un krach flash ou une bulle de volume, il n'a jamais rien vu de tel :
l'erreur explose. L'erreur de reconstruction EST le score d'anomalie.

CHOIX DES FEATURES
------------------
Les indicateurs de la zone Curated ne sont pas tous utilisables tels quels.
ma_5 et ma_20 sont des prix absolus : Amazon cote plusieurs milliers de dollars,
Coca-Cola quelques dizaines. Un modele global apprendrait surtout 'Amazon est
chere', pas 'cette journee est anormale'.

On derive donc huit features SANS ECHELLE, comparables d'une valeur a l'autre :
rendements, ratios, indicateurs bornes. C'est la condition pour qu'un modele
unique ait du sens sur les 505 valeurs.

SEPARATION ENTRAINEMENT / INFERENCE
-----------------------------------
Ce script entraine et sauvegarde. C'est apply_model.py qui score. L'entrainement
est couteux et tournera en @weekly dans Airflow ; l'inference est legere et
tourne a chaque passage du pipeline.

Le modele est serialise dans MinIO sous 'models/', aux cotes des donnees brutes :
un artefact de ML est une donnee comme une autre, il a sa place dans le lac.

Usage:
    python src/train_model.py
    python src/train_model.py --contamination 0.02 --sample-size 50000
"""
import argparse
import io
import pickle
from datetime import datetime, timezone

import boto3
import numpy as np
import pymongo
from botocore.exceptions import BotoCoreError, ClientError
from pymongo.errors import PyMongoError
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Les huit features du modele, dans l'ordre. Toutes sans echelle.
MODEL_FEATURES = [
    "daily_return",
    "volatility_20",
    "rsi_norm",
    "volume_ratio",
    "hl_range",
    "gap",
    "ma_ratio",
    "price_to_ma20",
]


def build_model_matrix(documents):
    """
    Derive les features du modele a partir des documents de la zone Curated.

    Les indicateurs bruts (ma_5, ma_20, close) sont transformes en ratios afin
    d'etre comparables entre des valeurs dont les prix different d'un facteur
    cent. Un document dont un denominateur serait nul est ecarte.

    Parameters
    ----------
    documents : list of dict
        Documents issus de la collection Curated.

    Returns
    -------
    numpy.ndarray
        Matrice (n_documents_valides, 8).
    """
    rows = []

    for doc in documents:
        f = doc["features"]
        close = doc["ohlcv"]["close"]
        ma_20 = f["ma_20"]
        ma_5 = f["ma_5"]

        # Un prix moyen nul n'a pas de sens : la ligne est ignoree plutot que
        # de produire un infini qui casserait l'entrainement.
        if ma_20 == 0 or ma_5 == 0:
            continue

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

    return np.array(rows, dtype=np.float64)


def load_training_sample(mongo_uri, database, collection, sample_size):
    """
    Charge un echantillon aleatoire de la zone Curated pour l'entrainement.

    Entrainer sur les 600 000 documents serait long sans gain notable : un
    echantillon aleatoire suffit a capturer la distribution du comportement
    normal. L'INFERENCE, elle, portera bien sur l'integralite des documents.

    L'echantillonnage se fait cote MongoDB ($sample), pas en Python : on ne
    rapatrie que ce dont on a besoin.

    Returns
    -------
    numpy.ndarray
    """
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    coll = client[database][collection]

    total = coll.count_documents({})
    print(f"  {total} documents dans la zone Curated.")

    if total == 0:
        client.close()
        return np.array([])

    size = min(sample_size, total)
    print(f"  Echantillon d'entrainement : {size} documents.")

    documents = list(coll.aggregate([{"$sample": {"size": size}}]))
    client.close()

    return build_model_matrix(documents)


def train_autoencoder(X, hidden_layers=(6, 3, 6), max_iter=60, random_state=42):
    """
    Entraine l'autoencodeur.

    Un autoencodeur est un reseau entraine a predire sa propre entree : on
    passe donc X en entree ET en cible. La couche centrale a 3 neurones est le
    goulot d'etranglement qui force la compression.

    Parameters
    ----------
    X : numpy.ndarray
        Matrice des features, standardisee.
    hidden_layers : tuple
        Architecture des couches cachees.
    max_iter : int
        Nombre maximal d'epoques.

    Returns
    -------
    sklearn.neural_network.MLPRegressor
    """
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )

    # La cible EST l'entree : c'est ce qui fait de ce reseau un autoencodeur.
    model.fit(X, X)

    return model


def reconstruction_errors(model, X):
    """
    Calcule l'erreur de reconstruction de chaque observation.

    L'erreur est l'ecart quadratique moyen entre l'entree et sa reconstruction.
    Elle constitue le score d'anomalie : plus elle est elevee, plus le modele
    a ete surpris.

    Returns
    -------
    numpy.ndarray
        Un score par observation.
    """
    X_reconstructed = model.predict(X)
    return np.mean((X - X_reconstructed) ** 2, axis=1)


def compute_threshold(errors, contamination):
    """
    Determine le seuil au-dela duquel une observation est declaree anormale.

    Le seuil est le quantile correspondant au taux de contamination attendu :
    avec contamination=0.01, on considere que 1 % des journees sont anormales,
    et le seuil est donc le 99e percentile des erreurs d'entrainement.

    C'est une hypothese, et elle doit etre assumee : on ne connait pas la
    'vraie' proportion d'anomalies. Le taux de contamination est un parametre
    metier, pas une verite statistique.

    Returns
    -------
    float
    """
    return float(np.percentile(errors, 100 * (1 - contamination)))


def save_model_to_raw(artifact, s3, bucket, key):
    """
    Serialise le modele et le depose dans MinIO.

    Le modele voyage avec son scaler, son seuil et la liste ordonnee de ses
    features : sans eux, il est inutilisable. Un artefact de ML n'est pas
    seulement des poids, c'est un contrat d'inference complet.
    """
    buffer = io.BytesIO()
    pickle.dump(artifact, buffer)
    buffer.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Entraine l'autoencodeur de detection d'anomalies."
    )
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/")
    parser.add_argument("--mongo-db", type=str, default="curated")
    parser.add_argument("--mongo-collection", type=str, default="quotes_features")
    parser.add_argument("--bucket", type=str, default="raw")
    parser.add_argument("--model-key", type=str, default="models/autoencoder.pkl")
    parser.add_argument("--s3-endpoint", type=str, default="http://localhost:9000")
    parser.add_argument("--s3-access-key", type=str, default="minioadmin")
    parser.add_argument("--s3-secret-key", type=str, default="minioadmin")
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Proportion attendue d'anomalies (fixe le seuil).",
    )
    args = parser.parse_args()

    # 1. Echantillon d'entrainement
    print("Lecture de la zone Curated...")
    try:
        X_raw = load_training_sample(
            args.mongo_uri, args.mongo_db, args.mongo_collection, args.sample_size
        )
    except PyMongoError as e:
        print(f"ERREUR : zone Curated injoignable ({e}).")
        return

    if X_raw.size == 0:
        print("ERREUR : aucune donnee. Lancez d'abord staging_to_curated.py.")
        return

    print(f"  Matrice d'entrainement : {X_raw.shape[0]} x {X_raw.shape[1]}")

    # 2. Standardisation
    # Sans elle, la volatilite (de l'ordre de 0.01) pesera mille fois moins
    # que le volume_ratio (de l'ordre de 1) dans l'erreur quadratique.
    print("Standardisation des features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 3. Entrainement
    print("Entrainement de l'autoencodeur (8 -> 6 -> 3 -> 6 -> 8)...")
    model = train_autoencoder(X)
    print(f"  Convergence en {model.n_iter_} epoques.")

    # 4. Qualite de la reconstruction
    errors = reconstruction_errors(model, X)
    mse = mean_squared_error(X, model.predict(X))

    print(f"  Erreur de reconstruction moyenne : {mse:.5f}")
    print(f"  Erreur mediane   : {np.median(errors):.5f}")
    print(f"  Erreur maximale  : {np.max(errors):.5f}")

    # 5. Seuil d'anomalie
    threshold = compute_threshold(errors, args.contamination)
    flagged = int(np.sum(errors > threshold))

    print(f"\n  Contamination retenue : {args.contamination:.1%}")
    print(f"  Seuil d'anomalie      : {threshold:.5f}")
    print(f"  Anomalies dans l'echantillon : {flagged} / {len(errors)}")

    # 6. Sauvegarde dans la zone Raw
    artifact = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "features": MODEL_FEATURES,
        "contamination": args.contamination,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_size": int(X.shape[0]),
    }

    print(f"\nSauvegarde du modele dans s3://{args.bucket}/{args.model_key} ...")
    s3 = boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint,
        aws_access_key_id=args.s3_access_key,
        aws_secret_access_key=args.s3_secret_key,
        region_name="us-east-1",
    )

    try:
        save_model_to_raw(artifact, s3, args.bucket, args.model_key)
    except (BotoCoreError, ClientError) as e:
        print(f"ERREUR : sauvegarde impossible ({e}).")
        return

    print("\nModele entraine et sauvegarde. On peut passer au scoring.")


if __name__ == "__main__":
    main()
    