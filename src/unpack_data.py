"""
Etape 2 : Ingestion de la source fichier (dataset S&P 500) vers la zone Raw.

La zone Raw stocke la donnee **telle qu'elle a ete recue**, sans transformation.
Aucun nettoyage, aucun typage : c'est la copie de reference, immuable, qui permet
de rejouer tout le pipeline si la logique aval change.

Les fichiers sont deposes dans MinIO sous le prefixe 'source_dataset/', afin de
les separer de la seconde source (l'API), qui ira sous 'source_api/'.

Usage:
    python src/unpack_data.py \
        --input-dir data/individual_stocks_5yr \
        --bucket raw \
        --prefix source_dataset
"""
import argparse
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def get_s3_client(endpoint_url, access_key, secret_key):
    """
    Cree un client S3 pointant vers MinIO.

    MinIO expose l'API S3 : le code serait identique sur AWS, seul l'endpoint
    change. C'est ce qui rend la zone Raw portable.

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
    Cree le bucket s'il n'existe pas encore.

    L'operation est idempotente : relancer le script ne provoque pas d'erreur.
    C'est une exigence de base d'un pipeline reproductible.
    """
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket)
        print(f"  Bucket '{bucket}' cree.")
    else:
        print(f"  Bucket '{bucket}' deja present.")


def list_csv_files(input_dir):
    """
    Liste les fichiers CSV a ingerer.

    Parameters
    ----------
    input_dir : str
        Repertoire contenant les CSV du dataset.

    Returns
    -------
    list of pathlib.Path
        Fichiers CSV tries par nom.

    Raises
    ------
    FileNotFoundError
        Si le repertoire n'existe pas ou ne contient aucun CSV.
    """
    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(
            f"Repertoire introuvable : {input_path}. "
            "Avez-vous telecharge le dataset Kaggle ?"
        )

    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV dans {input_path}.")

    return csv_files


def upload_files(s3, bucket, prefix, csv_files):
    """
    Depose chaque CSV dans la zone Raw, sans le modifier.

    Les fichiers vides sont ignores (et signales) plutot que de polluer le lac :
    c'est une regle de gouvernance minimale contre le 'data swamp'.

    Parameters
    ----------
    s3 : botocore.client.S3
    bucket : str
        Bucket de la zone Raw.
    prefix : str
        Prefixe identifiant la source (ici 'source_dataset').
    csv_files : list of pathlib.Path

    Returns
    -------
    tuple of (int, int)
        Nombre de fichiers ingeres, nombre de fichiers ignores.
    """
    uploaded = 0
    skipped = 0

    for csv_file in csv_files:
        if csv_file.stat().st_size == 0:
            print(f"  IGNORE (fichier vide) : {csv_file.name}")
            skipped += 1
            continue

        key = f"{prefix}/{csv_file.name}"

        try:
            s3.upload_file(str(csv_file), bucket, key)
            uploaded += 1

            if uploaded % 100 == 0:
                print(f"  {uploaded} fichiers ingeres...")

        except (BotoCoreError, ClientError) as e:
            print(f"  ERREUR sur {csv_file.name} : {e}")
            skipped += 1

    return uploaded, skipped


def count_objects(s3, bucket, prefix):
    """
    Compte les objets presents sous un prefixe donne.

    Sert de controle final : on verifie que ce qu'on a envoye est bien arrive.
    La pagination est necessaire car S3 ne renvoie que 1000 cles par appel.
    """
    paginator = s3.get_paginator("list_objects_v2")
    total = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        total += len(page.get("Contents", []))

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Ingere le dataset S&P 500 (CSV) dans la zone Raw."
    )
    parser.add_argument("--input-dir", type=str, default="data/individual_stocks_5yr/individual_stocks_5yr")
    parser.add_argument("--bucket", type=str, default="raw")
    parser.add_argument("--prefix", type=str, default="source_dataset")
    parser.add_argument("--s3-endpoint", type=str, default="http://localhost:9000")
    parser.add_argument("--s3-access-key", type=str, default="minioadmin")
    parser.add_argument("--s3-secret-key", type=str, default="minioadmin")
    args = parser.parse_args()

    # 1. Lister les fichiers a ingerer
    print(f"Lecture du repertoire '{args.input_dir}'...")
    try:
        csv_files = list_csv_files(args.input_dir)
    except FileNotFoundError as e:
        print(f"ERREUR : {e}")
        return
    print(f"  {len(csv_files)} fichiers CSV trouves.")

    # 2. Preparer la zone Raw
    print("Connexion a la zone Raw (MinIO)...")
    s3 = get_s3_client(args.s3_endpoint, args.s3_access_key, args.s3_secret_key)
    try:
        ensure_bucket(s3, args.bucket)
    except (BotoCoreError, ClientError) as e:
        print(f"ERREUR : zone Raw injoignable ({e}).")
        return

    # 3. Ingerer les fichiers, tels quels
    print(f"Ingestion vers s3://{args.bucket}/{args.prefix}/ ...")
    uploaded, skipped = upload_files(s3, args.bucket, args.prefix, csv_files)

    # 4. Controle : ce qu'on a envoye est-il bien arrive ?
    total = count_objects(s3, args.bucket, args.prefix)

    print("\n--- Resume de l'ingestion ---")
    print(f"Fichiers ingeres  : {uploaded}")
    print(f"Fichiers ignores  : {skipped}")
    print(f"Objets dans Raw   : {total}")

    if total == uploaded:
        print("\nZone Raw alimentee. On peut passer a l'ingestion de l'API.")
    else:
        print("\nATTENTION : ecart entre les fichiers envoyes et les objets presents.")


if __name__ == "__main__":
    main()