"""
Etape 1 : Test des connexions aux trois zones du data lake.

Ce script est le point de controle de l'environnement. Il doit passer au vert
AVANT d'ecrire la moindre ligne de pipeline : si une zone n'est pas joignable,
tous les scripts suivants echoueront de toute facon.

    Raw     -> LocalStack (S3)   sur localhost:4566
    Staging -> MySQL             sur localhost:3306
    Curated -> MongoDB           sur localhost:27017

Usage:
    docker compose up -d
    (attendre ~30s que MySQL s'initialise)
    python src/test_connections.py
"""
import argparse

import boto3
import mysql.connector
import pymongo
from botocore.exceptions import BotoCoreError, ClientError


def test_s3(endpoint_url, bucket):
    """
    Verifie que la zone Raw (LocalStack S3) est joignable.

    Le test cree le bucket s'il n'existe pas, y depose un petit objet temoin,
    le relit, puis le supprime. Cela valide a la fois la connexion et les
    droits en ecriture/lecture.

    Parameters
    ----------
    endpoint_url : str
        URL du service S3 (LocalStack en local).
    bucket : str
        Nom du bucket de la zone Raw.

    Returns
    -------
    bool
        True si la zone Raw est operationnelle.
    """
    print("Test de connexion S3 (LocalStack)...")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
        )

        # Creation idempotente du bucket : ne leve pas d'erreur s'il existe deja
        existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
        if bucket not in existing:
            s3.create_bucket(Bucket=bucket)
            print(f"  Bucket '{bucket}' cree.")

        # Aller-retour d'un objet temoin
        s3.put_object(Bucket=bucket, Key="_healthcheck.txt", Body=b"ok")
        body = s3.get_object(Bucket=bucket, Key="_healthcheck.txt")["Body"].read()
        s3.delete_object(Bucket=bucket, Key="_healthcheck.txt")

        print(f"  S3 OK : bucket '{bucket}', lecture = {body.decode()}")
        return True

    except (BotoCoreError, ClientError) as e:
        print(f"  S3 ERREUR : {e}")
        return False


def test_mysql(host, user, password, database):
    """
    Verifie que la zone Staging (MySQL) est joignable.

    Returns
    -------
    bool
        True si la connexion et une requete simple aboutissent.
    """
    print("Test de connexion MySQL...")
    try:
        conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        print(f"  MySQL OK : base '{database}', version {version}")
        cursor.close()
        conn.close()
        return True

    except mysql.connector.Error as e:
        print(f"  MySQL ERREUR : {e}")
        return False


def test_mongodb(uri):
    """
    Verifie que la zone Curated (MongoDB) est joignable.

    Le test insere un document temoin puis nettoie derriere lui.

    Returns
    -------
    bool
        True si la zone Curated est operationnelle.
    """
    print("Test de connexion MongoDB...")
    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.server_info()  # Force la connexion (sinon MongoClient est paresseux)

        db = client["_healthcheck"]
        db["ping"].insert_one({"status": "ok"})
        result = db["ping"].find_one()
        print(f"  MongoDB OK : {result['status']}")

        client.drop_database("_healthcheck")
        client.close()
        return True

    except pymongo.errors.PyMongoError as e:
        print(f"  MongoDB ERREUR : {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Teste les connexions aux trois zones du data lake."
    )
    parser.add_argument("--s3-endpoint", type=str, default="http://localhost:4566")
    parser.add_argument("--bucket", type=str, default="raw")
    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-user", type=str, default="root")
    parser.add_argument("--db-password", type=str, default="root")
    parser.add_argument("--db-name", type=str, default="staging")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/")
    args = parser.parse_args()

    s3_ok = test_s3(args.s3_endpoint, args.bucket)
    mysql_ok = test_mysql(args.db_host, args.db_user, args.db_password, args.db_name)
    mongo_ok = test_mongodb(args.mongo_uri)

    print("\n--- Resume ---")
    print(f"Raw     (S3)      : {'OK' if s3_ok else 'ECHEC'}")
    print(f"Staging (MySQL)   : {'OK' if mysql_ok else 'ECHEC'}")
    print(f"Curated (MongoDB) : {'OK' if mongo_ok else 'ECHEC'}")

    if s3_ok and mysql_ok and mongo_ok:
        print("\nLes trois zones sont pretes. On peut passer a l'ingestion.")
    else:
        print("\nCorrigez les erreurs avant de continuer (docker compose ps).")


if __name__ == "__main__":
    main()