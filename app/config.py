# Set Banxico API token, token validator, and general settings for SQL DB, and project

import os
import pathlib

import requests
import pandas as pd
from datetime import datetime

import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import mysql.connector
from mysql.connector import errorcode

from dotenv import load_dotenv, find_dotenv


logging.basicConfig(level=logging.INFO)
load_dotenv(find_dotenv())

DEBUG = True

# Root or main directory for the app.
ROOT_DIR = pathlib.Path(__file__).parent.resolve()

# BANXICO API KEY
token = os.getenv("BANXICO_API_KEY")
series =  os.getenv("BANXICO_SERIES_EXAMPLE")

# MySQL settings
host = os.getenv('MySQL_HOST')
user = os.getenv('MySQL_USER')
password = os.getenv('MYSQL_PWD')
port = os.getenv('MySQL_PORT')
db = "mei"

db_connection = {
    'host': host,
    'user': user,
    'password': password,
    'db': 'mei',
}

today_str = str(datetime.today().date())


def ensure_database() -> None:
    """
    Verifica si existe la base de datos `dbname`. Si no existe, la crea.
    """
    logger = logging.getLogger("DB")
    dbname = db
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
        )
        cursor = conn.cursor()
        # Verificar existencia
        cursor.execute("SHOW DATABASES LIKE %s", (dbname,))
        exists = cursor.fetchone()
        if exists:
            logger.info(f"✅ La base de datos '{dbname}' ya existe.")
        else:
            logger.info(f"⚙️  Creando la base de datos '{dbname}' …")
            cursor.execute(
                f"CREATE DATABASE `{dbname}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
            logger.info(f"✅ Base de datos '{dbname}' creada.")
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            logger.error("Credenciales inválidas para MySQL", exc_info=True)
        else:
            logger.error("Error al verificar/crear la base de datos", exc_info=True)
        raise

def init_db_engine(echo: bool = False, pool_pre_ping: bool = True):
    """
    Crea y valida el Engine de SQLAlchemy para MySQL.
    - echo: si True, SQLAlchemy imprimirá cada sentencia SQL (útil en desarrollo).
    - pool_pre_ping: activa comprobación previa en cada checkout del pool
      para evitar errores de conexiones muertas.
    Retorna el engine si la conexión es exitosa, lanza excepción si falla.
    """
    logger = logging.getLogger("DB")
    ensure_database()
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    
    engine = create_engine(
        url,
        echo=echo,
        pool_pre_ping=pool_pre_ping,
        future=True  # usar la API 2.0 de SQLAlchemy
    )
    # Validación física de la conexión
    try:
        with engine.connect() as conn:
            logger.info("✅ Conexión a MySQL exitosa")
        return engine
    except OperationalError as e:
        logger.error("❌ No se pudo conectar a MySQL", exc_info=e)
        raise

init_db_engine(echo=False)


def token_validator(token=token, series=series):
    end_date = pd.to_datetime('today', format='%Y-%m-%d')
    start_date = "2024-01-01"
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    url = f'https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series}/datos/{start_date}/{date_range[-1].date()}'

    print(url)
    headers = {'Bmx-Token': token}
    response = requests.get(url, headers=headers)
    status = response.status_code

    if status == 400:
        print("Error 400: Token expirado. Genera un nuevo token en: https://www.banxico.org.mx/SieAPIRest/service/v1/token")
        return None
    elif status != 200:
        print(f"Error, status code: {status}. Recomendamos indagar sobre las causas del error y generar un nuevo token en: https://www.banxico.org.mx/SieAPIRest/service/v1/token")
        return None

    elif status == 200:
        print("Token de Banxico Validado...continua con tu consulta")


def path_validator():
    if not os.path.exists('./app/data'):
        # os.mkdir('./app/data')
        os.makedirs('./app/data')

    if not os.path.exists('./app/viz'):
        os.makedirs('app/viz')

    print("Project path files OK!")


path_validator()

# Evita imprimir información innecesaria
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
