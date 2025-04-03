# Set Banxico API token, token validator, and general settings for SQL DB, and project

import os
import pathlib

import requests
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv, find_dotenv

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

db_connection = {
    'host': host,
    'user': user,
    'password': password,
    'db': 'mei',
}

today_str = str(datetime.today().date())


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
