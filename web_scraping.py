import pandas as pd

import numpy as np
import requests
import json
import urllib

from data_gathering import get_time


def download_bmx_timeseries(token, series, start_date: str):
    """
    Web Scraping function to get time series data from Banxico repository.
    :param series: IDSerie, ej. SF63528 (str). https://www.banxico.org.mx/SieAPIRest/service/v1/
    :param start_date: date format yyyy-mm-dd.
    :param token: API token Banxico, length 64 characters. https://www.banxico.org.mx/SieAPIRest/service/v1/token
    :return: Pandas DataFrame.
    """
    end_date = pd.to_datetime('today', format='%Y-%m-%d')
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
        print(f"Error, status code: {status}")
        return None

    raw_data = response.json()
    data = raw_data['bmx']['series'][0]['datos']
    df = pd.DataFrame(data)
    df.replace(to_replace='N/E', value='NaN', inplace=True)
    df['dato'] = df['dato'].apply(lambda x: float(x))
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
    df.set_index('fecha', inplace=True)

    return df


def get_mme_prices():
    """
    Only web scrape oil & gas mx prices, no token require.
    Args:
        - No args.
    Return:
        - pandas DataFrame.
    """
    # https://stackoverflow.com/questions/53426787/how-to-replace-a-float-value-with-nan-in-pandas
    # https://stackoverflow.com/questions/15138614/how-can-i-read-the-contents-of-an-url-with-python
    # Precios MME Banxico
    url = "https://www.banxico.org.mx/SieInternet/consultaSerieGrafica.do?s=SI744,CI38"
    f = urllib.request.urlopen(url)
    myfile = f.read()

    # Get string data format
    string_data = str(myfile, 'utf-8')

    # Convert string to json format
    json_data = json.loads(string_data)

    # Tabular data
    df = pd.DataFrame(json_data['valores'], columns=['fecha', 'MME'])

    # replace values "-989898.00" related with NaN values 
    array = np.where(np.isclose(df['MME'].values, -989898.00), np.nan, df['MME'].values)

    # Transform to pandas DataFrame
    mme_clean = pd.DataFrame(array, columns=['MME'])
    mme_clean

    # Insert NaN values
    df['MME'] = mme_clean

    # Set datetimeindex
    df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
    df.set_index('fecha', inplace=True)

    return df


def download_production_rates(
    url_file='https://sih.hidrocarburos.gob.mx/downloads/PRODUCCION_POZOS.zip',
    filepath='./app/data/PRODUCCION_POZOS.zip'
):
    """
    Get all hydrocarbons production rates since 1930 until today, Mexico.
    Args:
        -url_file: download url.
        -filepath: path to download url file
    """
    file_response = requests.get(url_file)
    # Save the downloaded file
    with open(filepath, 'wb') as file:
        file.write(file_response.content)

    print(f"Download at...{get_time()}")
