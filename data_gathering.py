from sqlalchemy import create_engine, text
import pandas as pd
import pymysql

import os
import time
import zipfile


def get_time() -> str:
    """
    Return current datetime.
    """
    return time.strftime('%X (%d/%m/%Y)')


def send_sql_data(db_connection: dict, df: pd.DataFrame, name: str):
    """
    Save data gathering into mySQL database.
    Args: 
        - db_connection: database connection credentials. (host, user, password, database)
        - df: pandas DataFrame.
        - name: new table name.
    Returns:
        - None
    """
    try:
        engine = create_engine(f"mysql+pymysql://{db_connection['user']}:{db_connection['password']}@{db_connection['host']}/{db_connection['db']}")

        df.to_sql(name, engine, if_exists='fail')

        print("Data send succesfully!")
    except Exception as e:
        print(f"Error sending data to SQL: {e}")


def read_sql_data(db_connection: dict, tablename: str):
    """
    Load data from mySQL database.
    Args: 
        - db_connection: database connection credentials. (host, user, password, database)
        - tablename: table name to query.
    Returns:
        - pandas DataFrame.
    """
    engine = create_engine(f"mysql+pymysql://{db_connection['user']}:{db_connection['password']}@{db_connection['host']}/{db_connection['db']}")
    query = f'SELECT * FROM {tablename}'
    df = pd.read_sql_query(text(query), con=engine.connect())

    return df


def compare_and_aggregate(local_data, db_data_query, db_connection, table_name):
    """
    Compares local data with database data and aggregates new data to a MySQL table.

    Args:
        local_data (pandas DataFrame): Local data to compare.
        db_data_query (str): SQL query to retrieve database data.
        db_connection (dict): Dictionary containing MySQL database connection details.

    Returns:
        None
    """
    # Check if local_data is a Pandas DataFrame
    if not isinstance(local_data, pd.DataFrame):
        raise ValueError("local_data must be a Pandas DataFrame")

    # Check if local_data has at least one row
    if local_data.empty:
        print("local_data is empty, no comparison or aggregation, check your data.")
        return None

    # Connect to MySQL database
    conn = pymysql.connect(
        host=db_connection['host'],
        user=db_connection['user'],
        password=db_connection['password'],
        db=db_connection['db'],
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        # Retrieve data from MySQL table
        with conn.cursor() as cursor:
            cursor.execute(db_data_query)
            result = cursor.fetchall()

        # Convert MySQL data to DataFrame
        db_df = pd.DataFrame(result)

        # Check if local_data has the same number of columns as db_df
        if local_data.shape[1] != db_df.shape[1]:
            raise ValueError("local_data has a different number of columns compared to database data")

        # Compare local data with database data
        common_cols = list(set(local_data.columns) & set(db_df.columns))
        new_data = local_data.loc[~local_data[common_cols].apply(tuple, axis=1).isin(db_df[common_cols].apply(tuple, axis=1))]

        # Aggregate new data to MySQL table
        if not new_data.empty:
            with conn.cursor() as cursor:
                for _, row in new_data.iterrows():
                    cols = ', '.join(new_data.columns)
                    vals = ', '.join([f"'{value}'" for value in row.values])
                    query = f"INSERT INTO {table_name} ({cols}) VALUES ({vals})"
                    cursor.execute(query)
                conn.commit()
            print(f"{len(new_data)} rows of new data added to MySQL table.")
        else:
            print("No new data to add to MySQL table.")
    finally:
        # Close database connection
        conn.close()


def zip_2_df(zip_path='./data/PRODUCCION_POZOS.zip', csv_filename='POZOS_COMPILADO.csv', encoding="ISO-8859-1", h=10, short_data=False):
    """
    Read data from zip file and convert it to pandas DataFrame.
    Args:
        -zip_path: path to zip file, python string.
        -csv_filename: python string.
        -encoding: encoder to read csv data, default ISO-8859-1 or latin 1.
        -h: header to read data, integer.
        -short_data: if false, param low_memory from pandas dataframe turns false.
    Returns:
        -pandas DataFrame.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(csv_filename, './data')

    csv_path = os.path.join('./data', csv_filename)
    df = pd.read_csv(csv_path, encoding=encoding, header=h, low_memory=short_data)

    return df
