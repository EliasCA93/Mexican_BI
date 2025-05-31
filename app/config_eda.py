from data_gathering import read_sql_data
from app.config import db_connection

pib_mx = read_sql_data(db_connection, 'pib_mx')
pib_mx.drop("index", axis=1, inplace=True)
pib_mx["year"] = pib_mx["year"].astype(int)

indicadores_mx = read_sql_data(db_connection, 'indicadores_mx')
produccion_acumulada = read_sql_data(db_connection, 'produccion_acumulada')


if "Fecha" in list(produccion_acumulada.columns):
        produccion_acumulada.rename(columns={"Fecha": "fecha"}, inplace=True)


def pipeline(df):
    data = df.copy()
    data.drop_duplicates(subset="fecha", keep="first", inplace=True)
    data.set_index("fecha", inplace=True)

    if "dls_mxn" in list(data.columns):
        cols_std = {
            "dls_mxn": "Peso Mexicano (mxn)",
            "UDI": "UDI (mxn)",
            "MME": "MME (dólar)",
            "CETES": "CETES (%)",
            "inflacion_anual": "Inflación anual Banxico (%)",
            "inflacion_subyacente": "Inflación anual subyacente Banxico (%)",
            "interes_interbancario_28": "Tasa de referencia Banxico (%)"
        }
        data.rename(cols_std, axis=1, inplace=True)

    return data

df2charts = pipeline(indicadores_mx)
produccion_acumulada = pipeline(produccion_acumulada)

cols = list(df2charts.columns)

if len(df2charts) and len(produccion_acumulada) and len(cols) > 0:
    print("Set data for EDA OK!")