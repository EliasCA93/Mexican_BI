import numpy as np
import pandas as pd

from pylab import rcParams
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import seaborn as sns
import solara

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import scipy.cluster.hierarchy as sch

import json
import random

from app.config import db_connection, today_str
from data_gathering import read_sql_data


cols_std = {
    "dls_mxn": "Peso Mexicano (mxn)",
    "UDI": "Unidad de Inversión (mxn)",
    "MME": "Mezcla Mexicana de Exportación (dólar)",
    "CETES": "Bonos de la Tesoreria (%)",
    "inflacion_anual": "Inflación anual Banxico (%)",
    "inflacion_subyacente": "Inflación anual subyacente Banxico (%)",
    "interes_interbancario_28": "Tasa de referencia Banxico (%)"
}


def highlight_cells(val):
    """
    Conditonal formatting based on correlation values.
    Args:
        - val: values to compare.
    Returns:
        - format color
    """
    color = '#f5c242' if (round(val, 2) >= 0.6 or round(val, 2) <= -0.6) else '#4287f5' # Pastel blue

    return 'background-color: {}'.format(color)


# Matplotlib colors table 
# https://matplotlib.org/stable/gallery/color/named_colors.html

def plot_colortable(colors, sort_colors=True, emptycols=0):
    """
    Function to visualize the available colors in matplotlib.
    :param colors: CSS matplotlib colors (matplotlib.colors.CSS4_COLORS)
    :return: Matplotlib figure with available colors based on CSS.
    """

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig


def plot_correlation(color, title, df={}):
    """
    Function to plot pearson correlation gradients between variables.
    :param color: cmap heatmap option. i.e. "Blues", "coolwarm", "jet", etc.
    :param df: Pandas DataFrame.
    :returns: matplotlib figure.
    """
    # calculate pearson correlation
    corr = df.corr()

    # mask for mask heatmap parameter
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # create figure
    fig = plt.subplots(figsize=(14, 8))

    # create plot
    sns.heatmap(corr, mask=mask, annot=True, cmap=color, linewidths=1, center=0)
    plt.title(title)
    plt.tight_layout()

    return fig


def seasonal_decomposition(df, select_period, model='additive'):
    """
    Decompose a timeseries dataset into trend, seasonal, residuals, and plot it.
    :param df: Timeseries data in Pandas DataFrame format. It's optional.
    :param model: seasonal decompose model, i.e., additive
    :param period: period for seasonal decompose, i.e., 12 means monthly seasonal, 3 means quarter
    :return: plot figure seasonal decompose (matplotlib figure).
    """

    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['text.color'] = 'k'

    rcParams['figure.figsize'] = 18, 8
    decomposition = seasonal_decompose(x=df, model=model, period=select_period)
    fig = decomposition.plot()

    plt.tight_layout()

    return fig


def pca_pipeline_viz(df, xlabel: str, ylabel: str, title: str):
    """
    Plot PCA elbow method. 
    Visual tool that helps to choose the number of principal components for PCA.
    :param df: Pandas DataFrame
    :param xlabel: X label legend for plot xaxis.
    :param ylabel: Y label legend for plot yaxis.
    :param title: Legend for plot title.
    :return: Matplotlib figure to visualize PCA elbow method.
    """

    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('dr', PCA())
    ])

    pipe.fit(df)

    var = pipe.steps[1][1].explained_variance_ratio_.cumsum()

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.plot(var, marker='o')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()

    return var, fig


def plot_dendrogram(df, xlabel: str, ylabel: str, title: str, y_top: float, y_base: float):
    """Hieriarchical clustering dendrogram figure.
    Visual tool that helps to choose the number of cluster for an unsupervised model.
    :param df: Pandas DataFrame
    :param xlabel: X label legend for plot xaxis.
    :param ylabel: Y label legend for plot yaxis.
    :param title: Legend for plot title.
    :return: Matplotlib figure to visualize Dendrogram method for clustering models.
    """

    fig, ax = plt.subplots(figsize=(15, 10))
    dend = sch.dendrogram(sch.linkage(df, method='ward'))

    ax.axhline(y=y_top, c='grey', lw=1, linestyle='dashed')
    ax.axhline(y=y_base, c='grey', lw=1, linestyle='dashed')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    plt.show()

    return fig


def plot_silhoutte_coeff(array, df, xlabel: str, ylabel: str, title: str): 
    """Silhouette coefficient helps to check the quality of our predicted clusters.
    Y_label = Cluster.
    X_label = Silhouette coefficient.
    Title = Silhouette coefficient plot.
    :param df: Pandas DataFrame
    :param xlabel: X label legend for plot xaxis.
    :param ylabel: Y label legend for plot yaxis.
    :param title: Legend for plot title.
    :return: Matplotlib figure to visualize Silhouette coefficient to measure the quality of clustering results
    """
    cluster_labels = np.unique(array)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(df, array, metric='euclidean')
    y_ax_lower, y_ax_upper = 0,0
    yticks = []
    fig = plt.subplots(figsize=(15,10))
    for i, c in enumerate (cluster_labels):
        c_silhouette_vals  = silhouette_vals[array==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i)/n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals,height=1, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    plt.show()

    return fig


### Solara
def draw_table_per_variable(df, selected_variables):
    # Estadísticas en un dict
    stat_dict = {
        "Promedio":   {var: df[var].mean().round(3) for var in selected_variables},
        "Mediana":    {var: df[var].median().round(3) for var in selected_variables},
        "STD":        {var: df[var].std().round(3) for var in selected_variables},
        "Máximo":     {var: df[var].max().round(3) for var in selected_variables},
        "Mínimo":     {var: df[var].min().round(3) for var in selected_variables},
        "Q1":         {var: df[var].quantile(0.25).round(3) for var in selected_variables},
        "Q3":         {var: df[var].quantile(0.75).round(3) for var in selected_variables},
        "% NaN":      {var: (df[var].isnull().mean() * 100).round(2) for var in selected_variables},
    }

    # Cabecera
    headers = "".join(
        f"<th style='border: 1px solid currentColor; padding: 6px; text-align:center; font-weight: bold;'>{var}</th>"
        for var in selected_variables
    )
    
    # Filas
    rows = []
    for stat_name, values in stat_dict.items():
        cells = "".join(
            f"<td style='border: 1px solid currentColor; padding: 6px; text-align:center; font-weight: bold;'>{values[var]}</td>"
            for var in selected_variables
        )
        rows.append(
            f"<tr><th style='border: 1px solid currentColor; padding: 6px; text-align:center; font-weight: bold;'>{stat_name}</th>{cells}</tr>"
        )

    html = f"""
    <div style='text-align:center; margin-bottom:10px; font-weight:bold; font-size:18px;'>
        Estadística descriptiva de indicadores económicos de México
    </div>
    <table style='margin: 0 auto; border-collapse: collapse;'>
        <thead>
            <tr>
                <th style='border: 1px solid currentColor; padding: 6px;'></th>
                {headers}
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """
    return html


def options_HeatMapCorrelation(df,column_names, dark):
    """ 
    Generate the json for the options on a echart, it creates a Heatmap on Cartesian visual type.

    Paramters
    ---------
    df: Dataframe
        Dataframe containing the data series
    colum_names: list str
        Contains the list of 

    Returns:
    dictionary
        Dictonary compatible with json for echarts
    """
    data = df.copy()
    # df = df[column_names]
    data = data[column_names]
    correlation_matrix = data.corr(numeric_only=True).round(3)
    correlation_matrix.fillna(np.nan, inplace=True)

    # Create plot data
    correlation_data = []
    
    # raw-column iteration
    for i, col1 in enumerate(correlation_matrix.columns):
        for j, col2 in enumerate(correlation_matrix.columns):
            if i != j:  # Avoid same features
                correlation_value = correlation_matrix.iloc[i, j]
                correlation_data.append([i, j, correlation_value])
    
    text_color = "#fff" if dark else "#000"
    bg_color = "#333" if dark else "#fff"
    line_color = "#aaa" if dark else "#333"

    result = {
        "backgroundColor": bg_color,
        "title": {
            "text": "Grados de correlación de Pearson",
            "textStyle": {"color": text_color},
            "left": "center",
            # "top": "2%"
            # "bottom": '2%'
            #  "textAlign": "center",
            },
        "tooltip": {
            "position": 'top'
         },
        "toolbox": {
            "feature": {
                "dataZoom": {"yAxisIndex": "all"},
                "brush": {"type": ["lineX", "clear"]},
                "saveAsImage": {"show": True, "name": f"correlacion-banxico-{','.join(column_names)}_{today_str}"}
            },
            "bottom": '5%'
        },
        "grid": {
            "height": '50%',
            "top": '10%',
            "left": '20%'
        },
        "xAxis": {
            "type": 'category',
            "data": column_names,
            "splitArea": {
                "show": True
            },
            "axisLine": {"lineStyle": {"color": text_color}},
            "axisLabel": {"color": text_color, "rotate": 30,},
        },
        "yAxis": {
            "type": 'category',
            "data": column_names,
            "splitArea": {
                "show": True
            },
            "axisLine": {"lineStyle": {"color": text_color}},
            "axisLabel": {"color": text_color, "rotate": 30,},
            # "nameRotate": 30
        },
        "visualMap": {
            "min": -1,
            "max": 1,
            "calculable": True,
            "orient": 'horizontal',
            "left": 'center',
            "bottom": '5%'
        },
        "series": [
            {
              "name": 'Heat Map',  
              "type": 'heatmap',
              "data": correlation_data,
              "label": {
                  "show": True,
              },
              "emphasis": {
                  "itemStyle": {
                      "shadowBlur": 10,
                      "shadowColor": 'rgba(128, 128, 128, 0.5)'
                  }
              } 
            }
          ]
    }

    return result

@solara.component
def HeatMapCorrelation(df,variable_names, dark): 
    selected_variables, set_selected_variables=solara.use_state(random.sample(variable_names, 6))
    
    with solara.VBox() as main:
        with solara.Card(f"Variable selection"):
            solara.SelectMultiple("Variables", selected_variables, variable_names, on_value=set_selected_variables)
            # solara.Markdown(draw_table_per_variable(df, selected_variables))
        with solara.Card("Mapa de calor"):
            solara.FigureEcharts(option=options_HeatMapCorrelation(
                df,
                selected_variables,
                dark
            ),responsive=True,
            )
            pass
    return main


def options_BoxPlot(df: pd.DataFrame, column_names: list[str], get_outliers: bool = True, dark: bool = False):
    """
    Genera una opción de ECharts para boxplot múltiple: una caja por variable.

    Parámetros:
    df            : DataFrame con las series.
    column_names  : lista de columnas a graficar.
    get_outliers  : si True, marca los outliers en el boxplot.

    Retorna:
    Dict con la configuración 'option' compatible con echarts.
    """
    # 1. Preparar datos para cada variable
    data_values = [df[col].dropna().tolist() for col in column_names]

    # 2. Función para calcular estadísticos de boxplot
    def _box_stats(arr:list[float]) -> list[float]:
        q1 = round(np.percentile(arr, 25), 2)
        median = round(np.percentile(arr, 50), 2)
        q3 = round(np.percentile(arr, 75), 2)
        iqr = q3 - q1
        low = round(np.min([x for x in arr if x >= q1 - 1.5 * iqr]), 2)
        high = round(np.max([x for x in arr if x <= q3 + 1.5 * iqr]), 2)
        return [low, q1, median, q3, high]

    # 3. Calcular datos de boxplot
    box_data = [_box_stats(vals) for vals in data_values]

    # 4. Construir option
    text_color = "#fff" if dark else "#000"
    bg_color = "#333" if dark else "#fff"
    line_color = "#aaa" if dark else "#333"
    option = {
        "backgroundColor": bg_color,
        "textStyle": {"color": text_color},
        "legend": {"textStyle": {"color": text_color}, "type": "scroll", "data": column_names, "top": 50, "left": "center"},
        "title": {"text": "Estadística descriptiva para la detección de anomalías", "left": "center", "textStyle": {"color": text_color}},
        "tooltip": {"trigger": "item", "axisPointer": {"type": "shadow"}},
        "toolbox": {
            "feature": {
                "dataZoom": {"yAxisIndex": "all"},
                "brush": {"type": ["lineX", "clear"]},
                "saveAsImage": {"show": True, "name": f"diagrama-caja-{','.join(column_names)}_{today_str}"}
            }
        },
        "xAxis": {
            "type": "category",
            "data": column_names,
            "axisLine": {"lineStyle": {"color": text_color}},
            "axisLabel": {"color": text_color, "rotate": 30},
            },
        "yAxis": {
            "type": "value",
            "name": "Valor",
            "axisLine": {"lineStyle": {"color": text_color}},
            "axisLabel": {"color": text_color},
            },
        "grid": {"left": "10%", "right": "10%", "bottom": "15%"},
        "series": [
            {
                "name": "boxplot",
                "type": "boxplot",
                "data": box_data,
                "itemStyle": {"color": "#057cfc"},
            }
        ]
    }

    # 5. Opcional: marcar outliers como scatter
    if get_outliers:
        all_outliers = []
        for i, vals in enumerate(data_values):
            q1, median, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            for x in vals:
                if x < lower_bound or x > upper_bound:
                    all_outliers.append([i, x])
        option["series"].append({
            "name": "outliers",
            "type": "scatter",
            "data": all_outliers,
            "itemStyle": {"color": "#EE6666"}
        })

    # 6. Marcador del último valor de cada variable (estrella)
    last_points = []
    for i, col in enumerate(column_names):
        # Obtener último valor no nulo
        last_val = df[col].dropna().iloc[-1].round(2)
        last_points.append([i, last_val])
    # Añadir serie tipo scatter con símbolo estrella
    option["series"].append({
        "name": f"Valor más reciente: {df.index[-1].strftime('%Y-%m-%d')}",
        "type": "scatter",
        "data": last_points,
        "symbol": "circle",
        "symbolSize": 14,
        "itemStyle": {"color": "#FFD700"},  # dorado
        "z": 10,
        "tooltip": {"show": True}
    })

    return option

@solara.component
def BoxPlot(df: pd.DataFrame, column_names: list[str], show_outliers: bool = True, dark: bool = False):
    """Componente Solara que renderiza el boxplot múltiple"""
    init_vars = tuple(df.columns)
    selected_vars, set_selected_vars = solara.use_state([v for v in init_vars if v in column_names])
    column_names.sort()
    with solara.VBox():
        with solara.Card("Variables a graficar"):
            # selected, set_selected = solara.use_state(column_names)
            solara.SelectMultiple("Columnas", selected_vars, column_names, on_value=set_selected_vars)
            # solara.Switch(label="Mostrar anomalías", value=show_outliers, on_value=set_show_outliers)
        with solara.Card("Detección de rangos críticos"):
            solara.FigureEcharts(option=options_BoxPlot(df, selected_vars, show_outliers, dark), responsive=True)
    return None


def options_TimeSeries(df, column_names, get_insights, dark):
    # Preparar eje X
    x_data = df.index.strftime("%Y-%m-%d").tolist()

    text_color = "#fff" if dark else "#000"
    bg_color = "#333" if dark else "#fff"
    line_color = "#aaa" if dark else "#333"

    # Configuración base sin barras
    option = {
        "animationDuration": 30000,
        "backgroundColor": bg_color,
        "textStyle": {"color": text_color},
        "title": {"text": "Indicadores Económicos Mexicanos a través del tiempo", "top": 10, "left": "center", "textStyle": {"color": text_color},},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}, "textStyle": {"color": text_color}, "axisPointer": {"lineStyle": {"color": line_color}}},
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "legend": {"textStyle": {"color": text_color}, "type": "scroll", "data": column_names, "top": 50, "left": "center"},
        "toolbox": {
            "feature": {
                "dataZoom": {"yAxisIndex": "all"},
                "brush": {"type": ["lineX", "clear"]},
                "saveAsImage": {"show": True, "name": f"datos-banxico-{','.join(column_names)}_{today_str}"}
            }
        },
        # "grid": [{"left": "10%", "right": "8%", "height": "80%"}],
        "grid": {"left": "10%", "right": "10%", "top": 100, "bottom": 100, "containLabel": True},
        "xAxis": [
            {
                "type": "category",
                "data": x_data,
                "boundaryGap": False,
                "splitLine": {"show": False},
                "axisLine": {"lineStyle": {"color": text_color}},
                "axisLabel": {"color": text_color},
             }
            ],
        "yAxis": [],
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0], "start": 85, "end": 100},
            {"type": "slider", "xAxisIndex": [0], "bottom": 20, "start": 85, "end": 100}
        ],
        "series": []
    }

    # Crear un eje Y y serie para cada variable
    for idx, col in enumerate(column_names):
        option["yAxis"].append({
            "type": "value",
            "scale": True,
            # "name": col,
            "position": "right" if idx % 2 else "left",
            "offset": 20 * idx,
            "splitLine": {"show": False},
            "axisLine": {"lineStyle": {"color": text_color}},
            "axisLabel": {"color": text_color},
        })
        # Datos con None en lugar de NaN
        data = df[col].round(2).where(df[col].notna(), '-').tolist()
        series_cfg = {
            "name": col,
            "type": "line",
            "smooth": True,
            "connectNulls": False,
            "yAxisIndex": idx,
            "xAxisIndex": 0,
            "data": data,
            "lineStyle": {"opacity": 0.8}
        }
        if get_insights:
            # series_cfg["markPoint"] = {"data": [{"type": "max"}, {"type": "min"}]}
            # series_cfg["markLine"] = {"data": [{"type": "average"}]}
            series_cfg["markPoint"] = {
                "symbol": "circle",          # forma de la marca
                "symbolSize": 40,            # tamaño fijo en pixeles
                "symbolOffset": [0, "0%"],   # centrar verticalmente
                "data": [
                    {"type": "max", "name": "Max"},
                    {"type": "min", "name": "Min"}
                ]
            }
        option["series"].append(series_cfg)

    return option

@solara.component
def TimeSeries(df, variable_names, get_insights, dark):
    # init_vars = ('inflacion_anual', 'MME', 'interes_interbancario_28')
    init_vars = tuple(df.columns)
    selected_vars, set_selected_vars = solara.use_state([v for v in init_vars if v in variable_names])
    variable_names.sort()

    with solara.VBox() as main:
        with solara.Card("Selecciona variables"):
            solara.SelectMultiple("Variables seleccionadas", selected_vars, variable_names, on_value=set_selected_vars)
            # solara.Markdown(draw_table_per_variable(df, selected_vars))
            html = draw_table_per_variable(df, selected_vars)
            solara.HTML(tag="div", unsafe_innerHTML=html)
        with solara.Card("Serie de tiempo"):
            solara.FigureEcharts(option=options_TimeSeries(df, selected_vars, get_insights, dark), responsive=True)
            file_object = df.to_csv()
            solara.FileDownload(file_object, "Indicadores_MX_Banxico.csv", mime_type="application/vnd.ms-excel")
    return main


def options_TimeSeriesInsights(
    df: pd.DataFrame,
    variables: list[str],
    events: dict[str, dict[str, str]] = None,
    get_insights: bool = True,
    dark: bool = None
) -> dict:
    # 1️⃣ Prepara el eje X
    x_data = df.index.strftime("%Y-%m-%d").tolist()

    # 2️⃣ Construye series para cada variable
    series = []
    colors = ["#5470C6", "#91CC75", "#EE6666", "#FAC858", "#73C0DE"]

    text_color = "#fff" if dark else "#000"
    bg_color = "#333" if dark else "#fff"
    line_color = "#aaa" if dark else "#333"

    for i, var in enumerate(variables):
        y = df[var].round(2).where(df[var].notna(), "-").tolist()
        # último valor
        last_val = df[var].dropna().iloc[-1].round(2)
        last_line = {
            "name": f"{var} – Último: {last_val}",
            "type": "line",
            "data": [last_val] * len(x_data),
            "lineStyle": {"type": "dashed", "color": colors[i % len(colors)], "width": 2},
            "symbol": "none",
            "z": 9,
        }
        # serie principal
        main_line = {
            "name": var,
            "type": "line",
            "data": y,
            "smooth": True,
            "connectNulls": False,
            "lineStyle": {"color": colors[i % len(colors)], "width": 3},
            "z": 2,
        }
        series.append(main_line)
        series.append(last_line)

    # 3️⃣ Eventos globales: áreas y líneas verticales
    mark_areas = []
    mark_lines = []
    if events:
        for name, ev in events.items():
            ds, de = ev.get("date_start"), ev.get("date_end")
            if ds and de:
                # área sombreada
                mark_areas.append([
                    {"xAxis": ds},
                    {"xAxis": de}
                ])
            if ds:
                # línea vertical
                mark_lines.append([
                    {"xAxis": ds, "yAxis": "min"},
                    {"xAxis": ds, "yAxis": "max"}
                ])
    # si hay áreas, las aplicamos en un markArea de una serie fantasma
    if mark_areas:
        series.append({
            "name": "Zonas de evento",
            "type": "line",
            "data": [None] * len(x_data),
            "symbol": "none",
            "markArea": {"silent": True, "itemStyle": {"color": "rgba(245,194,66,0.3)"}, "data": mark_areas},
            "z": 1,
        })
    # si hay líneas, marcamos con markLine en otra serie fantasma
    if mark_lines:
        series.append({
            "name": "Líneas de evento",
            "type": "line",
            "data": [None] * len(x_data),
            "symbol": "none",
            "markLine": {"silent": True, "lineStyle": {"type": "dashed", "color": "#FF4500"}, "data": mark_lines},
            "z": 10,
        })

    # 4️⃣ Armar option
    option = {
        "backgroundColor": bg_color,
        "textStyle": {"color": text_color},
        "animationDuration": 3000,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}, "textStyle": {"color": text_color}, "axisPointer": {"lineStyle": {"color": line_color}}},
        # "legend": {"data": variables, "top": 40, "left": "center"},
        "legend": {"textStyle": {"color": text_color}, "type": "scroll", "data": variables, "top": 50, "left": "center"},
        "toolbox": {"feature": {"saveAsImage": {"show": True}, "name": f"series-de-tiempo-banxico-{','.join(variables)}_{today_str}"}},
        "grid": {"left": "10%", "right": "8%", "bottom": "15%", "top": "15%"},
        "xAxis": {"type": "category", "data": x_data, "boundaryGap": False, "axisLine": {"lineStyle": {"color": text_color}}, "axisLabel": {"color": text_color}},
        "yAxis": {"type": "value", "scale": True, "axisLine": {"lineStyle": {"color": text_color}}, "axisLabel": {"color": text_color}},
        "dataZoom": [
            {"type": "inside", "start": 70, "end": 100},
            {"type": "slider", "top": "90%", "start": 70, "end": 100}
        ],
        "series": series,
    }

    # 5️⃣ Insights (markPoint/markLine) sobre las series de datos
    if get_insights:
        for s in option["series"]:
            if s["type"] == "line" and s.get("data") and any(isinstance(v, (int, float)) for v in s["data"]):
                s.setdefault("markPoint", {})["data"] = [{"type": "max"}, {"type": "min"}]
                s.setdefault("markLine", {})["data"] = [{"type": "median"}]

    return option


@solara.component
def TimeSeriesInsights(
    df: pd.DataFrame,
    variable_names: list[str],
    events: dict[str, dict[str, str]] = None,
    get_insights: bool = True
):
    # Estado de selección múltiple
    # selected, set_selected = solara.use_state(variable_names)
    default = ['Tasa de referencia Banxico (%)'] if 'Tasa de referencia Banxico (%)' in variable_names else [variable_names[0]]
    selected, set_selected = solara.use_state(default)

    with solara.VBox():
        solara.lab.ThemeToggle(enable_auto=False)
        dark = solara.lab.theme.dark_effective

        with solara.Card("Variables seleccionadas"):
            solara.SelectMultiple("Variables", selected, variable_names, on_value=set_selected)
            # solara.Markdown(draw_table_per_variable(df, selected))
            html = draw_table_per_variable(df, selected)
            solara.HTML(tag="div", unsafe_innerHTML=html)
        with solara.Card("Eventos de estudio"):
            solara.Markdown("```json\n" + json.dumps(events or {}, indent=2) + "\n```")
        with solara.Card("Indicadores económicos de Banxico"):
            opt = options_TimeSeriesInsights(df, selected, events, get_insights, dark)
            solara.FigureEcharts(option=opt, responsive=True)

    return None


def options_ParallelCoordinates(df,column_names, dark):
    """ Generate the json for the options on a echart

    Paramters
    ---------
    df: Dataframe
        Dataframe containing the data series
    colum_names: list str
        Contains the list of 

    Returns:
    dictionary
        Dictonary compatible with json for echarts
    """
    parallelAxis = []
    
    for i in range(len(column_names)):
        parallel_dict = {
            "dim": i,
            "name": column_names[i]
        }
        parallelAxis.append(parallel_dict)

    text_color = "#fff" if dark else "#000"
    bg_color = "#333" if dark else "#fff"
    line_color = "#aaa" if dark else "#333"

    option = {
        "parallelAxis": parallelAxis,
        "backgroundColor": bg_color,
        "textStyle": {"color": text_color},
        "title": {"text": "Conoce la ruta que siguen los datos en un rango determinado de zonas operativas", "left": "center", "textStyle": {"color": text_color}},
        "toolbox": {
            "feature": {
              "saveAsImage":{
                  'show':True,
                  'name': f"coordenadas-de-ruta-{','.join(column_names)}_{today_str}",
              }
            }
          },
        "xAxis": {
            "axisLabel": {
                "color": text_color,
            },
            "axisLine": {"lineStyle": {"color": text_color}},
            # "axisLabel": {},
        },
        "yAxis": {
            "axisLabel": {
                "color": text_color,
            },
            "axisLine": {"lineStyle": {"color": text_color}},
        },
        "series": {
            "type": 'parallel',
            "colorBy": "series",
            "lineStyle": {
              "width": 1,
              "opacity": 0.5,
              "type": "dashed",
              "color": "#0b62bf"
            },
            "label": {
                  "show": True,
              },
        "data": df[column_names].bfill().values.tolist()
        }
    }

    return option

@solara.component
def ParallelCoordinatesPlot(df,variable_names, dark): 
    selected_variables, set_selected_variables=solara.use_state(variable_names)
    variable_names.sort()
    
    with solara.VBox() as main:
        with solara.Card(f"Selecciona variables"):
            solara.SelectMultiple("Variables seleccionadas", selected_variables, variable_names, on_value=set_selected_variables)
            # solara.Markdown(draw_table_per_variable(df, selected_variables))
            html = draw_table_per_variable(df, selected_variables)
            solara.HTML(tag="div", unsafe_innerHTML=html)
        with solara.Card("Coordenadas paralelas"):
            solara.FigureEcharts(option=options_ParallelCoordinates(
                df,
                selected_variables,
                dark
            ), responsive=True)
    
    return main


def options_BarPib(df, dark):
    df["pib_mx"] = df["pib_mx"].fillna(0)
    # 1. Eje X
    x_data = df["year"].astype(str).tolist()
    # 2. Serie de datos: objetos con valor y color condicional
    data = []
    for v in df["pib_mx"].round(2):
        color = "#5470C6" if v >= 0 else "#AAAAAA"
        data.append({
            "value": v,
            "itemStyle": {"color": color}  # color individual :contentReference[oaicite:0]{index=0}
        })

    text_color = "#fff" if dark else "#000"
    bg_color = "#333" if dark else "#fff"
    line_color = "#aaa" if dark else "#333"

    # 3. Opción completa
    option = {
        "backgroundColor": bg_color,
        "textStyle": {"color": text_color},
        "title": {
            "text": "PIB de México a lo largo del tiempo",
            "left": "center",
            "top": 10,
            "textStyle": {"color": text_color}
        },
        "toolbox": {
            "feature": {
              "saveAsImage":{
                  'show':True,
                  'name':'PIB_MX',
              }
            }
          },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
        },
        "xAxis": {
            "type": "category",
            "data": x_data,
            "name": "Año",
            "axisLabel": {
                "color": text_color,
                "rotate": 45,
                "formatter": "{value}"  # Mostrar la fecha completa :contentReference[oaicite:1]{index=1}
            },
            "axisLine": {"lineStyle": {"color": text_color}},
            # "axisLabel": {},
        },
        "yAxis": {
            "type": "value",
            "name": "PIB MX anual (%)",
            "backgroundColor": bg_color,
            "textStyle": {"color": text_color},
        },
        "grid": {
            "left": "10%",
            "right": "10%",
            "bottom": "25%",
            "containLabel": True  # Asegura que las etiquetas no se recorten :contentReference[oaicite:2]{index=2}
        },
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0], "start": 85, "end": 100},
            {"type": "slider", "xAxisIndex": [0], "bottom": 20, "start": 85, "end": 100}
        ],
        "series": [
            {
                "name": "PIB México",
                "type": "bar",
                "label": {
                    "show": True,
                    "position": "top",
                    "fontSize": 10
                },
                "barMaxWidth": "50px",  # Ancho máximo de barra :contentReference[oaicite:3]{index=3}
                "data": data
            }
        ]
    }
    return option


@solara.component
def BarPibChart(dark):
    """
    Componente Solara que muestra un gráfico de barras del PIB de México.
    """
    df = read_sql_data(db_connection, 'pib_mx')
    df.drop("index", axis=1, inplace=True)

    with solara.Card(f"Úiltima actualización: {today_str}"):
        solara.FigureEcharts(option=options_BarPib(df, dark), responsive=True)
        file_object = df.to_csv(index=False)
        solara.FileDownload(file_object, "PIB_MX.csv", mime_type="application/vnd.ms-excel")

    return None


def options_DualTimeSeries(
    df_top: pd.DataFrame,
    series_top: list[tuple[str, str]],
    df_bottom: pd.DataFrame,
    series_bottom: list[tuple[str, str]],
    events: dict[str, dict[str, str]] = None,
    dark: bool = False
) -> dict:
    text_color = "#fff" if dark else "#000"
    bg_color   = "#333" if dark else "#fff"

    # Eje X compartido
    x_data = df_top.index.strftime("%Y-%m-%d").tolist()

    # Grid ajustado
    # grids = [
    #     {"left": "10%", "right": "8%", "top": "12%", "height": "45%"},
    #     {"left": "10%", "right": "8%", "top": "60%", "height": "30%"}
    # ]
    grids = [
        {"left": "10%", "right": "8%", "top": "12%", "height": "30%"},
        {"left": "10%", "right": "8%", "top": "55%", "height": "30%"}
    ]

    # Dos ejes X
    x_axes = [
        {"type": "category", "gridIndex": 0, "data": x_data, "boundaryGap": False,
         "axisLine": {"lineStyle": {"color": text_color}},
         "axisLabel": {"show": False}},
         # "axisLabel": {"color": text_color, "interval": len(x_data)//10}},
        {"type": "category", "gridIndex": 1, "data": x_data, "boundaryGap": False,
         "axisLine": {"lineStyle": {"color": text_color}},
         "axisLabel": {"color": text_color, "interval": len(x_data)//4}},
         # "axisLabel": {"show": False}}
    ]

    # Dos ejes Y
    y_axes = [
        {"type": "value", "gridIndex": 0, "name": series_top[0][0],
         "axisLine": {"lineStyle": {"color": text_color}},
         "axisLabel": {"color": text_color}, "splitLine": {"show": True}},
        {"type": "value", "gridIndex": 1, "name": series_bottom[0][0],
         "position": "right",  # mover panel inferior a derecha
         "axisLine": {"lineStyle": {"color": text_color}},
         "axisLabel": {"color": text_color}, "splitLine": {"show": True}}
    ]

    # Series
    series = []
    # Panel superior
    for col, clr in series_top:
        series.append({
            "name": col,
            "type": "line",
            "xAxisIndex": 0, "yAxisIndex": 0,
            "data": df_top[col].round(2).where(df_top[col].notna(), '-').tolist(),
            "lineStyle": {"color": clr, "width": 2},
            "showSymbol": False,
            # "emphasis": {"showSymbol": True, "symbol": "circle", "symbolSize": 6},
        })
    # Panel inferior
    for col, clr in series_bottom:
        series.append({
            "name": col,
            "type": "line",
            "xAxisIndex": 1, "yAxisIndex": 1,
            "data": df_bottom[col].round(2).where(df_bottom[col].notna(), '-').tolist(),
            "lineStyle": {"color": clr, "width": 2},
            "showSymbol": False,
            "emphasis": {"showSymbol": True, "symbol": "triangle", "symbolSize": 6},
        })

    # Eventos en panel superior
    if events:
        areas = [[{"xAxis": ev["date_start"][:10]}, {"xAxis": ev["date_end"][:10]}]
                 for ev in events.values() if ev.get("date_start") and ev.get("date_end")]
        if areas:
            series.append({
                "name": "Eventos",
                "type": "line", "xAxisIndex": 0, "yAxisIndex": 0,
                "data": [None]*len(x_data),
                "symbol": "none",
                "markArea": {"silent": True, "itemStyle": {"color": "rgba(255,173,177,0.3)"}, "data": areas},
                "z": 1
            })

    # Dos leyendas separadas
    legends = [
        {"orient": "horizontal", "left": "center", "top": 0,
         "data": [c for c,_ in series_top], "textStyle": {"color": text_color}},
        {"orient": "horizontal", "left": "center", "top": "48%",
         "data": [c for c,_ in series_bottom], "textStyle": {"color": text_color}}
    ]

    return {
        "backgroundColor": bg_color,
        "textStyle": {"color": text_color},
        "toolbox": {"feature": {"saveAsImage": {"show": True}, "name": f"series-de-tiempo-banxico-{today_str}"}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": legends,
        "grid": grids,
        "xAxis": x_axes,
        "yAxis": y_axes,
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0,1], "start": 0, "end": 100},
            {"type": "slider", "xAxisIndex": [0,1], "top": "90%", "start": 0, "end": 100}
        ],
        "series": series
    }

@solara.component
def DualTimeSeriesDashboard(df_macro, df_prod, events):
    # Panel superior: indicadores económicos
    top_vars = [
        ("CETES (%)", "#91CC75"),
        ("Inflación anual Banxico (%)", "#EE6666"),
        ("Tasa de referencia Banxico (%)", "#D14A61")
    ]
    # Panel inferior: producción petrolera
    name = list(df_prod.columns)
    # print(name)
    bot_vars = [
        (name[0], "#FAC858")
    ]

    with solara.VBox():
        # with solara.Card
        col_stats = list(df_macro.columns)
        html = draw_table_per_variable(df_macro, col_stats)
        solara.HTML(tag="div", unsafe_innerHTML=html)
        solara.FigureEcharts(
            option=options_DualTimeSeries(
                df_top=df_macro,
                series_top=top_vars,
                df_bottom=df_prod,
                series_bottom=bot_vars,
                events=events,
                dark=solara.lab.theme.dark_effective
            ),
            responsive=True
        )
    return None

# @solara.component
# def DataVisualizer(df: pd.DataFrame, variable_names: list[str]):
#     """
#     Componente único que agrupa diferentes tipos de gráficas (BoxPlot, HeatMapCorrelation, etc.)
#     y permite cambiar entre ellas dinámicamente.
#     """
#     import solara.lab
#     # 1️⃣ Definimos los tipos de gráfico disponibles
#     # chart_types = ["Diagrama de caja", "Correlación entre variables", "Análisis de series de tiempo", "Coordenadas paralelas"]
#     # chart_type, set_chart_type = solara.use_state(chart_types[0])

#     # 2️⃣ Estado para las columnas seleccionadas
#     #    Por default tomamos 3 al azar para el heatmap y todas para boxplot
#     # default = random.sample(variable_names, min(3, len(variable_names)))
#     # selected_columns, set_selected_columns = solara.use_state(default)
#     init_vars = tuple(df.columns)
#     selected_columns, set_selected_columns = solara.use_state([v for v in init_vars if v in variable_names])
#     # Interruptor de outliers (solo para BoxPlot)
#     show_outliers, set_show_outliers = solara.use_state(1)

#     with solara.VBox() as main:
#         # Tema claro/oscuro
#         solara.lab.ThemeToggle(enable_auto=False)
#         dark = solara.lab.theme.dark_effective

#         # Pestañas verticales con solo iconos (label="") :contentReference[oaicite:0]{index=0}
#         with solara.lab.Tabs(vertical=True, background_color="primary", slider_color="white", dark=dark):
#             with solara.lab.Tab("", icon_name="mdi-home"):
#                 BarPibChart(dark)

#             with solara.lab.Tab("", icon_name="mdi-chart-box"):  # Solo icono
#                 BoxPlot(df, selected_columns, show_outliers, set_show_outliers, dark)

#             with solara.lab.Tab("", icon_name="mdi-chart-bubble"):
#                 HeatMapCorrelation(df, selected_columns, dark)

#             with solara.lab.Tab("", icon_name="mdi-chart-line"):
#                 # data = df.describe().T.round(2)
#                 # solara.DataFrame(data, items_per_page=10)
#                 TimeSeries(df, selected_columns, True, dark)

#             with solara.lab.Tab("", icon_name="mdi-chart-gantt"):
#                 # data = df.describe().T.round(2)
#                 # solara.DataFrame(data, items_per_page=10)
#                 # solara.DataFrame(df, items_per_page=5)
#                 ParallelCoordinatesPlot(df, selected_columns, dark)

#     return main


@solara.component
def DataVisualizer(df: pd.DataFrame, variable_names: list[str]):
    """
    Componente único que agrupa diferentes tipos de gráficas (BoxPlot, HeatMapCorrelation, etc.)
    y permite cambiar entre ellas dinámicamente.
    """
    import solara.lab

    # init_vars = tuple(df.columns)
    # selected_columns, set_selected_columns = solara.use_state([v for v in init_vars if v in variable_names])
    # Interruptor de outliers (solo para BoxPlot)
    # show_outliers, set_show_outliers = solara.use_state(1)
    # show_outliers = solara.reactive(True)

    # solara.Style("""
    # /* Hacemos de la tabla un bloque centrable */
    # .table-markdown {
    # display: block !important;
    # margin: 1em auto !important;
    # border-collapse: collapse !important;
    # }
    # /* Encabezados en negrita y fondo suave */
    # .table-markdown th {
    # background-color: var(--v-theme-surface-variant) !important;
    # font-weight: bold !important;
    # text-align: center !important;
    # padding: 0.5em !important;
    # border: 1px solid var(--v-theme-on-surface) !important;
    # }
    # /* Celdas de datos: siempre centrado y negrita */
    # .table-markdown td {
    # text-align: center !important;
    # font-weight: bold !important;
    # padding: 0.5em !important;
    # border: 1px solid var(--v-theme-on-surface) !important;
    # }
    # /* Color según tema */
    # .theme--light .table-markdown,
    # .theme--light .home-markdown {
    # color: var(--v-theme-on-surface) !important;
    # }
    # .theme--dark .table-markdown,
    # .theme--dark .home-markdown {
    # color: var(--v-theme-on-primary) !important;
    # }
    # """)

    solara.Style("""
    .table-markdown {
    margin: 1em auto;                /* centra la tabla */
    border-collapse: collapse;
    display: table;                  /* para que margin:auto funcione */
    color: var(--v-theme-on-surface);/* texto y bordes en claro */
    border: 5px solid currentColor;  /* borde con el mismo color del texto */
    }
    .theme--dark .table-markdown {
    color: var(--v-theme-on-primary); /* en oscuro, texto y bordes en on-primary */
    }

    .table-markdown th,
    .table-markdown td {
    padding: 0.5em;
    text-align: center;
    font-weight: bold;
    border: 5px solid currentColor;  /* cada celda hereda currentColor */
    }

    .table-markdown th {
    background-color: var(--v-theme-surface-variant);
    }
    """)

    with solara.VBox():
        # Tema claro/oscuro
        # activar el tema claro/oscuro genera un bug en boxplot
        solara.lab.ThemeToggle(enable_auto=False)
        dark = solara.lab.theme.dark_effective

        # Pestañas verticales con solo iconos (label="") :contentReference[oaicite:0]{index=0}
        with solara.lab.Tabs(vertical=True, background_color="primary", slider_color="white", dark=dark):
            with solara.lab.Tab("", icon_name="mdi-home"):
                BarPibChart(dark)
            # revisar código, cuando se crea nacen mal o en desorden los elementos del boxplot, se corrige despues de usar el switch de anomalías
            with solara.lab.Tab("", icon_name="mdi-chart-box"):
                BoxPlot(df, variable_names, True, dark)

            with solara.lab.Tab("", icon_name="mdi-chart-bubble"):
                HeatMapCorrelation(df, variable_names, dark)

            with solara.lab.Tab("", icon_name="mdi-chart-line"):
                TimeSeries(df, variable_names, False, dark)

            with solara.lab.Tab("", icon_name="mdi-chart-gantt"):
                ParallelCoordinatesPlot(df, variable_names, dark)

    return None
