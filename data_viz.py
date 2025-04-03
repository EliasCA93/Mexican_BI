import numpy as np

from pylab import rcParams
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import scipy.cluster.hierarchy as sch


def highlight_cells(val):
    """
    Conditonal formatting based on correlation values.
    Args:
        - val: values to compare.
    Returns:
        - format color
    """
    color = 'yellow' if (val >= 0.6 or val <= -0.6) else '#C6E2E9' # Pastel blue

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
