import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas.plotting import scatter_matrix
import seaborn as sns

def plot_distributions(df: DataFrame, target_col: str) -> None:
    """
    Plot bar charts for the distributions of each feature in the dataset, grouped by the target variable.
    """

    target_values = df[target_col].unique()
    features = df.columns[df.columns != target_col]

    fig, axes = plt.subplots(nrows=len(target_values), ncols=len(features), figsize=(12, 10), sharex='col')

    for i, target in enumerate(target_values):
        axes[i, 0].set_ylabel(f'{target}', rotation=0, labelpad=50, fontsize=10, va='center')
        for j, feature in enumerate(features):
            subset = df[df[target_col] == target][feature]
            sns.kdeplot(data=subset, fill=True, alpha=0.7, ax=axes[i, j])
            axes[i, j].set_xlabel(feature)

    plt.show()



def plot_scatter_matrix(df: DataFrame, target_col: str) -> None:
    """
    Plots a scatter matrix for the Iris dataset, coloring points by species.
    """
    features = df.columns[df.columns != target_col]

    # Escolhe uma paleta de cores
    colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
    df['color'] = df[target_col].map(colors)

    # Gera o scatter_matrix
    _ = sns.pairplot(df, vars=features, hue=target_col, palette=colors, diag_kind='kde', plot_kws={'alpha': 0.5})

    plt.show()
