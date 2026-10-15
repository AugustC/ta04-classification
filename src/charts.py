import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas.plotting import scatter_matrix

def plot_distributions(df: DataFrame, target_col: str) -> None:
    """
    Plot bar charts for the distributions of each feature in the dataset, grouped by the target variable.
    """

    target_values = df[target_col].unique()
    features = df.columns[df.columns != target_col]

    fig, axes = plt.subplots(nrows=len(target_values), ncols=len(features), figsize=(12, 10), sharex='col')

    fig.suptitle('Feature Distributions by Target Variable', fontsize=12)
    for i, target in enumerate(target_values):
        axes[i, 0].set_ylabel(f'{target}', rotation=0, labelpad=50, fontsize=10, va='center')
        for j, feature in enumerate(features):
            axes[i, j].hist(df[df[target_col] == target][feature], bins=15, alpha=0.7)
            axes[i, j].set_xlabel(feature)

    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    axes = scatter_matrix(df[features],
                   figsize=(8, 8),
                   diagonal='hist',
                   color=df['color'],
                   alpha=0.5)

    # Reduz o tamanho da fonte dos rótulos dos eixos
    for ax in axes.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)

    legend_labels = [plt.Line2D([0], [0], marker='o', color='w', label=key, markersize=8, markerfacecolor=value) for key, value in colors.items()]

    plt.legend(handles=legend_labels, title='Species', bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.suptitle("Scatter Matrix of Iris Dataset", fontsize=16)
    plt.show()
