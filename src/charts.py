import matplotlib.pyplot as plt
from pandas import DataFrame

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
