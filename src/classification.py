import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def pca_decomposition(df: pd.DataFrame, target_col: str, n_components: int = 3):
    """
    Plot a PCA decomposition of the dataset, reducing it to n_components (default 3) dimensions.
    """
    target_values = df[target_col].unique()
    features = df.columns[df.columns != target_col]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d' if n_components==3 else None)

    X_reduced = PCA(n_components=n_components).fit_transform(df[features])
    if n_components == 3:
        scatter = ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            X_reduced[:, 2],
            c=df[target_col].astype('category').cat.codes,
        )
    elif n_components == 2:
        scatter = ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=df[target_col].astype('category').cat.codes,
        )
    else:
        raise ValueError("n_components must be either 2 or 3.")


    ax.set(
        title="Principal components",
        xlabel="1st Principal Component",
        ylabel="2nd Principal Component",
    )
    if n_components == 3:
        ax.set(zlabel="3rd Principal Component")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    if n_components == 3:
        ax.zaxis.set_ticklabels([])

    # Add a legend
    legend1 = ax.legend(
        scatter.legend_elements()[0],
        target_values.tolist(),
        loc="upper right",
        title="Classes",
    )
    ax.add_artist(legend1)

    plt.show()
