import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas.plotting import scatter_matrix
import seaborn as sns

# paleta categórica (ordem fixa) usada nos gráficos novos
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
GRAY = '#8a8984'
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

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
    plot_df = df.copy()

    # Escolhe uma paleta de cores
    colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
    plot_df['color'] = plot_df[target_col].map(colors)

    # Gera o scatter_matrix
    _ = sns.pairplot(plot_df, vars=features, hue=target_col, palette=colors, diag_kind='kde', plot_kws={'alpha': 0.5})

    plt.show()


# ---------------------------------------------------------------------------
# Gráficos da análise complementar (estilo compacto, para o relatório)
# ---------------------------------------------------------------------------

def set_report_style() -> None:
    """Compact matplotlib style used by the figures of the report."""
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
        'axes.edgecolor': '#8a8984', 'axes.axisbelow': True, 'lines.linewidth': 1.5, 'legend.frameon': False,
    })


def save_figure(fig, name: str, folder: str = 'figuras') -> None:
    """Saves the figure as PDF (for LaTeX) and PNG."""
    import os
    os.makedirs(folder, exist_ok=True)
    fig.savefig(f'{folder}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{folder}/{name}.png', bbox_inches='tight')


def plot_feature_densities(X: DataFrame, y):
    """One density plot per feature, with the three species overlaid."""
    labels = {'sepal length': 'sépala comp. (cm)', 'sepal width': 'sépala larg. (cm)',
              'petal length': 'pétala comp. (cm)', 'petal width': 'pétala larg. (cm)'}
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, f in zip(axs, X.columns):
        for c in range(3):
            sns.kdeplot(X.loc[y == c, f], ax=ax, color=COLORS[c], fill=True, alpha=0.25, linewidth=1.2,
                        label=CLASS_NAMES[c], warn_singular=False)
        ax.set_xlabel(labels.get(f, f)); ax.set_ylabel(''); ax.set_yticks([])
    axs[0].set_ylabel('densidade'); axs[-1].legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    return fig


def plot_sepal_petal(X: DataFrame, y):
    """Scatter plots of sepal and petal measurements."""
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.5))
    for ax, (fx, fy, title) in zip(axs, [('sepal length', 'sepal width', 'Sépalas'), ('petal length', 'petal width', 'Pétalas')]):
        for c in range(3):
            m = y == c
            ax.scatter(X.loc[m, fx], X.loc[m, fy], s=12, color=COLORS[c], alpha=0.8, edgecolors='white', linewidths=0.4, label=CLASS_NAMES[c])
        ax.set_xlabel('comprimento (cm)'); ax.set_ylabel('largura (cm)'); ax.set_title(title)
    axs[1].legend(loc='upper left')
    fig.tight_layout()
    return fig


def plot_pca_projections(projections: dict, y):
    """`projections` maps a title to (Z, explained variance ratio)."""
    fig, axs = plt.subplots(1, len(projections), figsize=(6.4, 2.6))
    for ax, (title, (Z, ev)) in zip(axs, projections.items()):
        for c in range(3):
            m = y == c
            ax.scatter(Z[m, 0], Z[m, 1], s=12, color=COLORS[c], alpha=0.8, edgecolors='white', linewidths=0.4, label=CLASS_NAMES[c])
        ax.set_xlabel(f'PC1 ({100 * ev.iloc[0]:.1f}%)'); ax.set_ylabel(f'PC2 ({100 * ev.iloc[1]:.1f}%)'); ax.set_title(title)
    axs[0].legend(loc='lower right', fontsize=7)
    fig.tight_layout()
    return fig


def plot_split_variability(accuracies: dict, n_test: int = 30):
    """Bar chart of the number of correct test predictions over many random splits."""
    hits = {k: np.round(v * n_test).astype(int) for k, v in accuracies.items()}
    values = np.arange(min(h.min() for h in hits.values()), n_test + 1)
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    x = np.arange(len(values)); w = 0.8 / len(hits)
    for i, (label, h) in enumerate(hits.items()):
        ax.bar(x + (i - (len(hits) - 1) / 2) * (w + 0.02), [np.mean(h == v) for v in values], w, color=COLORS[i], label=label)
    ax.set_xticks(x, [f'{v}/{n_test}' for v in values]); ax.set_xlabel('acertos no teste (k = 3)'); ax.set_ylabel('fração das divisões')
    ax.legend(loc='upper left'); ax.grid(axis='x', visible=False)
    fig.tight_layout()
    return fig


def plot_k_selection(df_k: DataFrame, best_k: int, n_evaluations: int = 100):
    """Training (resubstitution) and cross-validation accuracy as a function of k."""
    se = df_k['dp CV'] / np.sqrt(n_evaluations)   # erro-padrão da média
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.fill_between(df_k['k'], df_k['acc CV'] - se, df_k['acc CV'] + se, color=COLORS[1], alpha=0.2, linewidth=0)
    ax.plot(df_k['k'], df_k['acc treino'], color=COLORS[0], label='treino (ressubstituição)')
    ax.plot(df_k['k'], df_k['acc CV'], color=COLORS[1], label='validação cruzada 10×10')
    ax.axvline(best_k, color=GRAY, ls=':', lw=0.9)
    ax.annotate(f'melhor k = {best_k}', (best_k, df_k['acc CV'].max()), xytext=(6, -26), textcoords='offset points',
                fontsize=7, color='#52514e')
    ax.set_xlabel('k (número de vizinhos)'); ax.set_ylabel('acurácia'); ax.set_ylim(0.85, 1.005); ax.legend(loc='lower left')
    fig.tight_layout()
    return fig


def plot_confusion_and_errors(cm, X: DataFrame, y, errors, min_errors: int, n_repeats: int = 10):
    """Aggregated confusion matrix and samples misclassified at least `min_errors` times."""
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axs[0]
    ax.imshow(cm, cmap=mpl.colors.LinearSegmentedColormap.from_list('b', ['#f4f8fd', '#2a78d6', '#0d366b']))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color='white' if cm[i, j] > cm.max() / 2 else '#0b0b0b')
    ax.set_xticks(range(3), CLASS_NAMES, fontsize=7); ax.set_yticks(range(3), CLASS_NAMES, fontsize=7)
    ax.set_xlabel('previsto'); ax.set_ylabel('real'); ax.grid(False); ax.set_title(f'Matriz de confusão (CV 10×{n_repeats})')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax = axs[1]
    for c in range(3):
        m = y == c
        ax.scatter(X.loc[m, 'petal length'], X.loc[m, 'petal width'], s=12, color=COLORS[c], alpha=0.75, edgecolors='white', linewidths=0.4, label=CLASS_NAMES[c])
    m = errors >= min_errors
    ax.scatter(X.loc[m, 'petal length'], X.loc[m, 'petal width'], s=70, facecolors='none', edgecolors='#0b0b0b', linewidths=1.0,
               label=f'errada em ≥{min_errors} de {n_repeats}')
    ax.set_xlabel('comprimento da pétala (cm)'); ax.set_ylabel('largura da pétala (cm)'); ax.legend(loc='upper left', fontsize=7)
    ax.set_title('Amostras mais erradas')
    fig.tight_layout()
    return fig


def plot_decision_boundaries(Z_train, y_train, Z_test, y_test, ks, classifier_factory):
    """KNN decision regions in a 2D plane (e.g. the first two principal components) for several k."""
    gx, gy = np.meshgrid(np.linspace(Z_train[:, 0].min() - 0.7, Z_train[:, 0].max() + 0.7, 300),
                         np.linspace(Z_train[:, 1].min() - 0.7, Z_train[:, 1].max() + 0.7, 300))
    cmap_bg = mpl.colors.ListedColormap(['#dbe8f8', '#fbe0d4', '#d3f0e5'])
    fig, axs = plt.subplots(1, len(ks), figsize=(6.6, 2.4), sharey=True)
    results = []
    for ax, k in zip(axs, ks):
        model = classifier_factory(k).fit(Z_train, y_train)
        regions = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, regions, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_bg)
        ax.contour(gx, gy, regions, levels=[0.5, 1.5], colors=GRAY, linewidths=0.6)
        for c in range(3):
            m = y_train == c
            ax.scatter(Z_train[m, 0], Z_train[m, 1], s=8, color=COLORS[c], edgecolors='white', linewidths=0.3)
        acc_train, acc_test = model.score(Z_train, y_train), model.score(Z_test, y_test)
        results.append({'k': k, 'acc treino': acc_train, 'acc teste': acc_test})
        ax.set_title(f'k = {k}\n(treino {acc_train:.0%}, teste {acc_test:.0%})', fontsize=8); ax.set_xlabel('PC1'); ax.grid(False)
    axs[0].set_ylabel('PC2')
    fig.tight_layout()
    return fig, results
