from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from dataclasses import dataclass


FEATURES = ['sepal length', 'sepal width', 'petal length', 'petal width']


def _iris_uci_offline():
    """
    Rebuilds the UCI version of Iris (id=53) from scikit-learn when the UCI server
    is not reachable. The UCI file differs from Fisher's paper (and from scikit-learn)
    in two samples (35th and 38th, 1-based), documented on the UCI page.
    """
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data.copy(), columns=FEATURES)
    X.iloc[34] = [4.9, 3.1, 1.5, 0.1]
    X.iloc[37] = [4.9, 3.1, 1.5, 0.1]
    names = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = pd.DataFrame({'class': names[iris.target]})
    return X, y


class IRISDataset:

    def __init__(self):
        try:
            self.iris = fetch_ucirepo(id=53)
            self.X = self.iris.data.features
            self.y = self.iris.data.targets
            self.source = 'ucimlrepo'
        except Exception:
            # sem acesso ao UCI: mesma versão, reconstruída a partir do scikit-learn
            self.iris = None
            self.X, self.y = _iris_uci_offline()
            self.source = 'scikit-learn + correções UCI'


@dataclass
class Dataset:
    X : pd.DataFrame
    y : pd.DataFrame
