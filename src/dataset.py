from ucimlrepo import fetch_ucirepo
import pandas as pd
from dataclasses import dataclass


class IRISDataset:

    def __init__(self):
        self.iris = fetch_ucirepo(id=53)
        self.X = self.iris.data.features
        self.y = self.iris.data.targets


@dataclass
class Dataset:
    X : pd.DataFrame
    y : pd.DataFrame
