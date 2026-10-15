from ucimlrepo import fetch_ucirepo


class IRISDataset:

    def __init__(self):
        self.iris = fetch_ucirepo(id=53)
        self.X = self.iris.data.features
        self.y = self.iris.data.targets
