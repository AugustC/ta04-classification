import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from src.dataset import Dataset

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    features = df.columns[df.columns != target_col]
    target = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(df[features], target, test_size=test_size)

    train_set = Dataset(X=X_train, y=y_train.astype('category').cat.codes)
    test_set = Dataset(X=X_test, y=y_test.astype('category').cat.codes)

    return train_set, test_set

def train_knn(train_set: Dataset, n_neighbors: int = 3):
    knn_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    knn_pipeline.fit(train_set.X, train_set.y)

    return knn_pipeline


def evaluate_knn(knn : Pipeline, test_set: Dataset):

    y_pred = knn.predict(test_set.X)
    accuracy = metrics.accuracy_score(test_set.y, y_pred)
    precision = metrics.precision_score(test_set.y, y_pred, average='micro')
    recall = metrics.recall_score(test_set.y, y_pred, average='micro')

    return accuracy, precision, recall
