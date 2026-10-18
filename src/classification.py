import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from src.dataset import Dataset

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, stratify: bool = False, random_state=None):
    """
    Splits the dataset into train and test sets. The defaults (no stratification, no seed)
    reproduce the original experiment; stratify=True and a fixed random_state make it reproducible.
    """
    features = df.columns[df.columns != target_col]
    target = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], target, test_size=test_size,
        stratify=target if stratify else None, random_state=random_state)

    # códigos calculados com as categorias do conjunto inteiro: se uma classe faltar no
    # teste (possível sem estratificação), os códigos continuam consistentes com o treino
    categories = sorted(target.unique())
    train_set = Dataset(X=X_train, y=pd.Series(pd.Categorical(y_train, categories=categories).codes, index=y_train.index))
    test_set = Dataset(X=X_test, y=pd.Series(pd.Categorical(y_test, categories=categories).codes, index=y_test.index))

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


def evaluate_per_class(model, test_set: Dataset, class_names, features=None):
    """
    Returns a per-class report (precision, recall, F1, support), the confusion matrix
    (rows = true class) and the accuracy. `features` selects a subset of columns.
    """
    X = test_set.X if features is None else test_set.X[features]
    y_pred = model.predict(X)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(test_set.y, y_pred)
    report = pd.DataFrame({'precisão': precision, 'sensibilidade': recall, 'F1': f1, 'amostras': support},
                          index=class_names)
    cm = metrics.confusion_matrix(test_set.y, y_pred)
    return report, cm, metrics.accuracy_score(test_set.y, y_pred)


def split_variability(df: pd.DataFrame, target_col: str, n_splits: int = 1000, n_neighbors: int = 3,
                      stratify: bool = False):
    """
    Repeats the original experiment (80/20 split + StandardScaler + KNN) with seeds
    0..n_splits-1 and returns the test accuracies. Also checks that micro precision and
    micro recall are always equal to the accuracy.
    """
    accuracies, micro_equal = [], True
    for seed in range(n_splits):
        train_set, test_set = split_data(df, target_col, stratify=stratify, random_state=seed)
        acc, prc, rec = evaluate_knn(train_knn(train_set, n_neighbors), test_set)
        accuracies.append(acc)
        micro_equal &= bool(np.isclose(acc, prc) and np.isclose(acc, rec))
    return np.array(accuracies), micro_equal


def select_k(train_set: Dataset, ks, n_splits: int = 10, n_repeats: int = 10, random_state: int = 0):
    """
    Repeated stratified k-fold cross-validation on the training set for each k.
    Returns a DataFrame with the training (resubstitution) accuracy and the CV mean/std.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows = []
    for k in ks:
        scores = cross_val_score(train_knn_pipeline(k), train_set.X, train_set.y, cv=cv)
        train_acc = train_knn_pipeline(k).fit(train_set.X, train_set.y).score(train_set.X, train_set.y)
        rows.append({'k': k, 'acc treino': train_acc, 'acc CV': scores.mean(), 'dp CV': scores.std()})
    return pd.DataFrame(rows)


def train_knn_pipeline(n_neighbors: int = 3, steps=None, **knn_kwargs):
    """
    Builds (without fitting) a pipeline: optional preprocessing steps + KNN.
    The default (steps=None) is StandardScaler + KNN, as in train_knn.
    """
    if steps is None:
        steps = [('scaler', StandardScaler())]
    return Pipeline(steps=list(steps) + [('knn', KNeighborsClassifier(n_neighbors=n_neighbors, **knn_kwargs))])


def compare_configs(train_set: Dataset, configs: dict, n_splits: int = 10, n_repeats: int = 10, random_state: int = 0):
    """
    Cross-validates several configurations on the training set.
    `configs` maps a name to (list of feature columns, pipeline factory without arguments).
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows = []
    for name, (features, factory) in configs.items():
        scores = cross_val_score(factory(), train_set.X[features], train_set.y, cv=cv)
        rows.append({'configuração': name, 'acc CV': scores.mean(), 'dp CV': scores.std()})
    return pd.DataFrame(rows)


def cross_validation_errors(X: pd.DataFrame, y, factory, n_splits: int = 10, n_repeats: int = 10, random_state: int = 1):
    """
    Repeated stratified k-fold on the whole dataset. Returns the accuracies of each fold,
    the aggregated confusion matrix and how many times each sample was misclassified.
    """
    y = np.asarray(y)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    n_classes = len(np.unique(y))
    accuracies, cm, errors = [], np.zeros((n_classes, n_classes), int), np.zeros(len(y), int)
    for train_idx, test_idx in cv.split(X, y):
        y_pred = factory().fit(X.iloc[train_idx], y[train_idx]).predict(X.iloc[test_idx])
        accuracies.append(metrics.accuracy_score(y[test_idx], y_pred))
        cm += metrics.confusion_matrix(y[test_idx], y_pred, labels=list(range(n_classes)))
        errors[test_idx] += (y_pred != y[test_idx])
    return np.array(accuracies), cm, errors
