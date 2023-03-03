import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import decomposition
from functools import partial
import optuna

def optimize(trial, x:np.array, y:np.array):
    entropy = trial.suggest_categorical("criterion", ['gini', 'entropy'])
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    max_features = trial.suggest_float("max_features", 0.01, 1.0)
    
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion=entropy, 
        max_depth=max_depth, 
        max_features=max_features
    )
    # Have to manually do kfold validation
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        x_train = x[train_idx]
        y_train = y[train_idx]
        
        x_test = x[test_idx]
        y_test = y[test_idx]
        
        # Fit the model
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        fold_acc = metrics.accuracy_score(y_test, preds)
        accuracies.append(fold_acc)
        
    return -1.0 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    
    # Create the partial function
    optimization_function = partial(optimize, 
                                    x=X, 
                                    y=y)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(
        optimization_function,
        n_trials=15
        )