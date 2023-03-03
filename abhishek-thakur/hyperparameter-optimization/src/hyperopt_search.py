import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import decomposition

from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params, x:np.array, y:np.array):
    params = params
    model = RandomForestClassifier(**params)
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
    
    param_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "criterion": hp.choice("criterion",["gini", "entropy"]), 
        "max_features": hp.uniform("max_features",0.01, 1)
    }
    
    optimization_function = partial(
        optimize, 
        x=X, 
        y=y
    )
    
    # Initiate a trial
    trials = Trials()
    
    # Run the minimization function
    result = fmin(
        optimization_function, 
        space=param_space,
        algo=tpe.suggest,
        max_evals=15, 
        trials=trials
    )
    
    print(result)