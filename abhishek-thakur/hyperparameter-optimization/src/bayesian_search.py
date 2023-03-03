import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import decomposition
from functools import partial
from skopt import space
from skopt import gp_minimize

def optimize(params, param_names, x:np.array, y:np.array):
    params = dict(zip(param_names, params))
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
    
    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 600, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="Criterion"), 
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]
    
    param_names = [
        "max_depth",
        "n_estimators", 
        "criterion", 
        "max_features"
    ]
    
    optimization_function = partial(
        optimize, 
        param_names=param_names, 
        x=X, 
        y=y
    )
    
    # Run the minimization function
    result = gp_minimize(
        optimization_function, 
        dimensions=param_space,
        n_calls=15, 
        random_state=10, 
        verbose=10
    )
    
    print(
        dict(zip(param_names, 
             result.x))
    )