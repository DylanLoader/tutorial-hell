import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    
    # Perform grid search
    clf = RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        "criterion": ["gini", "entropy"],
    }
    
    model = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=10,
        n_jobs=10, 
        cv=5,
    )
    model.fit(X,y)
    print(f"The best model score was: {model.best_score_}")
    print(f"The model parameters used were: {model.best_estimator_.get_params()}")