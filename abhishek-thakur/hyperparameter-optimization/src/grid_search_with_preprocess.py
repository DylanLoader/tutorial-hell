import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import decomposition

if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    # Preprocessing
    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    # Perform grid search
    rf = RandomForestClassifier(n_jobs=-1)
    
    # Pipe
    clf = pipeline.Pipeline([
        ("scaling", scl),
        ("pca", pca),
        ("rf", rf)
    ])
    param_grid = {
        "pca__n_components": [5,6,7,8,9,10], 
        "rf__n_estimators": [100, 200, 300, 400, 500],
        "rf__max_depth": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        "rf__criterion": ["gini", "entropy"],
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