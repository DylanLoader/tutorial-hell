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
    param_distributions = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 20 ),
        "criterion": ["gini", "entropy"],
    }
    
    model = model_selection.RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        scoring='accuracy',
        n_iter=10,
        verbose=10,
        n_jobs=10, 
        cv=5,
    )
    model.fit(X,y)
    print(f"The best model score was: {model.best_score_}")
    print(f"The model parameters used were: {model.best_estimator_.get_params()}")
    print(f"The mean runtimes for the random search runs in seconds are: {np.round(model.cv_results_['mean_fit_time'],3)}")