import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def generate_models():
    logistic_regression = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ]
    )

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]
    )

    decision_tree = Pipeline(
        [
            ("model", DecisionTreeClassifier())
        ]
    )

    random_forest = Pipeline(
        [
            ("model", RandomForestClassifier())
        ]
    )

    return [logistic_regression, svm, decision_tree, random_forest]

def best_model(classifiers, X_train, y_train):
    scores = []
    for model in classifiers:
        scores.append(cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean())
    
    selected_model = classifiers[np.array(scores).argmax()]
    return selected_model

def modeling(X_train, y_train):
    classifiers = generate_models()
    selected_model = best_model(classifiers, X_train, y_train)
    selected_model.fit(X_train, y_train)
    
    return selected_model