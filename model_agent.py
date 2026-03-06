from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def run_model(df, task):

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    task = str(task).lower()

    results = {}

    # -----------------------------
    # REGRESSION
    # -----------------------------

    if "regression" in task:

        models = {

            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor()

        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = r2_score(y_test, preds)

            results[name] = round(score, 4)

    # -----------------------------
    # CLASSIFICATION
    # -----------------------------

    elif "classification" in task:

        models = {

            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier()

        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds)

            results[name] = round(score, 4)

    # Select best model
    best_model = max(results, key=results.get)

    return best_model, results