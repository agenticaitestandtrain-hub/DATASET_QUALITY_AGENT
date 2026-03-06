from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score


def run_model(df, task):

    if df.shape[1] < 2:
        return "Dataset too small for ML", "-"

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task and "classification" in task.lower():

        models = {
            "RandomForestClassifier": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000)
        }

        best_model = None
        best_score = 0

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds)

            if score > best_score:
                best_score = score
                best_model = name

        return best_model, round(best_score, 4)


    elif task and "regression" in task.lower():

        models = {
            "RandomForestRegressor": RandomForestRegressor(),
            "LinearRegression": LinearRegression()
        }

        best_model = None
        best_score = -999

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = r2_score(y_test, preds)

            if score > best_score:
                best_score = score
                best_model = name

        return best_model, round(best_score, 4)

    else:
        return "Unsupported ML Task", "-"