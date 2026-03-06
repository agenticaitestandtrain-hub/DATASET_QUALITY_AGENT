from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def run_model(df, task):

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task == "classification":

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        return "RandomForestClassifier", score


    elif task == "regression":

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        return "RandomForestRegressor", score