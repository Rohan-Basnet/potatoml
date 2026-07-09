from sklearn.metrics import accuracy_score


def evaluate(model, X_test, y_test):

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    return accuracy