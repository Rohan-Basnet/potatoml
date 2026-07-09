from sklearn.linear_model import LogisticRegression


def train_logistic(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model

from sklearn.model_selection import train_test_split

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

from sklearn.neighbors import KNeighborsClassifier


def train_knn(X_train, y_train):

    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train)

    return model

from sklearn.svm import SVC


def train_svm(X_train, y_train):

    model = SVC(kernel="rbf", probability=True)

    model.fit(X_train, y_train)

    return model