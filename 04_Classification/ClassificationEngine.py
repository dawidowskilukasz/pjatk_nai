import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/
# Unbalanced dataset (0: 500, 1: 268)

# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Unbalanced dataset

# Weights to apply


class Dataset:
    def __init__(self, name, data, ordinal=False):
        self.name = name
        self.X, self.y = self.read_data(data, ordinal)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.33,
                                                                                random_state=0,
                                                                                stratify=self.y)

    def read_data(self, data, ordinal):
        data = pd.read_csv(data)
        X = data.iloc[:, :-1]
        if ordinal:
            X = X.iloc[:, 1:]
        y = data.iloc[:, -1]
        return X, y


class Results:
    def __init__(self, name, classifier, dataset):
        self.name = name
        self.score, self.conf_matrix = self.classify(classifier, dataset)

    def classify(self, classifier, dataset):
        classifier.fit(dataset.X_train, dataset.y_train)

        score = classifier.score(dataset.X_test, dataset.y_test)
        y_pred = classifier.predict(dataset.X_test)

        conf_matrix = confusion_matrix(dataset.y_test, y_pred)

        return score, conf_matrix

    def __str__(self):
        return f"{self.name}:\nScore: {self.score}\nConfusion matrix:\n{self.conf_matrix}\n"


if __name__ == "__main__":
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    svm = SVC(random_state=1)

    diabetes = Dataset("Pima Indians Diabetes", 'pima_indians_diabetes.csv')
    credit_card_fraud = Dataset("Credit Card Fraud", 'credit_card_fraud.csv', True)

    print("* * * ", diabetes.name, "* * *\n")

    tree_application = Results("Decision Tree", tree, diabetes)
    print(tree_application)

    svm_application = Results("SVM", svm, diabetes)
    print(svm_application)

    print("* * * ", credit_card_fraud.name, "* * *\n")

    tree_application = Results("Decision Tree", tree, credit_card_fraud)
    print(tree_application)

    svm_application = Results("SVM", svm, credit_card_fraud)
    print(svm_application)
