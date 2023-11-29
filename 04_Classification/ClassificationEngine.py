"""
Classification Engine

Installation:
Assuming that you have pip installed, type this in a terminal: sudo pip install pandas

Overview:
An Engine that uses machine learning to determine output based on provides data with labels which are stored in
corresponding .csv files. This program do it for 2 problems for a Pima Indians diabetes problem and credit card frauds
problem. For created outputs it create bar charts showing how much data was determined correctly and how much is wrong

Authors:
By Maciej Zagórski (s23575) and Łukasz Dawidowski (s22621), group 72c (10:15-11:45).

Sources:
https://pandas.pydata.org/docs/ (pandas documentation)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class Dataset:
    """
    A class to represent a dataset.
    """

    def __init__(self, name, data, ordinal=False):
        """
        Initialize a Dataset object.
        """
        self.name = name
        self.X, self.y = self.read_data(data, ordinal)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.25,
                                                                                random_state=0,
                                                                                stratify=self.y)

    def read_data(self, data, ordinal):
        """
        Read the dataset from a CSV file.
        """
        data = pd.read_csv(data)
        X = data.iloc[:, :-1]
        if ordinal:
            X = X.iloc[:, 1:]
        y = data.iloc[:, -1]
        return X, y


class Results:
    """
    A class to represent the results of a classification model.
    """

    def __init__(self, name, classifier, dataset):
        """
        Initialize a Results object.
        """
        self.name = name
        self.score, self.conf_matrix = self.classify(classifier, dataset)

    def classify(self, classifier, dataset):
        """
        Train and evaluate the classifier on the dataset.
        """
        classifier.fit(dataset.X_train, dataset.y_train)

        score = classifier.score(dataset.X_test, dataset.y_test)
        y_pred = classifier.predict(dataset.X_test)

        conf_matrix = confusion_matrix(dataset.y_test, y_pred)

        return score, conf_matrix

    def get_confusion_metrics(self):
        """
        Extract individual metrics from the confusion matrix.
        """
        tn, fp, fn, tp = self.conf_matrix.ravel()
        return tn, fp, fn, tp

    def plot_confusion_metrics_bar_chart(self, dataset_name):
        """
        Plot a bar chart of the confusion metrics.
        """
        metrics = self.get_confusion_metrics()
        total_samples = sum(metrics)
        percentages = [f'{v / total_samples * 100:.2f}%' for v in metrics]
        labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
        plt.figure()
        plt.bar(labels, metrics, color=['blue', 'red', 'orange', 'green'])

        for i, (v, percent) in enumerate(zip(metrics, percentages)):
            plt.text(i, v + 0.05, f'{v}\n({percent})', ha='center', va='bottom')

        plt.title(f'Confusion Metrics - {self.name} on {dataset_name}')
        plt.xlabel('Metrics')
        plt.ylabel('Count')

    def __str__(self):
        """
        Return a string representation of the Results object.
        """
        return f"{self.name}:\nScore: {self.score}\nConfusion matrix:\n{self.conf_matrix}\n"


if __name__ == "__main__":
    tree = DecisionTreeClassifier(max_depth=4, random_state=1, class_weight="balanced")
    svm = SVC(random_state=1, class_weight="balanced")

    diabetes = Dataset("Pima Indians Diabetes", 'pima_indians_diabetes.csv')
    credit_card_fraud = Dataset("Credit Card Fraud", 'credit_card_fraud.csv', True)

    classifiers = {"Decision Tree": tree, "SVM": svm}
    datasets = [diabetes, credit_card_fraud]

    for ds in datasets:
        print("* * * ", ds.name, "* * *\n")
        for key in classifiers:
            results = Results(key, classifiers[key], ds)
            print(results)
            results.plot_confusion_metrics_bar_chart(ds.name)

    plt.show()
