"""
Classification Engine

Installation:
Assuming that you have pip installed, type this in a terminal: sudo pip install pandas (with regard to data
structures and data analysis functions used in the code), sudo pip install sklearn (with regard to classification
algorithms (classifiers) and related functions), and sudo pip install matplotlib (with regard to visualisation of
the program results).

Overview:
The Classification Engine that uses machine learning algorithms (classifiers) – decision tree and support vector
machines (SVM) – to determine output (0 or 1) based on the provided data (sets of features) stored in corresponding
.csv files. This program has been tested and used in respect of the following problems (datasets):

* the Pima Indians diabetes problem – a dataset consisting of sets of medical data and information (output) whether a
  specific person has or does not have diabetes,

* credit card frauds problem – a dataset consisting of sets of features of credit card transactions and information
  (output) whether a specific transaction should be or should not be considered as a fraud.

The program returns results (metrics) with regard to each of the classifiers, i.e. information, how many the outputs
were predicted by the alogirthms correctly and how many incorrectly. Additionally, the visualisation of the confusion
matrix, using a bar chart, is displayed.

Authors:
By Maciej Zagórski (s23575) and Łukasz Dawidowski (s22621), group 72c (10:15-11:45).

Datasets:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database (Pima Indians Diabetes dataset)
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (Credit Card Fraud dataset)

Sources:
https://pandas.pydata.org/docs/index.html (pandas documentation)
https://scikit-learn.org/stable/index.html (scikit-learn documentation)
https://matplotlib.org/stable/ (matplotlib documentation)
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

"""
Constant to represent the ratio of splitting the dataset into training and test datasets.
"""
T_SIZE = 0.33
# T_SIZE = 0.25


class Dataset:
    """
    A class to represent a dataset.
    """

    def __init__(self, name, data, ordinal=False):
        """
        Initialize a Dataset object, spliting the data into training set (used to train the classifier) and test set
        (used to verify how accurate the classyfier is).
        """
        self.name = name
        self.X, self.y = self.read_data(data, ordinal)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=T_SIZE,
                                                                                random_state=0,
                                                                                stratify=self.y)
        # stratify parameter allows to preserve the proportion between outputs (“0” and “1”) in the training set and
        # the test set.

    def read_data(self, data, ordinal):
        """
        Read the dataset from a CSV file and separate the data to “X” (sets of features) and “y” (outputs), taking into
        account whether the set consist or not ordinal column.
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
        self.score, self.conf_matrix, self.class_rep = self.classify(classifier, dataset)

    def classify(self, classifier, dataset):
        """
        Train and evaluate the classifier on the dataset.
        """
        classifier.fit(dataset.X_train, dataset.y_train)

        score = classifier.score(dataset.X_test, dataset.y_test)

        y_pred = classifier.predict(dataset.X_test)
        conf_matrix = confusion_matrix(dataset.y_test, y_pred)

        class_names = ['Class-0', 'Class-1']
        class_rep = classification_report(dataset.y_test, y_pred, target_names=class_names)

        return score, conf_matrix, class_rep

    def get_confusion_matrix(self):
        """
        Extract individual metrics from the confusion matrix.
        """
        tn, fp, fn, tp = self.conf_matrix.ravel()
        return tn, fp, fn, tp

    def plot_confusion_matrix_bar_chart(self, dataset_name):
        """
        Plot a bar chart of the confusion matrix.
        """
        matrix = self.get_confusion_matrix()
        total_samples = sum(matrix)
        percentages = [f'{v / total_samples * 100:.2f}%' for v in matrix]
        labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
        plt.figure()
        plt.bar(labels, matrix, color=['blue', 'red', 'orange', 'green'])

        for i, (v, percent) in enumerate(zip(matrix, percentages)):
            plt.text(i, v + 0.05, f'{v}\n({percent})', ha='center', va='bottom')

        plt.title(f'Confusion Matrix - {self.name} on {dataset_name}')
        plt.xlabel('Values')
        plt.ylabel('Count')

    def __str__(self):
        """
        Return a string representation of the Results object (results of running the program – applying the classifiers.
        """
        return f"{self.name}:\nScore: {self.score}\nConfusion matrix:\n{self.conf_matrix}\n" \
               f"Classification report:\n{self.class_rep}"


if __name__ == "__main__":
    tree = DecisionTreeClassifier(max_depth=4, random_state=1, class_weight="balanced")
    # class_weight="balanced" parameter has been used because of the unbalanced character of the dataset – more “0”
    # outputs than “1” outputs.
    svm = SVC(random_state=1)
    # svm = SVC(random_state=1, class_weight="balanced")
    # class_weight="balanced" parameter was omitted because of the long runtime of the program; however, this dataset
    # is unbalanced as well.

    diabetes = Dataset("Pima Indians Diabetes", 'pima_indians_diabetes.csv')
    credit_card_fraud = Dataset("Credit Card Fraud", 'credit_card_fraud.csv', True)

    classifiers = {"Decision Tree": tree, "SVM": svm}
    datasets = [diabetes, credit_card_fraud]

    for ds in datasets:
        print("* * * ", ds.name, "* * *\n")
        for key in classifiers:
            results = Results(key, classifiers[key], ds)
            print(results)
            results.plot_confusion_matrix_bar_chart(ds.name)

    plt.show()
