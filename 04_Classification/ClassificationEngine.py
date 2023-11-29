import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class Dataset:
    def __init__(self, name, data, ordinal=False):
        self.name = name
        self.X, self.y = self.read_data(data, ordinal)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.25,
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

    def get_confusion_metrics(self):
        tn, fp, fn, tp = self.conf_matrix.ravel()
        return tn, fp, fn, tp

    def plot_confusion_metrics_bar_chart(self, dataset_name):
        metrics = self.get_confusion_metrics()
        total_samples = sum(metrics)
        percentages = [f'{v / total_samples * 100:.2f}%' for v in metrics]
        labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
        plt.bar(labels, metrics, color=['blue', 'red', 'orange', 'green'])

        # Add numbers and percentages above the bars
        for i, (v, percent) in enumerate(zip(metrics, percentages)):
            plt.text(i, v + 0.05, f'{v}\n({percent})', ha='center', va='bottom')

        plt.title(f'Confusion Metrics - {self.name} on {dataset_name}')
        plt.xlabel('Metrics')
        plt.ylabel('Count')

    def __str__(self):
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
