import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score, roc_auc_score


class Adaboost:
    def __init__(self,n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas=[]
        self.models = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape  
        w = np.ones(n_samples) / n_samples 

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)  
            model.fit(X, y, sample_weight=w)  
            predictions = model.predict(X)  

            err = np.sum(w * (predictions != y)) / np.sum(w)

            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))

            self.models.append(model) 
            self.alphas.append(alpha)  

            w *= np.exp(-alpha * y * predictions)  
            w /= np.sum(w)

    def predict(self, X):
        strong_preds = np.zeros(X.shape[0])  

        for model, alpha in zip(self.models, self.alphas):
            predictions = model.predict(X)  
            strong_preds += alpha * predictions  

        return np.sign(strong_preds).astype(int)


if __name__ == "__main__":

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    adaboost = Adaboost(n_estimators=50)
    adaboost.fit(X_train, y_train)

    predictions = adaboost.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    try:
        roc_auc = roc_auc_score(y_test, predictions)
    except ValueError:
        roc_auc = 'Undefined (requires probability scores)'

    print(f"Accuracy: {accuracy * 100}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC-AUC: {roc_auc}")