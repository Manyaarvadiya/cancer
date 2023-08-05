import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def get_user_input(feature_names):
    # Function to get user input for feature values
    user_input = []
    print("Please enter the values for the following features:")
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        user_input.append(value)
    return np.array(user_input).reshape(1, -1)

def plot_feature_importance(feature_names, coef):
    # Function to plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coef, y=feature_names)
    plt.title("Feature Importance")
    plt.xlabel("Coefficient Magnitude")
    plt.ylabel("Features")
    plt.show()

def plot_roc_curve(y_test, y_prob):
    # Function to plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the logistic regression model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Get user input for features
    user_input = get_user_input(feature_names)

    # Predict the user input
    user_prediction = clf.predict(user_input)

    # Display the prediction
    if user_prediction[0] == 0:
        print("Prediction: Malignant")
    else:
        print("Prediction: Benign")

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['malignant', 'benign']))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Hyperparameter tuning using GridSearchCV
    param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Cross-validation using cross_val_score
    cross_val_scores = cross_val_score(LogisticRegression(**best_params), X, y, cv=5)
    print("Cross-validation Scores:", cross_val_scores)

    # Feature importance
    plot_feature_importance(feature_names, clf.coef_[0])

    # ROC Curve
    y_prob = clf.predict_proba(X_test)
    plot_roc_curve(y_test, y_prob)

if __name__ == "__main__":
    main()
