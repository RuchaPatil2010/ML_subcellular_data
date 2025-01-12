from data_retreival import get_occur_data
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from utils import multi_class_performance
import numpy as np
import pandas as pd

def get_test_train_split():
    """
    Splits the data into training and independent test sets.

    Returns:
        tuple: A tuple containing:
            - X_train (DataFrame): Features for the training set.
            - X_test (DataFrame): Features for the test set.
            - y_train (Series): Labels for the training set.
            - y_test (Series): Labels for the test set.
    """
    occur_data_df = get_occur_data().copy()
    X = occur_data_df.drop('Fold', axis=1)
    y = occur_data_df['Fold']

    # Split the data into training and an independent test set
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def iterate_k_folds(X_train, y_train, kf, k):
    """
    Performs k-fold cross-validation for a given value of k in KNN.
    
    Args:
        X_train (DataFrame): Features for training.
        y_train (Series): Labels for training.
        kf (StratifiedKFold): k-fold cross-validator instance.
        k (int): Number of neighbors for the KNN classifier.
    
    Returns:
        list: List of accuracy scores for each fold during cross-validation.
    """
    cv_accuracies = []
    for train_index, test_index in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Initialize KNN classifier with the current k value
        knn_cv = KNeighborsClassifier(n_neighbors=k)

        # Train the model on the training fold
        knn_cv.fit(X_train_fold, y_train_fold)

        # Make predictions on the validation fold
        y_val_pred = knn_cv.predict(X_val_fold)

        # Evaluate the model performance on the validation fold
        cv_accuracy = accuracy_score(y_val_fold, y_val_pred)
        cv_accuracies.append(cv_accuracy)

    return cv_accuracies

def get_results(y_test, y_test_pred, cv_accuracies, k):
    """
    Calculates and returns the performance metrics for a given k in KNN.
    
    Args:
        y_test (Series): Actual labels for the test set.
        y_test_pred (ndarray): Predicted labels for the test set.
        cv_accuracies (list): List of cross-validation accuracies.
        k (int): Number of neighbors in the KNN classifier.
    
    Returns:
        tuple: 
            - result (dict): Summary of metrics including accuracy, MCC, sensitivity, and specificity.
            - detailed_result (dict): Detailed metrics including CV accuracies and confusion matrix.
    """
    avg_cv_accuracy = np.mean(cv_accuracies)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    sensitivity, specificity = multi_class_performance(y_test, y_test_pred)

    result = {
        'K': k,
        'Accuracy_CV': avg_cv_accuracy,
        'Acc_Ind': test_accuracy,
        'MCC_Ind': mcc,
        'Sensitivity_Ind': sensitivity,
        'Specificity_Ind': specificity
    }
    detailed_result = {
        'K': k,
        'CV_Accuracies': cv_accuracies,
        'Accuracy': test_accuracy,
        'Test_Confusion_Matrix': conf_matrix,
        'MCC': mcc,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }

    return result, detailed_result

def knn(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates the KNN model for different values of k using cross-validation and test set.
    
    Args:
        X_train (DataFrame): Features for the training set.
        y_train (Series): Labels for the training set.
        X_test (DataFrame): Features for the test set.
        y_test (Series): Labels for the test set.
    
    Returns:
        tuple:
            - results (list): Summary of metrics for each value of k.
            - detailed_results (list): Detailed metrics including confusion matrix and CV accuracies.
    """
    k_folds = 5
    max_k = 16
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []
    detailed_results = []

    for k in range(1, max_k):
        # Perform cross-validation
        cv_accuracies = iterate_k_folds(X_train, y_train, kf, k)

        # Train the model on the entire training set and test on the independent test set
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_test_pred = knn.predict(X_test)

        # Get the results
        result, detailed_result = get_results(y_test, y_test_pred, cv_accuracies, k)
        results.append(result)
        detailed_results.append(detailed_result)

    return results, detailed_results

def print_results(results, detailed_results):
    """
    Prints the summary and detailed results of KNN for different values of k.
    
    Args:
        results (list): Summary of metrics for each value of k.
        detailed_results (list): Detailed metrics including confusion matrix and CV accuracies.
    """
    # Display summary results in a DataFrame
    summary_df = pd.DataFrame(results)
    print("\n\nSummary:")
    print(summary_df)

    # Display details for each iteration
    print("\nDetails for Each Iteration:")
    for result in detailed_results:
        print(f"\nK = {result['K']}:")
        print("Cross Validation Accuracies:")
        print(result['CV_Accuracies'])
        print("Test Accuracy (Independent): ", result['Accuracy'])
        print("\nTest Confusion Matrix (Independent):")
        print(result['Test_Confusion_Matrix'])
        print("\nTest MCC (Independent): ", result['MCC'])
        print("Test Sensitivity (Independent): ", result['Sensitivity'])
        print("Test Specificity (Independent): ", result['Specificity'])
        print("\n" + "=" * 50)  # Separation line

def main():
    """
    Main function to divide the data into train and test sets,
    train the KNN model, and display the results.
    """
    X_train, X_test, y_train, y_test = get_test_train_split()
    results, detailed_results = knn(X_train, y_train, X_test, y_test)
    print_results(results, detailed_results)

if __name__ == "__main__":
  main()
