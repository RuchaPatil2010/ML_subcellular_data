import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from utils import multi_class_performance
from data_retreival import get_occur_data

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


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, kernels, c_values):
  """
  Trains and evaluates SVM classifiers for different kernel types and C values.

  Args:
    X_train (DataFrame): Features for training the model.
    y_train (Series): Labels for training the model.
    X_test (DataFrame): Features for testing the model.
    y_test (Series): Labels for testing the model.
    kernels (list): List of kernel types to evaluate.
    c_values (list): List of C values (regularization parameters) to evaluate.

  Returns:
    tuple:
      - summary_results (list): Summary of performance metrics for all configurations.
      - detailed_results (list): Detailed metrics including confusion matrix and classification report.
  """
  summary_results = []
  detailed_results = []

  for kernel in kernels:
    for c in c_values:
      # Initialize and train the SVM classifier
      svm = SVC(kernel=kernel, C=c)
      svm.fit(X_train, y_train)
      y_pred = svm.predict(X_test)

      # Calculate evaluation metrics
      accuracy = accuracy_score(y_test, y_pred)
      mcc = matthews_corrcoef(y_test, y_pred)
      conf_matrix = confusion_matrix(y_test, y_pred)
      sensitivity, specificity = multi_class_performance(y_test, y_pred)

      # Append summary metrics
      summary_results.append({
        'Kernel': kernel,
        'C': c,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'MCC': mcc
      })

      # Append detailed metrics
      detailed_results.append({
        'Kernel': kernel,
        'C': c,
        'Confusion_Matrix': conf_matrix,
        'Classification_Report': classification_report(y_test, y_pred, zero_division=0),
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'MCC': mcc
      })

  return summary_results, detailed_results


def display_results(summary_results, detailed_results):
  """
  Displays the results of SVM evaluations in summary and detailed formats.

  Args:
    summary_results (list): Summary of performance metrics for all configurations.
    detailed_results (list): Detailed metrics including confusion matrix and classification report.
  """
  # Convert summary results to a DataFrame and display
  summary_df = pd.DataFrame(summary_results)
  print("Summary:")
  print(summary_df)

  # Print detailed results for each configuration
  for result in detailed_results:
    print(f'\nKernel: {result["Kernel"]}, C: {result["C"]}')
    print("Confusion Matrix:")
    print(result['Confusion_Matrix'])
    print(f'Accuracy: {result["Accuracy"]:.4f}')
    print(f'Sensitivity: {result["Sensitivity"]:.4f}')
    print(f'Specificity: {result["Specificity"]:.4f}')
    print(f'Matthews Correlation Coefficient (MCC): {result["MCC"]:.4f}')
    print("\nClassification Report:")
    print(result['Classification_Report'])
    print("\n" + "=" * 50)


def main():
  """
  Main function to train and evaluate SVM classifiers using different kernel types and C values.
  """
  # Define the kernels and C values to evaluate
  kernels = ['rbf', 'poly', 'linear']
  c_values = [1, 2, 3, 4]

  # Split the data into training and testing sets (replace with actual data loading)
  X_train, X_test, y_train, y_test = get_test_train_split()  # Replace with actual data splitting logic

  # Train and evaluate SVM classifiers
  summary_results, detailed_results = train_and_evaluate_svm(
    X_train, y_train, X_test, y_test, kernels, c_values
  )

  # Display the results
  display_results(summary_results, detailed_results)


if __name__ == "__main__":
  main()
