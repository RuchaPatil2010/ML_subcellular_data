from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, f1_score
from utils.data_retreival import get_occur_data
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
  return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def cross_validate_svm(X, y, kernels, c_values, num_folds=5):
  """
  Performs cross-validation to evaluate SVM classifiers with different kernel types and C values.

  Args:
    X (DataFrame): Features for the dataset.
    y (Series): Labels for the dataset.
    kernels (list): List of kernel types to evaluate.
    c_values (list): List of C values to evaluate.
    num_folds (int): Number of folds for StratifiedKFold.

  Returns:
    tuple:
      - best_model (SVC): Best performing SVM model based on cross-validation accuracy.
      - best_accuracy (float): Best cross-validation accuracy.
      - results (list): Summary of cross-validation metrics for all configurations.
  """
  stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
  results = []
  best_model = None
  best_accuracy = 0

  for kernel in kernels:
    for c in c_values:
      # Initialize the SVM classifier
      svm = SVC(kernel=kernel, C=c)

      # Cross-validation metrics
      accuracy_cv_list = []
      mcc_cv_list = []
      f1_cv_list = []

      # Perform cross-validation
      for train_index, test_index in stratified_kfold.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

        svm.fit(X_train_cv, y_train_cv)
        y_pred_cv = svm.predict(X_test_cv)

        # Calculate metrics
        accuracy_cv = np.mean(y_pred_cv == y_test_cv)
        mcc_cv = matthews_corrcoef(y_test_cv, y_pred_cv)
        f1_cv = f1_score(y_test_cv, y_pred_cv, average='weighted')

        accuracy_cv_list.append(accuracy_cv)
        mcc_cv_list.append(mcc_cv)
        f1_cv_list.append(f1_cv)

      # Calculate average metrics across folds
      avg_accuracy_cv = np.mean(accuracy_cv_list)
      avg_mcc_cv = np.mean(mcc_cv_list)
      avg_f1_cv = np.mean(f1_cv_list)

      # Track the best model based on accuracy
      if avg_accuracy_cv > best_accuracy:
        best_accuracy = avg_accuracy_cv
        best_model = svm

      results.append({
        'Kernel': kernel,
        'C': c,
        'Accuracy_CV': avg_accuracy_cv,
        'MCC_CV': avg_mcc_cv,
        'F1_CV': avg_f1_cv
      })

  return best_model, best_accuracy, results


def evaluate_model_on_test(model, X_test, y_test):
  """
  Evaluates the given model on an independent test set.

  Args:
    model (SVC): Trained SVM model.
    X_test (DataFrame): Features for the test set.
    y_test (Series): Labels for the test set.

  Returns:
    dict: Metrics including accuracy, MCC, and F1 score on the test set.
  """
  y_pred_test = model.predict(X_test)
  accuracy_test = np.mean(y_pred_test == y_test)
  mcc_test = matthews_corrcoef(y_test, y_pred_test)
  f1_test = f1_score(y_test, y_pred_test, average='weighted')
  return {
    'accuracy': accuracy_test,
    'mcc': mcc_test,
    'f1_score': f1_test
  }


def main():
  """
  Main function to perform cross-validation and test evaluation of SVM classifiers.
  """
  # Step 1: Split the data
  X_train, X_test_independent, y_train, y_test_independent = get_test_train_split()

  # Step 2: Define kernels and C values
  kernels = ['rbf', 'poly', 'linear']
  c_values = [1, 2, 3, 4]

  # Step 3: Perform cross-validation
  best_model, best_accuracy, cv_results = cross_validate_svm(
    X_train, y_train, kernels, c_values, num_folds=5
  )

  # Step 4: Evaluate the best model on the independent test set
  test_metrics = evaluate_model_on_test(best_model, X_test_independent, y_test_independent)

  # Step 5: Display results
  print("\nCross-Validation Results:")
  cv_results_df = pd.DataFrame(cv_results)
  print(cv_results_df)

  print("\nBest Model Parameters:")
  print(f"Kernel: {best_model.kernel}, C: {best_model.C}")
  print(f"Best Cross-Validation Accuracy: {best_accuracy:.4f}")

  print("\nIndependent Test Set Results:")
  print(f"Accuracy: {test_metrics['accuracy']:.4f}")
  print(f"MCC: {test_metrics['mcc']:.4f}")
  print(f"F1 Score: {test_metrics['f1_score']:.4f}")


if __name__ == "__main__":
  main()
