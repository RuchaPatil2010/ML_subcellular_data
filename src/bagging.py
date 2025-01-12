from data_retreival import get_occur_data
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np


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


def cross_validate_bagging(X, y, base_classifier, num_classifiers, num_folds=5):
  """
  Performs cross-validation for a bagging classifier using a specified base classifier.

  Args:
    X (DataFrame): Features for the dataset.
    y (Series): Labels for the dataset.
    base_classifier (Estimator): Base classifier for the bagging model.
    num_classifiers (int): Number of estimators for the bagging classifier.
    num_folds (int): Number of folds for StratifiedKFold.

  Returns:
    dict: Average metrics (accuracy, MCC, F1 score) across all folds.
  """
  # Initialize the bagging classifier
  bagging_classifier = BaggingClassifier(base_classifier, n_estimators=num_classifiers, random_state=42)

  # Initialize StratifiedKFold for cross-validation
  stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

  accuracy_cv_list = []
  mcc_cv_list = []
  f1_cv_list = []

  for train_index, test_index in stratified_kfold.split(X, y):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

    # Train the bagging classifier
    bagging_classifier.fit(X_train_cv, y_train_cv)

    # Make predictions
    y_pred_cv = bagging_classifier.predict(X_test_cv)

    # Calculate metrics
    accuracy_cv_list.append(accuracy_score(y_test_cv, y_pred_cv))
    mcc_cv_list.append(matthews_corrcoef(y_test_cv, y_pred_cv))
    f1_cv_list.append(f1_score(y_test_cv, y_pred_cv, average='weighted'))

  # Calculate average metrics across folds
  return {
    'avg_accuracy': np.mean(accuracy_cv_list),
    'avg_mcc': np.mean(mcc_cv_list),
    'avg_f1': np.mean(f1_cv_list)
  }


def evaluate_bagging_on_test(X_train, y_train, X_test, y_test, base_classifier, num_classifiers):
  """
  Trains and evaluates a bagging classifier on an independent test set.

  Args:
    X_train (DataFrame): Training features.
    y_train (Series): Training labels.
    X_test (DataFrame): Test features.
    y_test (Series): Test labels.
    base_classifier (Estimator): Base classifier for the bagging model.
    num_classifiers (int): Number of estimators for the bagging classifier.

  Returns:
    dict: Metrics (accuracy, MCC, F1 score) on the independent test set.
  """
  # Initialize and train the bagging classifier
  bagging_classifier = BaggingClassifier(base_classifier, n_estimators=num_classifiers, random_state=42)
  bagging_classifier.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred_test = bagging_classifier.predict(X_test)

  # Calculate metrics
  return {
    'accuracy': accuracy_score(y_test, y_pred_test),
    'mcc': matthews_corrcoef(y_test, y_pred_test),
    'f1_score': f1_score(y_test, y_pred_test, average='weighted')
  }


def main():
  """
  Main function to perform cross-validation and test evaluation for a bagging classifier.
  """
  # Step 1: Define the best parameters for the base SVM classifier
  best_kernel = 'linear'
  best_C = 2
  base_classifier = SVC(kernel=best_kernel, C=best_C)

  # Step 2: Load the dataset and split into train and independent test sets
  X_train, X_test_independent, y_train, y_test_independent = get_test_train_split()

  # Step 3: Perform cross-validation
  num_classifiers = 5
  cv_results = cross_validate_bagging(X_train, y_train, base_classifier, num_classifiers, num_folds=5)

  # Display cross-validation results
  print("\nCross-Validation Results:")
  print(f"Kernel: {best_kernel}, C: {best_C}")
  print(f"Average Cross-Validation Accuracy: {cv_results['avg_accuracy']:.4f}")
  print(f"Average Cross-Validation MCC: {cv_results['avg_mcc']:.4f}")
  print(f"Average Cross-Validation F1: {cv_results['avg_f1']:.4f}")
  print("-----------------------------")

  # Step 4: Evaluate on the independent test set
  test_results = evaluate_bagging_on_test(X_train, y_train, X_test_independent, y_test_independent, base_classifier, num_classifiers)

  # Display independent test set results
  print("\nIndependent Test Set Results:")
  print(f"Accuracy: {test_results['accuracy']:.4f}")
  print(f"MCC: {test_results['mcc']:.4f}")
  print(f"F1 Score: {test_results['f1_score']:.4f}")


if __name__ == "__main__":
  main()
