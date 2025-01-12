from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from data_retreival import get_occur_data
from utils import multi_class_performance
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


def train_evaluate_rf(X_train, y_train, X_test, y_test, criteria, max_features, 
                      bootstrap_values, class_weights):
  """
  Trains and evaluates Random Forest classifiers for various hyperparameter 
  combinations.

  Args:
    X_train (DataFrame): Training features.
    y_train (Series): Training labels.
    X_test (DataFrame): Test features.
    y_test (Series): Test labels.
    criteria (list): List of splitting criteria ('gini', 'entropy', etc.).
    max_features (list): List of max_features options for the Random Forest.
    bootstrap_values (list): List of bootstrap options (True/False).
    class_weights (list): List of class weight options.

  Returns:
    list: A list of dictionaries containing model parameters and evaluation 
    metrics.
  """
  results = []

  for c in criteria:
    for mf in max_features:
      for b in bootstrap_values:
        for wt in class_weights:
          # Initialize the Random Forest classifier
          rf_classifier = RandomForestClassifier(
            n_estimators=100, criterion=c, random_state=42,
            max_features=mf, bootstrap=b, class_weight=wt
          )

          # Train the model
          rf_classifier.fit(X_train, y_train)

          # Make predictions
          predictions = rf_classifier.predict(X_test)

          # Calculate metrics
          accuracy = accuracy_score(y_test, predictions)
          sensitivity, specificity = multi_class_performance(y_test, predictions)
          mcc = matthews_corrcoef(y_test, predictions)

          # Store results if accuracy meets the threshold
          if accuracy > 0.69:
            results.append({
              'n_estimators': 100,
              'Criterion': c,
              'max_features': mf,
              'bootstrap': b,
              'class_weight': wt,
              'Accuracy': accuracy,
              'Sensitivity': sensitivity,
              'Specificity': specificity,
              'MCC': mcc
            })

  return results


def cross_validate_rf(X, 
                      y, 
                      num_folds, 
                      criteria, 
                      max_features, 
                      bootstrap_values, 
                      class_weights):
  """
  Performs cross-validation for Random Forest classifiers with various 
  hyperparameter combinations.

  Args:
    X (DataFrame): Features for the dataset.
    y (Series): Labels for the dataset.
    num_folds (int): Number of folds for StratifiedKFold.
    criteria (list): List of splitting criteria ('gini', 'entropy', etc.).
    max_features (list): List of max_features options for the Random Forest.
    bootstrap_values (list): List of bootstrap options (True/False).
    class_weights (list): List of class weight options.

  Returns:
    list: A list of dictionaries containing model parameters and 
    cross-validation metrics.
  """
  stratified_kfold = StratifiedKFold(n_splits=num_folds, 
                                     shuffle=True, 
                                     random_state=42)
  results = []

  for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Evaluate Random Forest models on this fold
    fold_results = train_evaluate_rf(
      X_train,
      y_train, 
      X_test, 
      y_test, 
      criteria, 
      max_features, 
      bootstrap_values, 
      class_weights
    )
    results.extend(fold_results)

  return results


def main():
  """
  Main function to train, evaluate, and perform cross-validation for Random 
  Forest classifiers.
  """
  # Step 1: Load the dataset and split into train and independent test sets
  (X_train, 
   X_test_independent, 
   y_train, 
   y_test_independent) = get_test_train_split()

  # Step 2: Define Random Forest hyperparameters
  criteria = ["gini", "entropy", "log_loss"]
  max_features = ["sqrt", "log2", None]
  bootstrap_values = [True, False]
  class_weights = ["balanced", "balanced_subsample", None]

  # Step 3: Evaluate on the independent test set
  rf_results_independent = train_evaluate_rf(
    X_train, y_train, X_test_independent, y_test_independent,
    criteria, max_features, bootstrap_values, class_weights
  )

  # Display independent test set results
  print("Summary for independent dataset:")
  summary_df_independent = pd.DataFrame(rf_results_independent)
  print(summary_df_independent)

  # Step 4: Perform cross-validation
  num_folds = 5
  rf_results_cv = cross_validate_rf(
    X_train, y_train, num_folds,
    criteria, max_features, bootstrap_values, class_weights
  )

  # Display cross-validation results
  print("\nSummary for cross-validation dataset:")
  summary_df_cv = pd.DataFrame(rf_results_cv)
  print(summary_df_cv)


if __name__ == "__main__":
  main()
