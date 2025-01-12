from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.preprocessing import LabelEncoder
from utils.data_retreival import get_occur_data
import numpy as np
import pandas as pd


def get_test_train_split():
  """
  Splits the data into training and independent test sets after encoding labels.

  Returns:
    tuple: A tuple containing:
        - X_train (DataFrame): Features for the training set.
        - X_test (DataFrame): Features for the test set.
        - y_train (Series): Encoded labels for the training set.
        - y_test (Series): Encoded labels for the test set.
  """
  # Load the dataset
  occur_data_df = get_occur_data().copy()

  # Encode the 'Fold' column using LabelEncoder
  label_encoder_class = LabelEncoder()
  occur_data_df['Encoded_Fold'] = label_encoder_class.fit_transform(occur_data_df['Fold'])

  # Separate features and target variable
  X = occur_data_df.drop('Fold', axis=1)
  y = occur_data_df['Encoded_Fold']

  # Split the data into training and test sets with stratification
  return train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )


def perform_grid_search(X_train, y_train):
  """
  Performs a grid search to tune hyperparameters for the KNN classifier.

  Args:
    X_train (DataFrame): Features for the training set.
    y_train (Series): Labels for the training set.

  Returns:
    tuple:
      - best_model (KNeighborsClassifier): The best KNN model found by grid search.
      - best_params (dict): The best hyperparameters found.
      - best_score (float): The best cross-validation accuracy.
  """
  # Define the parameter grid for number of neighbors
  param_grid = {'n_neighbors': range(1, 30)}

  # Initialize KNN classifier
  knn = KNeighborsClassifier(p=2)  # Using Euclidean distance

  # Initialize StratifiedKFold for cross-validation
  k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

  # Perform grid search with cross-validation
  grid_search = GridSearchCV(knn, param_grid, cv=k_folds, scoring='accuracy')
  grid_search.fit(X_train, y_train)

  # Extract the best model, parameters, and score
  best_model = grid_search.best_estimator_
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_

  return best_model, best_params, best_score


def evaluate_model(model, X_test, y_test):
  """
  Evaluates the performance of the model on the test set.

  Args:
    model (KNeighborsClassifier): The trained KNN model.
    X_test (DataFrame): Features for the test set.
    y_test (Series): Labels for the test set.

  Returns:
    dict: A dictionary containing evaluation metrics (accuracy, MCC, F1 score).
  """
  # Predict on the test set
  y_test_pred = model.predict(X_test)

  # Calculate evaluation metrics
  test_accuracy = accuracy_score(y_test, y_test_pred)
  mcc = matthews_corrcoef(y_test, y_test_pred)
  f1 = f1_score(y_test, y_test_pred, average='weighted')

  # Return the metrics as a dictionary
  return {
    'accuracy': test_accuracy,
    'mcc': mcc,
    'f1_score': f1
  }


def main():
  """
  Main function to perform data preprocessing, model training, hyperparameter tuning,
  and evaluation for the KNN classifier.
  """
  # Step 1: Split the data into training and test sets
  X_train, X_test, y_train, y_test = get_test_train_split()

  # Step 2: Perform grid search to find the best KNN model
  best_model, best_params, best_score = perform_grid_search(X_train, y_train)

  # Step 3: Print grid search results
  print("Grid Search Results:")
  print(f"Best Parameters: {best_params}")
  print(f"Best Cross-Validation Accuracy: {best_score:.4f}")

  # Step 4: Evaluate the best model on the independent test set
  evaluation_metrics = evaluate_model(best_model, X_test, y_test)

  # Step 5: Print evaluation metrics
  print("\nEvaluation on Test Set:")
  print(f"Accuracy: {evaluation_metrics['accuracy']:.4f}")
  print(f"MCC: {evaluation_metrics['mcc']:.4f}")
  print(f"F1 Score: {evaluation_metrics['f1_score']:.4f}")


if __name__ == "__main__":
  main()
