from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def multi_class_performance(y_true, y_pred):
  """
  Calculates the overall sensitivity (recall) and specificity for a multi-class 
  classification problem.
  
  Args:
      y_true (array-like): True labels for the dataset.
      y_pred (array-like): Predicted labels for the dataset.
  
  Returns:
      tuple:
          - sensitivity (float): Weighted average sensitivity across all classes.
          - specificity (float): Weighted average specificity across all classes.
  """
  # Initialize accumulators for sensitivity and specificity
  sum_sensitivity = 0
  sum_specificity = 0
  num_sensitivity = 0
  num_specificity = 0

  # Iterate over each unique label in the true labels
  for l in set(y_true):
    # Calculate precision, recall, support, and weights for the current class
    prec, recall, _, wt = precision_recall_fscore_support(
        np.array(y_true) == l,  # True binary labels for the current class
        np.array(y_pred) == l,  # Predicted binary labels for the current class
        pos_label=True,         # Positive class indicator
        average=None,           # No averaging, compute for each class separately
        zero_division=0         # Handle cases where there are no positive samples
    )

    # Update sensitivity and specificity sums using weighted recall values
    sum_sensitivity += (recall[1] * wt[1])  # Recall for positive cases
    num_sensitivity += wt[1]                # Weight for positive cases
    sum_specificity += (recall[0] * wt[0])  # Recall for negative cases (specificity)
    num_specificity += wt[0]                # Weight for negative cases

  # Calculate overall sensitivity and specificity
  sensitivity = sum_sensitivity / num_sensitivity
  specificity = sum_specificity / num_specificity

  return sensitivity, specificity
