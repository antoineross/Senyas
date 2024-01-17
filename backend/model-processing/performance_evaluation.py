import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Creating a dictionary from the provided scores
# Given data
data = {
    "Ako": {"true": 25},
    "Bakit": {"true": 20, "false": {"Ako": 3, "Magandang Umaga": 1, "N": 1}},
    "Hi": {"true": 20, "false": {"Hindi": 2, "O": 3}},
    "Hindi": {"true": 15, "false": {"Hi": 2, "Magandang Umaga": 2, "P": 6}},
    "Ikaw": {"true": 21, "false": {"Ako": 4}},
    "Kamusta": {"true": 23, "false": {"Ikaw": 2}},
    "Maganda": {"true": 16, "false": {"Magandang Umaga": 4, "Hi": 3, "Ikaw": 2}},
    "Magandang Umaga": {"true": 25},
    "Salamat": {"true": 14, "false": {"Magandang Umaga": 4}},
    "Oo": {"true": 17, "false": {"N": 8}},
    "L": {"true": 25},
    "O": {"true": 25},
    "F": {"true": 25},
    "N": {"true": 21, "false": {"Oo": 4}},
    "P": {"true": 25},
    "None": {"true": 21, "false": {"Oo": 4}}, 
}

# Constructing a confusion matrix
# Extract only valid class labels (ignore non-dictionary items)
labels = [label for label in data.keys() if isinstance(data[label], dict)]
num_classes = len(labels)
confusion_matrix = pd.DataFrame(0, index=labels, columns=labels)

# Filling in the diagonal with the true positive counts
for label in labels:
    true_positive = data[label]['true'] if 'true' in data[label] else 0
    confusion_matrix.loc[label, label] = true_positive

# Filling in the false negatives using the false data from the data dictionary
for label in labels:
    false_data = data[label]['false'] if 'false' in data[label] else {}
    for false_label, count in false_data.items():
        if false_label in labels:  # Ensure false_label is a valid class
            confusion_matrix.loc[label, false_label] = count

# Plotting the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Multi-Class Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
#Initialize lists for true and predicted labels

y_true = []
y_pred = []

#Extract true positives and false negatives

for label, data_dict in data.items():
    if 'true' in data_dict:
        y_true.extend([label] * data_dict['true'])
        y_pred.extend([label] * data_dict['true'])
    if 'false' in data_dict:
        for false_label, count in data_dict['false'].items():
            y_true.extend([label] * count)
            y_pred.extend([false_label] * count)

#Calculate the confusion matrix

cm_labels = sorted(set(y_true))
conf_matrix = confusion_matrix(y_true, y_pred, labels=cm_labels)

# Calculate accuracy

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

class_report = classification_report(y_true, y_pred, target_names=cm_labels)

print("Confusion Matrix:\n", conf_matrix)
print("\nAccuracy:", accuracy)
print("\nPrecision:", precision)
print("\nRecall:", recall)
print("\nF1 Score:", f1_score)
print("\nClassification Report:\n", class_report)#

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

# Reformatting the data for confusion matrix computation and classification report
y_true = []
y_pred = []

# True positives
for label, value in data.items():
    if isinstance(value, dict) and 'true' in value:
        y_true.extend([label] * value['true'])
        y_pred.extend([label] * value['true'])

# False negatives and false positives
for label, value in data.items():
    if isinstance(value, dict) and 'false' in value:
        for false_label, count in value['false'].items():
            y_true.extend([label] * count)
            y_pred.extend([false_label] * count)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

# Create a classification report
class_report = classification_report(y_true, y_pred, target_names=labels)

conf_matrix, accuracy, precision, recall, f1_score, class_report

