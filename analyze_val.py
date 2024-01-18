from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import pyplot as plt

# Define the root directory for the training set
root_dir = 'data_set/Testing'

# Initialize a dictionary to hold the count of images in each class
class_distribution = {}

# Walk through the subdirectories and count images
for subdir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, subdir)):  # Check if it's a directory
        class_images = os.listdir(os.path.join(root_dir, subdir))
        # Filter out files that are not images (if necessary, adjust the extensions)
        class_images = [img for img in class_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_distribution[subdir] = len(class_images)

# class_distribution now holds the number of images per class
# Names of classes
classes = list(class_distribution.keys())

# Corresponding counts
counts = [class_distribution[cls] for cls in classes]

val_predictions = np.loadtxt('validation_predictions.csv', delimiter=',')

# Calculate the predicted class index for each prediction
val_predicted_labels = np.argmax(val_predictions, axis=1)


y_val = np.loadtxt("validation_actual.csv", delimiter=",")

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_val, val_predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Print the classification report
report = classification_report(y_val, val_predicted_labels, target_names=classes)
print(report)

# Save the report to a file
with open("classification_report.txt", "w") as text_file:
    print(report, file=text_file)
