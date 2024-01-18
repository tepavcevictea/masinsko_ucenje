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

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color='skyblue')

plt.title('Distribution of Classes in Training Set')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)  # Rotate class names for better readability if necessary

# Show the plot
plt.tight_layout()  # Adjust layout for better fit
plt.show()
