import os
from matplotlib import pyplot as plt

# Define the root directory for the training set
root_dir = 'data_set/Testing'

# Initialize a dictionary to hold the count of images in each class
class_distribution = {}

# Iteracija kroz poddirektorijume i brojanje slika
for subdir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, subdir)):  # Check if it's a directory
        class_images = os.listdir(os.path.join(root_dir, subdir))
         # Filtriranje fajlova koji nisu slike (po potrebi, prilagoditi ekstenzije)
        class_images = [img for img in class_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_distribution[subdir] = len(class_images)

# class_distribution sada sadrži broj slika po klasi
# Imena klasa
classes = list(class_distribution.keys())

# Odgovarajući brojevi
counts = [class_distribution[cls] for cls in classes]

# Kreiranje bar dijagrama
plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color='skyblue')

plt.title('Distribution of Classes in Testing Set')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45) # Rotiranje imena klasa radi bolje čitljivosti ako je potrebno


plt.tight_layout()  # Adjust layout for better fit
plt.show()
