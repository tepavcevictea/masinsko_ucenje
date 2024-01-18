import os
from PIL import Image
import matplotlib.pyplot as plt

# Define the root directory
root_dir = 'data_set'

# List to hold the sizes (width, height)
image_sizes = []

# Walk through all subdirectories and files
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Modify or add other image extensions you need
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    image_sizes.append(img.size)  # (width, height)
            except Exception as e:
                print(f"Error opening image {filepath}: {e}")

# Separate widths and heights
widths, heights = zip(*image_sizes)

# # Plotting the widths
# plt.figure(figsize=(10, 6))
# plt.hist(widths, bins=30, color='skyblue')
# plt.title('Distribution of Image Widths')
# plt.xlabel('Width')
# plt.ylabel('Frequency')
# plt.show()

# # Plotting the heights
# plt.figure(figsize=(10, 6))
# plt.hist(heights, bins=30, color='lightgreen')
# plt.title('Distribution of Image Heights')
# plt.xlabel('Height')
# plt.ylabel('Frequency')
# plt.show()

# # Optionally, plot the aspect ratios
# aspect_ratios = [w/h for w, h in zip(widths, heights)]
# plt.figure(figsize=(10, 6))
# plt.hist(aspect_ratios, bins=30, color='salmon')
# plt.title('Distribution of Image Aspect Ratios')
# plt.xlabel('Aspect Ratio (Width/Height)')
# plt.ylabel('Frequency')
# plt.show()

plt.figure(figsize=(10, 6))

# Create a scatter plot
plt.scatter(widths, heights, alpha=0.5, color='blue')

plt.title('Scatter Plot of Image Sizes')
plt.xlabel('Width')
plt.ylabel('Height')
plt.grid(True)

# Optionally, you can add a line representing the common aspect ratio
# For example, for an aspect ratio of 16:9, you can uncomment the following line:
# plt.plot(widths, [w * 9/16 for w in widths], color='red', linestyle='--', label='16:9 Aspect Ratio')

plt.show()
