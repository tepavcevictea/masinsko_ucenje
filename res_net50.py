import os
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Lokacija dataset-a
dataset_path = "data_set"

# Definisanje
classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Inicijalizacija lista za cuvanje podataka
data = []
labels = []

# Ucitavanje slika kroz klase
for class_name in classes:
    class_path = os.path.join(dataset_path, "Training", class_name)
    for filename in os.listdir(class_path):
        img_path = os.path.join(class_path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Da sve slike budu 224x224px
        data.append(img)
        labels.append(classes.index(class_name))

# KOnvertovanje listi u numpy
data = np.array(data) / 255.0  # Normalizacija vrednosti piksela
labels = np.array(labels)

# Podela podataka u train i test
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# ImageDataGenerator za data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Učitavanje pretreniranog ResNet50 modela (include_top=False isključuje gornje (potpuno povezane) slojeve)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classes=1000,
    
)

# Zamrzavanje slojeva pretreniranog modela
for layer in resnet_model.layers:
    layer.trainable = False

# Kreiranje novog modela dodavanjem pretreniranog ResNet50 i dodatnih slojeva
model = Sequential()
model.add(resnet_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Kompajliranje modela
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treniranje modela
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

# Evaluacija modela na test skupu
test_data = []
test_labels = []

for class_name in classes:
    class_path = os.path.join(dataset_path, "Testing", class_name)
    for filename in os.listdir(class_path):
        img_path = os.path.join(class_path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        test_data.append(img)
        test_labels.append(classes.index(class_name))

test_data = np.array(test_data) / 255.0
test_labels = np.array(test_labels)

loss, accuracy = model.evaluate(test_data, test_labels)

print(f"Tacnost na test skupu sa Dropout-om: {accuracy}")