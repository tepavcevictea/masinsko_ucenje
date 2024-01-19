import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Constants
DATASET_PATH = "../data_set"
TRAIN = "Training"
TEST = "Testing"
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMAGE_SIZE = (224, 224)

def load_images_and_labels(dataset_path, train_test, classes):
    data = []
    labels = []
    for class_name in classes:
        class_path = os.path.join(dataset_path, train_test, class_name)
        for filename in os.listdir(class_path):
            try:
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                img = cv2.resize(img, IMAGE_SIZE)
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                img = img / 255.0  # Normalize pixel values
                data.append(img)
                labels.append(classes.index(class_name))
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return np.array(data), np.array(labels)

def create_model(input_shape, num_classes):
    # Load the pre-trained ResNet model
    resnet_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    # Previously tried fine tuning, but unfroze all at the end
    for layer in resnet_model.layers:
        layer.trainable = True
    
    # Create a new sequential model
    model = Sequential([
        # Replicate the grayscale channel to 3 channels
        Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1), input_shape=input_shape),
        resnet_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    initial_learning_rate = 0.01
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    optimizer = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # Load images and labels
    data, labels = load_images_and_labels(DATASET_PATH, TRAIN, CLASSES)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    checkpoint_callback = ModelCheckpoint(
        filepath="run6/weights/weights_epoch_{epoch:02d}_val_accuracy_{val_accuracy:.4f}.h5",
        monitor='val_accuracy',  # Monitor the validation accuracy
        verbose=1,
        save_best_only=True,  # Save only the best model based on validation accuracy
        save_weights_only=True,
        mode='max',  # 'max' mode means the callback will look for the maximum value of the monitored metric
        save_freq='epoch'
        # Note: 'period' parameter is not needed for TensorFlow 2.0 and above
    )

    # Create and train the model
    # When calling create_modelmarkdown preview en
    model = create_model(IMAGE_SIZE + (1,), len(CLASSES))  # Ensure the input_shape reflects the single channel

    ### Evaluate ###
    model.load_weights("run5_weights_epoch_87_val_accuracy_0.8519.h5")

    ### Train ###
    # data = model.fit(datagen.flow(X_train, y_train, batch_size=16), epochs=120, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])
    # hist_df = pd.DataFrame(data.history)
    # hist_df.to_excel("run6/training_metrics.xlsx")
    
    # Evaluate the model on the test set
    test_data , test_labels = load_images_and_labels(DATASET_PATH, TEST, CLASSES)

    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy for the current run: {accuracy}")

if __name__ == "__main__":
    main()
