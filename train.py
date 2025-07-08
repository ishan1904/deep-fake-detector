import os
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2







print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

#constants
IMG_SIZE = 128
BATCH_SIZE =32

# Data generator for scaling
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1
    )

#load training data
train_generator = datagen.flow_from_directory(
    "data_sample/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load validation data
val_generator = datagen.flow_from_directory(
    'data_sample/valid',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load test data
test_generator = datagen.flow_from_directory(
    'data_sample/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Important for accurate label prediction mapping
)

# Print out basic info
print("Class indices:", train_generator.class_indices)
print("Train batches:", len(train_generator))
print("Validation batches:", len(val_generator))
print("Test batches:", len(test_generator))

#Build CNN

# model = Sequential([
#        # Input layer
#     Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

#     # Conv Block 1
#     Conv2D(32, (3, 3), activation="relu"),
#     BatchNormalization(),                      # Stabilize learning
#     MaxPooling2D(2, 2),

#     # Conv Block 2
#     Conv2D(64, (3, 3), activation="relu"),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),

#     # Conv Block 3 - NEW: more abstraction
#     Conv2D(128, (3, 3), activation="relu"),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),

#     # Flatten to feed into dense layers
#     Flatten(),

#     # Dense Block
#     Dense(128, activation='relu'),
#     Dropout(0.5),                              # Still good to keep this

#     # Dense(64, activation='relu'),              # NEW: additional dense layer
#     # Dropout(0.3),                              # Slightly lower dropout

#     # Output layer
#     Dense(1, activation='sigmoid')             # Binary classification
# ])

# # Compile the model
# model.compile(
#     optimizer=Adam(learning_rate=1e-4), 
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )


# # Summary of the model architecture
# model.summary()

# Load the MobileNetV2 base (excluding top dense layers)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model for now

# Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


EPOCHS = 15
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stop]
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

model.save("deepfake_detector.keras")
print(f"Model saved at: deepfake_detector.keras")

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
