import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib


#constants
IMG_SIZE = 128
BATCH_SIZE =32

# Data generator for scaling
datagen = ImageDataGenerator(rescale=1./255)

#load training data
train_generator = datagen.flow_from_directory(
    "data/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load validation data
val_generator = datagen.flow_from_directory(
    'data/valid',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load test data
test_generator = datagen.flow_from_directory(
    'data/test',
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

model = Sequential([
     # layer 1:
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),

     # Layer 2: MaxPooling
    MaxPooling2D(2, 2),
    
    # Layer 3: Second Conv layer
    Conv2D(64, (3, 3), activation='relu'),
    
    # Layer 4: Second MaxPooling
    MaxPooling2D(2, 2),

    # Flatten layer: 2D → 1D
    Flatten(),

    # Dense (fully connected) layer
    Dense(128, activation='relu'),

    # Dropout: randomly turn off 50% of neurons during training (for regularization)
    Dropout(0.5),

    # Output layer: 1 neuron for binary classification
    Dense(1, activation='sigmoid')          
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Summary of the model architecture
model.summary()

Epoochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=Epoochs,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

model.save("deppfake-detector.h5")
print("Model saved as deepfake_model.h5")

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
