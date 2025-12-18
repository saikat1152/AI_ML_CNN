import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset (32x32 color images in 10 classes)
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}")

# Build CNN model
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Display model architecture
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("\nTraining history saved as 'training_history.png'")

# Make predictions on a few test images
num_samples = 10
predictions = model.predict(x_test[:num_samples])

# Display predictions
plt.figure(figsize=(15, 3))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.axis('off')

    pred_class = np.argmax(predictions[i])
    true_class = y_test[i][0]

    color = 'green' if pred_class == true_class else 'red'
    plt.title(f"Pred: {class_names[pred_class]}\nTrue: {class_names[true_class]}",
              color=color, fontsize=9)

plt.tight_layout()
plt.savefig('predictions.png')
print("Sample predictions saved as 'predictions.png'")

# Save the model
model.save('cifar10_cnn_model.h5')
print("\nModel saved as 'cifar10_cnn_model.h5'")


# Function to predict a single image
def predict_image(model, image, true_label=None):
    """Predict the class of a single image"""
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image, verbose=0)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"Predicted: {class_names[pred_class]} ({confidence:.2f}% confidence)")
    if true_label is not None:
        print(f"True label: {class_names[true_label]}")

    return pred_class


# Test the prediction function
print("\n--- Testing single image prediction ---")
test_idx = 42
predict_image(model, x_test[test_idx], y_test[test_idx][0])

print("\n✓ Project completed successfully!")