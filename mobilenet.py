import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


print(tf.__version__)

# Restrict TensorFlow to only allocate necessary GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Define the MobileNet architecture with adjustable hyperparameters
def create_mobilenet(input_shape=(250, 250, 3), num_classes=5, num_dense_units=128, learning_rate=0.001, optimizer='adam'):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # MobileNet base model with pre-trained weights for transfer learning
    mobilenet_base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=1.0,
        include_top=False,
        weights='imagenet'  # Use pre-trained weights for transfer learning
    )

    # Add the MobileNet base model as a feature extractor
    x = mobilenet_base(inputs)

    # Global Average Pooling layer
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer with adjustable number of units
    x = layers.Dense(num_dense_units, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Add dropout to prevent overfitting

    # Output layer with softmax activation (5 classes)
    output = layers.Dense(num_classes, activation='softmax')(x)

    # Build the model
    model = models.Model(inputs, output)

    # Set the optimizer
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Function to train the model
def train_model(model, train_generator, validation_generator, batch_size=32, epochs=1):
    # Early stopping and model checkpointing
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('logdir/mobilenet/best_model.h5', monitor='val_accuracy', save_best_only=True)

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint, lr_scheduler],
        verbose=1
    )

    return history

# Function to test the model
def test_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    return test_generator

# Path to your dataset
train_data_dir = 'data/RiceData/train'
test_data_dir = 'data/RiceData/test'

# ImageDataGenerator to preprocess and augment images
train_datagen = ImageDataGenerator(
    validation_split=0.2,  # Reserve 20% of training data for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()  # Only rescale for testing

# Load training and validation data from directories
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(250, 250),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Subset for training
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(250, 250),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Subset for validation
)

# Load test data from directory
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(250, 250),
    batch_size=32,
    class_mode='categorical'
)

# Define hyperparameters
input_shape = (250, 250, 3)  # RGB image
num_classes = 5            # 5 rice varieties (classes)
num_dense_units = 128      # Number of units in the dense layer
learning_rate = 0.0001     # Learning rate for the optimizer
optimizer = 'adam'         # Optimizer choice ('adam', 'sgd', 'rmsprop')

# Create the model
model = create_mobilenet(input_shape=input_shape, num_classes=num_classes, 
                         num_dense_units=num_dense_units, learning_rate=learning_rate, optimizer=optimizer)

# Train the model
history = train_model(model, train_generator, validation_generator, batch_size=32, epochs=1)

# Test the model and get the test data
rator = test_model(model, test_generator)

# Collect true labels and predictions from the test generator
test_labels = test_generator.classes  # True labels (from the generator)
test_class_indices = list(test_generator.class_indices.keys())  # Class labels

# Predict for the entire test set
test_generator.reset()  # Reset generator for predictions
test_predictions = model.predict(test_generator, verbose=1)

# Convert predictions to class indices
test_pred_labels = np.argmax(test_predictions, axis=1)

# Ensure the number of predictions matches the number of samples
if len(test_labels) != len(test_pred_labels):
    raise ValueError(f"Mismatch: {len(test_labels)} true labels vs {len(test_pred_labels)} predicted labels")

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, test_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_class_indices, yticklabels=test_class_indices)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the confusion matrix
plt.show()

# Optionally, calculate and display the normalized confusion matrix
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=test_class_indices, yticklabels=test_class_indices)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Normalized Confusion Matrix')
plt.savefig('confusion_matrix_normalized.png')  # Save the normalized confusion matrix
plt.show()

# Classification report
class_report = classification_report(test_labels, test_pred_labels, target_names=test_generator.class_indices.keys())
print("Classification Report:\n", class_report)

# Plot and save training accuracy graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('model_accuracy.png')  # Save accuracy graph
plt.show()

# Plot and save loss graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('model_loss.png')  # Save loss graph
plt.show()

# Optionally, save the final model
model.save('logdir/mobilenet/rice_variety_model')
