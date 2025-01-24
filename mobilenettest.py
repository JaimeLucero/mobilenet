import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the saved model
model_path = 'logdir/mobilenet/best_model.h5'

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Path to your test dataset
test_data_dir = 'data/RiceData/test'

# ImageDataGenerator to preprocess test images
test_datagen = ImageDataGenerator()  # Only rescale for testing

# Load test data from the directory
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(250, 250),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for consistent label ordering
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict for the entire test set
test_generator.reset()  # Reset generator for predictions
test_predictions = model.predict(test_generator, verbose=1)

print('preds: ', test_predictions)

# Convert predictions to class indices
test_pred_labels = np.argmax(test_predictions, axis=1)

# Get true labels and class indices
test_labels = test_generator.classes  # True labels
class_indices = list(test_generator.class_indices.keys())  # Class labels

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, test_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_indices, yticklabels=class_indices)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_test.png')  # Save confusion matrix
plt.show()

# Classification Report
class_report = classification_report(test_labels, test_pred_labels, target_names=class_indices)
print("Classification Report:\n", class_report)


