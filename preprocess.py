import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split

# Define the dataset path
dataset_path = 'data/Rice_Image_Dataset'
output_train_dir = 'data/RiceData/train'
output_test_dir = 'data/RiceData/test'

# Function to remove .DS_Store files from directories
def remove_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        if '.DS_Store' in files:
            os.remove(os.path.join(root, '.DS_Store'))

# Function to create the train/test split
def preprocess_and_split(dataset_path, output_train_dir, output_test_dir, test_size=0.2):
    # Create output directories for train and test
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)

    # Create a list to hold file paths and corresponding labels
    image_paths = []
    labels = []
    class_names = os.listdir(dataset_path)
    
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_path, img_name))
                    labels.append(class_name)

    # Split the dataset into train and test sets (80/20 split)
    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=test_size, stratify=labels)
    
    # Create subdirectories for each class in train and test directories
    for class_name in class_names:
        os.makedirs(os.path.join(output_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(output_test_dir, class_name), exist_ok=True)
    
    # Function to process and save images
    def process_and_save(image_paths, labels, output_dir):
        for img_path, label in zip(image_paths, labels):
            img = load_img(img_path)  # Load image without resizing or grayscale conversion
            img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            label_dir = os.path.join(output_dir, label)
            img_name = os.path.basename(img_path)
            img_array = img_array.astype('float32')
            # Save the image in the respective class directory
            output_img_path = os.path.join(label_dir, img_name)
            tf.keras.preprocessing.image.save_img(output_img_path, img_array)

    # Process and save the train and test images
    process_and_save(train_paths, train_labels, output_train_dir)
    process_and_save(test_paths, test_labels, output_test_dir)

    # Remove .DS_Store files from the output directories
    remove_ds_store(output_train_dir)
    remove_ds_store(output_test_dir)

    print(f"Dataset split complete. Train and Test sets saved in {output_train_dir} and {output_test_dir}")

# Call the function to preprocess and split the dataset
preprocess_and_split(dataset_path, output_train_dir, output_test_dir, test_size=0.2)
