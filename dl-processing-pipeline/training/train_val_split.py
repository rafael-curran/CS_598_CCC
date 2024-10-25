import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Split a dataset of images into training and validation sets.

    Parameters:
        source_dir (str): The source directory containing subfolders of images.
        train_dir (str): The target directory for the training images.
        val_dir (str): The target directory for the validation images.
        split_ratio (float): The ratio of images to go into the training set.
    """

    # Create the train and val directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Loop through each subfolder (class) in the source directory
    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)

        # Skip if not a directory (important for ignoring hidden files)
        if not os.path.isdir(class_path):
            continue

        # Get all image files in the class folder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Shuffle the images
        random.shuffle(images)

        # Calculate the split index
        split_index = int(len(images) * split_ratio)

        # Split the images into train and val sets
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create corresponding class directories in train and val folders
        train_class_dir = os.path.join(train_dir, class_folder)
        val_class_dir = os.path.join(val_dir, class_folder)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Move training images
        for img in train_images:
            src_img_path = os.path.join(class_path, img)
            dst_img_path = os.path.join(train_class_dir, img)
            shutil.copy(src_img_path, dst_img_path)

        # Move validation images
        for img in val_images:
            src_img_path = os.path.join(class_path, img)
            dst_img_path = os.path.join(val_class_dir, img)
            shutil.copy(src_img_path, dst_img_path)

        print(f"Class '{class_folder}': {len(train_images)} images for training, {len(val_images)} images for validation")

# Example usage:
source_directory = '/Users/bohu/Documents/GitHub Desktop/CS_598_CCC/dl-processing-pipeline/training/imagenet/train'  # Replace with the path to your source directory
train_directory = '/Users/bohu/Documents/GitHub Desktop/CS_598_CCC/dl-processing-pipeline/training/imagenet/training'
val_directory = '/Users/bohu/Documents/GitHub Desktop/CS_598_CCC/dl-processing-pipeline/training/imagenet/validation'
split_ratio = 0.8  # 80% train, 20% val

split_dataset(source_directory, train_directory, val_directory, split_ratio)
