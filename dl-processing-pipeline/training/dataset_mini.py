import os
import shutil


def copy_first_20_subfolders(source_folder, target_folder):
    # Get a list of all subfolders in the source directory
    subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    # Sort and select the first 20 subfolders
    subfolders.sort()
    subfolders_to_copy = subfolders[:20]

    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Copy each selected subfolder to the target directory
    for folder in subfolders_to_copy:
        source_path = os.path.join(source_folder, folder)
        target_path = os.path.join(target_folder, folder)

        # Copy the subfolder and its contents to the target folder
        shutil.copytree(source_path, target_path)

    print(f"Copied the first 20 subfolders from {source_folder} to {target_folder}.")


# Example usage
source_folder = '/Users/bohu/Documents/GitHub Desktop/CS_598_CCC/dl-processing-pipeline/training/imagenet/train'
target_folder = '/Users/bohu/Documents/GitHub Desktop/CS_598_CCC/dl-processing-pipeline/training/imagenet/traincopy'
copy_first_20_subfolders(source_folder, target_folder)

source_folder = '/Users/bohu/Documents/GitHub Desktop/CS_598_CCC/dl-processing-pipeline/training/imagenet/val'
target_folder = '/Users/bohu/Documents/GitHub Desktop/CS_598_CCC/dl-processing-pipeline/training/imagenet/valcopy'
copy_first_20_subfolders(source_folder, target_folder)