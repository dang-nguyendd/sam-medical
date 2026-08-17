# import pickle

# # Open the file in read-binary mode
# with open('./model_pth/YOLOSAM_v1_run1_CVC-ColonDB/training_history.pkl', 'rb') as file:
#     data = pickle.load(file)

# print(data)

import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Define dataset structure and their corresponding extensions
base_dir = "./data/CVC-ClinicDB"
folder_extensions = {
    "images": ".png",
    "masks": ".png",
    "labels": ".txt"
}

def rebalance_dataset():
    val_img_dir = os.path.join(base_dir, "images", "val")
    test_img_dir = os.path.join(base_dir, "images", "test")
    
    if not os.path.exists(val_img_dir) or not os.path.exists(test_img_dir):
        print("Error: Could not find 'val' or 'test' image directories. Check your path.")
        return

    # 1. Get base filenames (without extensions) from the images directory
    val_bases = sorted([os.path.splitext(f)[0] for f in os.listdir(val_img_dir) if f.endswith('.png')])
    test_bases = sorted([os.path.splitext(f)[0] for f in os.listdir(test_img_dir) if f.endswith('.png')])
    
    # 2. Calculate the exact number of files to move (1/3 of the 15% split)
    num_to_move_val = len(val_bases) // 3
    num_to_move_test = len(test_bases) // 3
    
    # Randomly sample the base names to be moved
    bases_to_move_from_val = random.sample(val_bases, num_to_move_val)
    bases_to_move_from_test = random.sample(test_bases, num_to_move_test)
    
    print(f"Moving {num_to_move_val} file sets from 'val' to 'train'...")
    print(f"Moving {num_to_move_test} file sets from 'test' to 'train'...")

    # 3. Safely execute file transfers using correct extension for each folder
    for folder, ext in folder_extensions.items():
        # Move validation files
        for base_name in bases_to_move_from_val:
            file_name = base_name + ext
            src = os.path.join(base_dir, folder, "val", file_name)
            dst = os.path.join(base_dir, folder, "train", file_name)
            if os.path.exists(src):
                shutil.move(src, dst)
                
        # Move test files
        for base_name in bases_to_move_from_test:
            file_name = base_name + ext
            src = os.path.join(base_dir, folder, "test", file_name)
            dst = os.path.join(base_dir, folder, "train", file_name)
            if os.path.exists(src):
                shutil.move(src, dst)

    print("Success: Dataset rebalanced to 80/10/10 split with correct extension mapping.")

if __name__ == "__main__":
    rebalance_dataset()