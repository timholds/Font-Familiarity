import os
import numpy as np
from PIL import Image

def analyze_image_format(root_dir):
    # Get first image we can find
    for font_dir in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, font_dir)
        if os.path.isdir(dir_path):
            for img_file in os.listdir(dir_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dir_path, img_file)
                    with Image.open(img_path) as img:
                        # Print original mode
                        print(f"Original image mode: {img.mode}")
                        
                        # Load as grayscale
                        img_array = np.array(img)
                        
                        # Print original file size
                        original_size = os.path.getsize(img_path)
                        # Print numpy array size
                        array_size = img_array.nbytes
                        # Print array shape
                        shape = img_array.shape
                        # Print array dtype
                        dtype = img_array.dtype
                        
                        print(f"Original file size: {original_size / 1024:.2f} KB")
                        print(f"Numpy array size: {array_size / 1024:.2f} KB")
                        print(f"Image shape: {shape}")
                        print(f"Data type: {dtype}")
                        
                        # Print unique values to check if binary
                        unique_vals = np.unique(img_array)
                        print(f"Unique values: {unique_vals}")
                        return

root_dir = "font-images2"
analyze_image_format(root_dir)