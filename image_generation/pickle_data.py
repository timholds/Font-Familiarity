import os
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
from tqdm import tqdm

def create_label_mapping(root_dir):
    """Create a mapping of font names to integer labels"""
    font_dirs = [d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d))]
    return {font: idx for idx, font in enumerate(sorted(font_dirs))}

def process_dataset(root_dir, output_file, batch_size=10000):
    """Process the dataset and save as pickle file"""
    
    # Create label mapping
    label_mapping = create_label_mapping(root_dir)
    print(f"Found {len(label_mapping)} font classes")
    
    # Count total files for progress bar
    total_files = sum(len(files) for _, _, files in os.walk(root_dir))
    
    # Storage for current batch
    current_batch = []
    current_labels = []
    batch_count = 1
    
    with tqdm(total=total_files, desc="Processing images") as pbar:
        # Iterate through font directories
        for font_name in sorted(os.listdir(root_dir)):
            font_dir = os.path.join(root_dir, font_name)
            if not os.path.isdir(font_dir):
                continue
            
            label = label_mapping[font_name]
            
            # Process all images in font directory
            for img_file in sorted(os.listdir(font_dir)):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(font_dir, img_file)
                
                try:
                    # Load image as numpy array
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_array = np.array(img)
                        
                        # Convert to PyTorch's expected format (C, H, W)
                        img_array = img_array.transpose(2, 0, 1)
                        
                        current_batch.append(img_array)
                        current_labels.append(label)
                        
                        # Save batch if it reaches the batch size
                        if len(current_batch) >= batch_size:
                            save_batch(current_batch, current_labels, 
                                     f"{output_file}_{batch_count}.pkl")
                            current_batch = []
                            current_labels = []
                            batch_count += 1
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                
                pbar.update(1)
    
    # Save any remaining images in the final batch
    if current_batch:
        save_batch(current_batch, current_labels, 
                  f"{output_file}_{batch_count}.pkl")
    
    # Save label mapping
    with open(f"{output_file}_mapping.pkl", 'wb') as f:
        pickle.dump(label_mapping, f)
    
    return label_mapping

def save_batch(images, labels, filename):
    """Save a batch of images and labels"""
    data = {
        'data': np.array(images),    # Shape: (N, C, H, W)
        'labels': np.array(labels)   # Shape: (N,)
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    root_dir = "font-images"
    output_file = "font_dataset"
    
    label_mapping = process_dataset(root_dir, output_file)
    print("Dataset processing complete!")
    print(f"Label mapping saved to {output_file}_mapping.pkl")