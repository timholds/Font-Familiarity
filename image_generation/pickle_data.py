import os
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm

def debug_batch_size(images, labels):
    """Print debug info about batch before saving"""
    data_array = np.array(images)
    labels_array = np.array(labels)
    
    print(f"\nDebug batch info:")
    print(f"Number of images: {len(images)}")
    print(f"Data array shape: {data_array.shape}")
    print(f"Data array dtype: {data_array.dtype}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Memory usage of data array: {data_array.nbytes / (1024**3):.2f} GB")
    
    # Check first few images
    print("\nFirst few image shapes:")
    for i in range(min(3, len(images))):
        print(f"Image {i} shape: {images[i].shape}")

def save_batch_with_debug(images, labels, filename):
    """Save a batch of images and labels with debugging info"""
    debug_batch_size(images, labels)
    
    data = {
        'data': np.array(images),
        'labels': np.array(labels)
    }
    
    # Get size before saving
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    # Print file size after saving
    file_size = os.path.getsize(filename) / (1024**3)  # Convert to GB
    print(f"Saved file size: {file_size:.2f} GB")

# Test with just first batch
def test_processing(root_dir, output_dir, max_images=70000):
    os.makedirs(output_dir, exist_ok=True)
    
    current_batch = []
    current_labels = []
    processed = 0
    
    # Process just enough images for one batch
    for font_name in sorted(os.listdir(root_dir)):
        if processed >= max_images:
            break
            
        font_dir = os.path.join(root_dir, font_name)
        if not os.path.isdir(font_dir):
            continue
        
        for img_file in sorted(os.listdir(font_dir)):
            if processed >= max_images:
                break
                
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(font_dir, img_file)
            
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                    current_batch.append(img_array)
                    current_labels.append(0)  # Dummy label for testing
                    processed += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Save test batch
    if current_batch:
        save_batch_with_debug(current_batch, current_labels,
                            os.path.join(output_dir, 'test_batch.pkl'))

if __name__ == "__main__":
    root_dir = "font-images2"  # Updated directory name
    output_dir = "font_dataset_test"
    
    print("Running test processing...")
    test_processing(root_dir, output_dir, max_images=70000)