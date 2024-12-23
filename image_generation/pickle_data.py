import os
import numpy as np
from PIL import Image
import pickle
import gzip
from tqdm import tqdm

def save_batch_numpy(images, labels, filename):
    """Save using numpy's compressed format"""
    data_array = np.array(images)
    labels_array = np.array(labels)
    
    print(f"\nNumpy compression:")
    print(f"Number of images: {len(images)}")
    print(f"Data array shape: {data_array.shape}")
    print(f"Memory usage before compression: {data_array.nbytes / (1024**3):.2f} GB")
    
    np.savez_compressed(filename, data=data_array, labels=labels_array)
    file_size = os.path.getsize(filename) / (1024**3)
    print(f"Compressed file size: {file_size:.2f} GB")
    
    # Test loading
    loaded = np.load(filename)
    print("Successfully tested loading")
    return file_size

def save_batch_pickle_gzip(images, labels, filename):
    """Save using pickle with gzip compression"""
    data_array = np.array(images)
    labels_array = np.array(labels)
    
    print(f"\nPickle+Gzip compression:")
    print(f"Number of images: {len(images)}")
    print(f"Data array shape: {data_array.shape}")
    print(f"Memory usage before compression: {data_array.nbytes / (1024**3):.2f} GB")
    
    data = {
        'data': data_array,
        'labels': labels_array
    }
    
    with gzip.open(filename, 'wb', compresslevel=9) as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(filename) / (1024**3)
    print(f"Compressed file size: {file_size:.2f} GB")
    
    # Test loading
    with gzip.open(filename, 'rb') as f:
        loaded = pickle.load(f)
    print("Successfully tested loading")
    return file_size

def test_compression(root_dir, output_dir, max_images=70000):
    """Test both compression methods"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect images
    print("Collecting images...")
    current_batch = []
    current_labels = []
    processed = 0
    
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
                    
                    if processed % 1000 == 0:
                        print(f"Processed {processed} images")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Test both compression methods
    if current_batch:
        numpy_size = save_batch_numpy(current_batch, current_labels,
                                    os.path.join(output_dir, 'test_batch.npz'))
        pickle_size = save_batch_pickle_gzip(current_batch, current_labels,
                                           os.path.join(output_dir, 'test_batch.pkl.gz'))
        
        print(f"\nComparison:")
        print(f"Numpy compressed size: {numpy_size:.2f} GB")
        print(f"Pickle+Gzip size: {pickle_size:.2f} GB")

if __name__ == "__main__":
    root_dir = "font-images2"
    output_dir = "font_dataset_test"
    
    print("Testing compression methods...")
    test_compression(root_dir, output_dir)