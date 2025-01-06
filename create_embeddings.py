import torch
import numpy as np
from ML.model import SimpleCNN
from ML.dataset import get_dataloaders
import argparse
from tqdm import tqdm

def load_model(model_path: str) -> tuple[SimpleCNN, torch.device]:
    """
    Load the trained model and extract architecture details.
    
    The model architecture flow is:
    1. Input image (1, 64, 64)
    2. CNN features -> (128, 4, 4)
    3. Flatten -> 2048
    4. Embedding layer -> 128
    5. Classifier -> num_classes
    
    Args:
        model_path: Path to the saved model checkpoint

    Returns:
        model: Loaded model
        device: Torch device (cuda or cpu)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved state
    state = torch.load(model_path, map_location=device)
    state_dict = state['model_state_dict']
    
    # Extract architecture parameters from state dict
    embedding_weight = state_dict['embedding_layer.0.weight']  # Shape: [embedding_dim, flatten_dim]
    classifier_weight = state_dict['classifier.weight']        # Shape: [num_classes, embedding_dim]
    
    # Get dimensions from the weights
    flatten_dim = embedding_weight.shape[1]    # 2048 (128 channels * 4 * 4)
    embedding_dim = embedding_weight.shape[0]  # 128 (dimension of learned features)
    num_classes = classifier_weight.shape[0]   # Number of font classes
    
    print("\nModel architecture from checkpoint:")
    print(f"- Flattened CNN features: {flatten_dim} (128 channels * 4 * 4 spatial)")
    print(f"- Embedding dimension: {embedding_dim}")
    print(f"- Number of classes: {num_classes}")
    
    # Initialize model with correct parameters
    model = SimpleCNN(
        num_classes=num_classes,
        embedding_dim=embedding_dim
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, device

def compute_class_embeddings(model: SimpleCNN, 
                           dataloader: torch.utils.data.DataLoader, 
                           num_classes: int, 
                           device: torch.device) -> np.ndarray:
    """
    Compute average embedding for each class.
    
    The embeddings are taken from the output of the embedding_layer (128-dim),
    NOT from the flattened CNN features (2048-dim).
    
    Args:
        model: Trained SimpleCNN model
        dataloader: DataLoader containing font images
        num_classes: Number of font classes
        device: Torch device to use for computation

    Returns:
        class_embeddings: Array of shape [num_classes, embedding_dim]
    """
    embedding_dim = model.embedding_layer[0].out_features  # Should be 128
    print(f"\nComputing {embedding_dim}-dimensional embeddings for {num_classes} classes...")
    
    # Initialize storage for embeddings and counts
    class_embeddings = torch.zeros(num_classes, embedding_dim).to(device)
    class_counts = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Computing embeddings'):
            data, target = data.to(device), target.to(device)
            
            # Get embeddings (128-dim features after embedding_layer)
            embeddings = model.get_embedding(data)  # Shape: [batch_size, embedding_dim]
            
            # Accumulate embeddings for each class
            for i in range(len(target)):
                class_idx = target[i].item()
                class_embeddings[class_idx] += embeddings[i]
                class_counts[class_idx] += 1
    
    # Compute averages
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_embeddings[i] /= class_counts[i]
    
    # L2 normalize embeddings for cosine similarity computation
    class_embeddings = torch.nn.functional.normalize(class_embeddings, p=2, dim=1)
    
    # Verify no classes were empty
    empty_classes = (class_counts == 0).sum().item()
    if empty_classes > 0:
        print(f"Warning: {empty_classes} classes had no samples!")
    
    final_embeddings = class_embeddings.cpu().numpy()
    print(f"Final embeddings shape: {final_embeddings.shape}")
    return final_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model .pt file")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output_path", default="class_embeddings.npy", 
                       help="Path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, device = load_model(args.model_path)
    
    # Get dataloaders
    # test_loader, _, num_classes = get_dataloaders(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size
    # )

    from ML.dataset import FontDataset
    from torch.utils.data import DataLoader

    test_dataset = FontDataset(args.data_dir, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    
    # Compute embeddings
    class_embeddings = compute_class_embeddings(model, test_loader, test_dataset.num_classes, device)
    
    # Save embeddings
    print(f"\nSaving embeddings to {args.output_path}")
    np.save(args.output_path, class_embeddings)

if __name__ == "__main__":
    main()