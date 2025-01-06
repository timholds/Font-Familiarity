import torch
import numpy as np
from model import SimpleCNN
from dataset import get_dataloaders
import argparse
from tqdm import tqdm

def load_model(model_path):
    """Load the trained model and move it to appropriate device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved state
    state = torch.load(model_path, map_location=device)
    
    # Initialize model with same parameters as training
    model = SimpleCNN(
        num_classes=state['num_classes'],
        embedding_dim=state['model_state_dict']['embedding_layer.0.weight'].shape[1]
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    return model, device

def compute_class_embeddings(model, dataloader, num_classes, device):
    """Compute average embedding for each class."""
    print("\nComputing class embeddings...")
    
    # Initialize storage for embeddings and counts
    class_embeddings = torch.zeros(num_classes, model.embedding_layer[0].out_features).to(device)
    class_counts = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Computing embeddings'):
            data, target = data.to(device), target.to(device)
            
            # Get embeddings (features before final classification layer)
            embeddings = model.get_embedding(data)
            
            # Accumulate embeddings for each class
            for i in range(len(target)):
                class_idx = target[i].item()
                class_embeddings[class_idx] += embeddings[i]
                class_counts[class_idx] += 1
    
    # Compute averages and normalize
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_embeddings[i] /= class_counts[i]
    
    # L2 normalize embeddings
    class_embeddings = torch.nn.functional.normalize(class_embeddings, p=2, dim=1)
    
    # Verify no classes were empty
    empty_classes = (class_counts == 0).sum().item()
    if empty_classes > 0:
        print(f"Warning: {empty_classes} classes had no samples!")
    
    return class_embeddings.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model .pt file")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output_path", default="class_embeddings.npy", help="Path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, device = load_model(args.model_path)
    
    # Get dataloaders
    train_loader, _, num_classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Compute embeddings
    class_embeddings = compute_class_embeddings(model, train_loader, num_classes, device)
    
    # Save embeddings
    print(f"\nSaving embeddings to {args.output_path}")
    np.save(args.output_path, class_embeddings)
    print(f"Shape of embeddings: {class_embeddings.shape}")

if __name__ == "__main__":
    main()