import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from dataset import get_dataloaders
from model import SimpleCNN
import argparse
import wandb

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Calculate per-class and overall metrics using PyTorch."""
    # Convert to 1D tensors if needed
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate per-class metrics
    per_class_correct = torch.zeros(num_classes, device=predictions.device)
    per_class_total = torch.zeros(num_classes, device=predictions.device)
    
    for cls in range(num_classes):
        cls_mask = targets == cls
        per_class_correct[cls] = (predictions[cls_mask] == cls).sum()
        per_class_total[cls] = cls_mask.sum()
    
    # Calculate metrics
    per_class_acc = per_class_correct / per_class_total
    overall_acc = per_class_correct.sum() / per_class_total.sum()
    
    return overall_acc, per_class_acc

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        batch_correct = predicted.eq(target).sum().item()
        correct += batch_correct
        
        # Calculate batch accuracy
        batch_acc = 100. * batch_correct / target.size(0)
        
        # Log batch-level metrics to wandb
        wandb.log({
            "batch_loss": loss.item(),
            "batch_acc": batch_acc,
            "batch": batch_idx
        })
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device, num_classes):
    model.eval()
    test_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, pred = output.max(1)
            
            predictions.append(pred)
            targets.append(target)
    
    # Concatenate all batches
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    
    # Calculate metrics
    overall_acc, per_class_acc = calculate_metrics(predictions, targets, num_classes)
    
    # Format metrics for printing
    metrics_report = "\nPer-class accuracies:\n"
    for cls in range(num_classes):
        if cls % 5 == 0 and cls != 0:  # Add newline every 5 classes
            metrics_report += "\n"
        metrics_report += f"Class {cls:3d}: {per_class_acc[cls]*100:5.2f}%  "
    
    return test_loss / len(test_loader), overall_acc * 100, metrics_report

def compute_class_embeddings(model, dataloader, num_classes, device):
    """Compute and store average embeddings for each class."""
    print("\nComputing class embeddings...")
    model.eval()
    
    # Initialize storage for embeddings and counts
    class_embeddings = torch.zeros(num_classes, 1024).to(device)  # 1024 is embedding dim
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
    
    # Compute averages
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_embeddings[i] /= class_counts[i]
    
    # Verify no classes were empty
    empty_classes = (class_counts == 0).sum().item()
    if empty_classes > 0:
        print(f"Warning: {empty_classes} classes had no samples!")
    
    return class_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="font_dataset_npz/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    wandb.init(
        project="Font-Familiarity",
        name=f"experiment_{time.strftime('%Y%m%d_%H%M%S')}",  # Gives each run a unique name
        config={
            "architecture": "CNN",  # or whatever you're using
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            # Add any other hyperparameters
            }
        )
    # Training settings
    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading data...")
    train_loader, test_loader, num_classes = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size
    )
    
    # Initialize model, criterion, and optimizer
    print(f"Initializing model (num_classes={num_classes})...")
    model = SimpleCNN(num_classes=num_classes).to(device)
    wandb.watch(model, log_freq=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch+1}/{epochs}')
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_loss, test_acc, metrics_report = evaluate(
            model, test_loader, criterion, device, num_classes
        )
        
        epoch_time = time.time() - start_time
        
        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time": epoch_time,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        for cls in range(num_classes):
            wandb.log({
                f"class_{cls}_accuracy": per_class_acc[cls].item() * 100,
                "epoch": epoch + 1
            })
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Time taken: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
        print(metrics_report)
        
        # Save if best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"New best model! (Test Acc: {test_acc:.2f}%)")
            
            # Compute class embeddings using training data
            class_embeddings = compute_class_embeddings(
                model, train_loader, num_classes, device
            )
            
            # Save both model and embeddings
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'class_embeddings': class_embeddings,
                'num_classes': num_classes
            }, 'best_model.pt')

    print("\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")

if __name__ == "__main__":
    main()
# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from model import FontEncoder, ContrastiveLoss
# from typing import List, Dict, Tuple

# @torch.compile
# class FontSimilarityModel:
#     def __init__(self, latent_dim: int = 128):
#         self.encoder = FontEncoder(latent_dim)
#         self.criterion = ContrastiveLoss()
#         self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.encoder.to(self.device)
        
#     def train_epoch(self, dataloader: DataLoader) -> float:
#         self.encoder.train()
#         total_loss = 0
        
#         for batch_idx, (images, labels) in enumerate(dataloader):
#             images, labels = images.to(self.device), labels.to(self.device)
            
#             self.optimizer.zero_grad()
#             embeddings = self.encoder(images)
#             loss = self.criterion(embeddings, labels)
            
#             loss.backward()
#             self.optimizer.step()
            
#             total_loss += loss.item()
            
#         return total_loss / len(dataloader)
    

#     def compute_centroids(self, dataloader: DataLoader) -> Dict[int, torch.Tensor]:
#         self.encoder.eval()
#         centroids = {}
#         sample_counts = {}
        
#         with torch.no_grad():
#             for images, labels in dataloader:
#                 images = images.to(self.device)
#                 embeddings = self.encoder(images)
                
#                 for emb, label in zip(embeddings, labels):
#                     label = label.item()
#                     if label not in centroids:
#                         centroids[label] = emb
#                         sample_counts[label] = 1
#                     else:
#                         centroids[label] += emb
#                         sample_counts[label] += 1
        
#         # Compute averages and normalize
#         for label in centroids:
#             centroids[label] = centroids[label] / sample_counts[label]
#             centroids[label] = F.normalize(centroids[label], p=2, dim=0)
        
#         return centroids
    
#     def find_similar_fonts(self, query_idx: int, centroids: Dict[int, torch.Tensor], 
#                           k: int = 5) -> List[Tuple[int, float]]:
#         query_centroid = centroids[query_idx]
#         similarities = []
        
#         for idx, centroid in centroids.items():
#             if idx != query_idx:
#                 sim = F.cosine_similarity(query_centroid.unsqueeze(0), 
#                                         centroid.unsqueeze(0))
#                 similarities.append((idx, sim.item()))
        
#         return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
