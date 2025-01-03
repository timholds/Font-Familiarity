import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from dataset import get_dataloaders
from model import SimpleCNN
import argparse
import wandb
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.optim import AdamW
from metrics import ClassificationMetrics

# Add this with your other imports at the top of the file
from metrics import ClassificationMetrics


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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                warmup_epochs, warmup_scheduler, main_scheduler, metrics_calculator):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (data, target) in enumerate(pbar):
        # Start timing this batch
        metrics_calculator.start_batch()
        
        # Time data transfer
        data_transfer_start = time.time()
        data, target = data.to(device), target.to(device)
        data_transfer_time = time.time() - data_transfer_start
        
        # Forward/backward pass
        forward_start = time.time()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        forward_backward_time = time.time() - forward_start
        
        # Optimizer step
        optimizer_start = time.time()
        optimizer.step()
        optimizer_time = time.time() - optimizer_start
        
        # Update schedulers
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        # Compute batch metrics
        batch_metrics = metrics_calculator.compute_batch_metrics(
            logits=output,
            targets=target,
            loss=loss.item(),
            batch_size=target.size(0),
            optimizer=optimizer,
            batch_times={
                'data_transfer': data_transfer_time,
                'forward_backward': forward_backward_time,
                'optimizer': optimizer_time
            }
        )
        
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
        
        # Log metrics periodically
        if batch_idx % 50 == 0:
            batch_metrics['global_step'] = batch_idx + epoch * len(train_loader)
            wandb.log(batch_metrics)
        
        # Update progress bar
        pbar.set_postfix(
            metrics_calculator.get_progress_bar_stats(
                loss.item(), output, target, target.size(0)
            )
        )
    
    return running_loss


def evaluate(model, test_loader, criterion, device, metrics_calculator, epoch=None):
    """Evaluate the model."""
    model.eval()
    test_loss = 0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            
            test_loss += criterion(logits, target).item()
            all_logits.append(logits)
            all_targets.append(target)
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    # Compute metrics
    metrics = metrics_calculator.compute_all_metrics(
        logits=all_logits,
        targets=all_targets,
        model=model,
        epoch=epoch
    )
    
    # Add loss to metrics
    metrics['test_loss'] = test_loss / len(test_loader)
    
    return metrics

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
    parser.add_argument("--data_dir", default="font_dataset_npz_test/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--initial_channels", type=int, default=16)
    
    args = parser.parse_args()
    warmup_epochs = max(args.epochs // 5, 1)
    
    # Initialize wandb
    wandb.init(
        project="Font-Familiarity",
        name=f"experiment_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            **vars(args),
            "architecture": "CNN",
            "warmup_epochs": warmup_epochs,
            "optimizer": "AdamW"
        }
    )
    
    # Set up metrics logging
    wandb.define_metric("batch_loss", step_metric="global_step")
    wandb.define_metric("batch_acc", step_metric="global_step")
    wandb.define_metric("*", step_metric="epoch")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading data...")
    train_loader, test_loader, num_classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Initialize model
    print(f"Initializing model (num_classes={num_classes})...")
    model = SimpleCNN(
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        input_size=args.resolution,
        initial_channels=args.initial_channels
    ).to(device)
    
    # Initialize metrics calculator
    metrics_calculator = ClassificationMetrics(num_classes=num_classes, device=device)
    
    # Set up training
    wandb.watch(model, log_freq=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up schedulers
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs * len(train_loader)
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - warmup_epochs) * len(train_loader),
        eta_min=1e-6
    )
    
    # Training loop
    print("Starting training...")
    best_test_acc = 0.0
    metrics_calculator.reset_timing()  # Start timing from here
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            warmup_epochs, warmup_scheduler, main_scheduler, metrics_calculator
        )
        
        # Evaluate
        test_metrics = evaluate(
            model, test_loader, criterion, device, metrics_calculator, epoch
        )
        
        # Add training metrics
        test_metrics.update({
            'train_loss': train_loss,
            'epoch_time': time.time() - start_time,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })
        
        # Log all metrics
        wandb.log(test_metrics)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Test Loss: {test_metrics["test_loss"]:.3f}')
        print(f'Test Acc: {test_metrics["top1_acc"]:.2f}%')
        print(f'Top-5 Acc: {test_metrics["top5_acc"]:.2f}%')
        print(f'Mean Per-Class Acc: {test_metrics["mean_per_class_acc"]:.2f}%')
        
        # Save best model
        if test_metrics['top1_acc'] > best_test_acc:
            best_test_acc = test_metrics['top1_acc']
            print(f"New best model! (Test Acc: {best_test_acc:.2f}%)")
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': test_metrics,
                'num_classes': num_classes
            }
        
        # Update wandb summary periodically
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            wandb.run.summary.update({
                "best_test_acc": best_test_acc,
                "final_train_loss": train_loss,
                "epochs_to_best": epoch + 1,
                "train_test_gap": abs(test_metrics['train_loss'] - test_metrics['test_loss'])
            })
    
    print("\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    
    # Save best model
    torch.save(best_model_state, 'best_model.pt')


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
