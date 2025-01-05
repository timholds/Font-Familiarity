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

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                warmup_epochs, warmup_scheduler, main_scheduler, metrics_calculator):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (data, target) in enumerate(pbar):
        metrics_calculator.start_batch()
        
        # Forward pass and compute loss
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update schedulers
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        # Update running statistics
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
        pred = output.argmax(dim=1)
        total_correct += (pred == target).sum().item()
        total_samples += target.size(0)
        current_acc = 100. * total_correct / total_samples
        
        # Compute and log batch metrics
        if batch_idx % 50 == 0:
            batch_metrics = {
                'train/batch_loss': loss.item(),
                'train/running_loss': running_loss,
                'train/running_acc': current_acc,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'global_step': batch_idx + epoch * len(train_loader)
            }
            wandb.log(batch_metrics)
        
        # Update progress bar with basic metrics
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'avg_loss': f'{running_loss:.3f}',
            'acc': f'{current_acc:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # Compute final training metrics
    train_metrics = {
        'train/loss': running_loss,
        'train/top1_acc': current_acc,  # Changed to match test metrics naming
        #'train/samples': total_samples,
    }
    
    return train_metrics

def evaluate(model, test_loader, criterion, device, metrics_calculator, epoch=None):
    """Evaluate the model."""
    model.eval()
    test_loss = 0
    all_logits = []
    all_targets = []
    
    # Basic accuracy tracking
    correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Accumulate tensors for advanced metrics
            all_logits.append(output)
            all_targets.append(target)
            
            # Compute basic metrics on the fly
            test_loss += criterion(output, target).item()
            
            # Top-1 accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            
            # Top-5 accuracy
            _, pred5 = output.topk(5, 1, True, True)
            target_expanded = target.view(-1, 1).expand_as(pred5)
            correct5 = pred5.eq(target_expanded).any(dim=1).sum().item()
            top5_correct += correct5
            
            total += target.size(0)
    
    # Compute basic metrics
    avg_loss = test_loss / len(test_loader)
    top1_acc = 100. * correct / total
    top5_acc = 100. * top5_correct / total
    
    # Initialize metrics dictionary with basic metrics
    test_metrics = {
        'test/loss': avg_loss,
        'test/top1_acc': top1_acc,
        'test/top5_acc': top5_acc,
        'test/samples': total,
    }
    
    # Compute advanced metrics on the entire test set
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    # Get advanced metrics from the metrics calculator
    advanced_metrics = metrics_calculator.compute_all_metrics(
        logits=all_logits,
        targets=all_targets,
        model=model,
        epoch=epoch
    )
    
    # Add advanced metrics with 'test/' prefix, excluding duplicates
    for key, value in advanced_metrics.items():
        # Skip metrics we already have
        if key not in ['top1_acc', 'top5_acc', 'loss']:
            test_metrics[f'test/{key}'] = value
    
    return test_metrics

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

    print(f"num training batches {len(train_loader)}")
    print(f"num training datapoints {len(train_loader.dataset)}")
    print(f"num test batches {len(test_loader)}")
    print(f"num tes datapoints {len(test_loader.dataset)}")
    
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
    
    print("Starting training...")
    best_test_acc = 0.0
    metrics_calculator.reset_timing()
    
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            warmup_epochs, warmup_scheduler, main_scheduler, metrics_calculator
        )
        
        # Evaluate
        test_metrics = evaluate(
            model, test_loader, criterion, device, metrics_calculator, epoch
        )
        
        # Combine metrics and add epoch info
        combined_metrics = {
            'epoch': epoch + 1,
            'epoch_time': time.time() - epoch_start_time,
            **train_metrics,
            **test_metrics
        }
        
        # Log all metrics
        wandb.log(combined_metrics)
        
        # Extract key metrics for printing and model saving
        train_loss = train_metrics['train/loss']
        test_loss = test_metrics['test/loss']
        train_acc = train_metrics['train/top1_acc']  # Updated to match new key
        test_acc = test_metrics['test/top1_acc']     # Updated to match new key
        
        # Print epoch summary
        print('\nEpoch Summary:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss:  {test_loss:.3f} | Test Acc:  {test_acc:.2f}%')
        print(f'Test Top-5 Acc: {test_metrics["test/top5_acc"]:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Only print top-5 acc if available
        if 'test/top5_acc' in test_metrics:
            print(f'Test Top-5 Acc: {test_metrics["test/top5_acc"]:.2f}%')
        
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model based on test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"New best model! (Test Acc: {test_acc:.2f}%)")
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'num_classes': num_classes
            }

            # Save best model
            model_name = f"fontCNN_BS{args.batch_size}-ED{args.embedding_dim}-IC{args.initial_channels}-{time.strftime('%Y-%m-%d_%H-%M')}.pt"
            torch.save(best_model_state, model_name)
        
        # Update wandb summary periodically
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            wandb.run.summary.update({
                "best_test_acc": best_test_acc,
                "final_train_loss": train_loss,
                "final_test_loss": test_loss,
                "epochs_to_best": epoch + 1,
                "train_test_gap": abs(train_loss - test_loss),
                "final_train_test_acc_gap": abs(train_acc - test_acc)
            })
    
    print("\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print("Calculating class embeddings for each font using the training data")

    # class_embeddings = compute_class_embeddings(
    #     model, train_loader, num_classes, device
    # )
    
    # # Add embeddings to the best model state and save
    # best_model_state['class_embeddings'] = class_embeddings
    # torch.save(best_model_state, 'best_model.pt')


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
