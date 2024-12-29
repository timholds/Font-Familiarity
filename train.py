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
from torch.optim import AdamW  # Consider using AdamW instead of Adam
import numpy as np

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

def train_epoch(model, train_loader, criterion, optimizer,
                 device, epoch, warmup_epochs, warmup_scheduler,
                main_scheduler, train_start_time):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Add timing metrics
    batch_start_time = time.time()
    batch_times = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        # Time the data transfer
        data_transfer_start = time.time()
        data, target = data.to(device), target.to(device)
        data_transfer_time = time.time() - data_transfer_start
        
        # Time the forward/backward pass
        forward_start = time.time()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        forward_backward_time = time.time() - forward_start
        
        # Time the optimizer step
        optimizer_start = time.time()
        optimizer.step()
        optimizer_time = time.time() - optimizer_start
        
        # Calculate total batch time
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        # Regular training metrics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        batch_correct = predicted.eq(target).sum().item()
        correct += batch_correct
        
        # Log batch timing metrics (every N batches to reduce noise)
        if batch_idx % 50 == 0:  # Log every 50 batches
            global_step = batch_idx + epoch * len(train_loader)
            wandb.log({
                "step": global_step, 
                #"samples_per_second": target.size(0) / batch_time,
                "learning_rate": optimizer.param_groups[0]['lr'],
                #"batch": batch_idx + epoch * len(train_loader),
                "total_training_time(s)": time.time() - train_start_time  # Add this as global var
            })
            batch_start_time = time.time()  # Reset for next batch
            # wandb.log({
            #     "batch_time": batch_time,
            #     "data_transfer_time": data_transfer_time,
            #     "forward_backward_time": forward_backward_time,
            #     "optimizer_time": optimizer_time,
            #     "samples_per_second": target.size(0) / batch_time,
            #     "global_step": batch_idx + len(train_loader) * epoch
            # })
        
        # Update progress bar with timing info
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100. * correct / total:.2f}%',
            'batch_time': f'{batch_time:.3f}s',
            'samples/sec': f'{target.size(0) / batch_time:.1f}',
            "total_training_time": time.time() - train_start_time  # Add this as global var

        })
        
        batch_start_time = time.time()  # Reset for next batch
        
    # Log epoch-level timing statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    # wandb.log({
    #     "avg_batch_time": avg_batch_time,
    #     # "min_batch_time": min(batch_times),
    #     # "max_batch_time": max(batch_times),
    #     # "batch_time_std": np.std(batch_times),
    #     "total_epoch_time": sum(batch_times)
    # })
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device, metrics_calculator):
    """
    Evaluate the model on the test set.
    Returns test loss and all classification metrics.
    """
    model.eval()
    test_loss = 0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            
            # Calculate loss
            test_loss += criterion(logits, target).item()
            
            # Store predictions and targets for metric calculation
            all_logits.append(logits)
            all_targets.append(target)
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    # Calculate average loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Compute all metrics
    metrics = metrics_calculator.compute_all_metrics(all_logits, all_targets)
    
    # Add loss to metrics dictionary
    metrics['test_loss'] = avg_test_loss
    
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
    
    # Calculate warmup_epochs before wandb init since we need it in the config
    warmup_epochs = max(args.epochs // 5, 1)

    # Initialize wandb and get config right after parsing args
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
    wandb.define_metric("batch_loss", step_metric="batch")
    wandb.define_metric("batch_acc", step_metric="batch")
    wandb.define_metric("*", step_metric="epoch")

    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Now use config instead of args for values that might be swept
    print("Loading data...")
    train_loader, test_loader, num_classes = get_dataloaders(
        data_dir=args.data_dir,  # not swept
        batch_size=config.batch_size  # swept
    )

    print(f"Initializing model (num_classes={num_classes})...")
    model = SimpleCNN(
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        input_size=config.resolution,
        initial_channels=config.initial_channels
    ).to(device)

    # Watch model right after it's created
    wandb.watch(model, log_freq=100)
    metrics_calculator = ClassificationMetrics(num_classes=num_classes, device=device)


    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
   
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs * len(train_loader)
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(config.epochs - warmup_epochs) * len(train_loader),  # Total number of steps
        eta_min=1e-6
    )
   
    # Training loop
    print("Starting training...")
    best_test_acc = 0.0
    train_start_time = time.time()
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            warmup_epochs, warmup_scheduler, main_scheduler, train_start_time  # Pass schedulers
        )

        # Step the appropriate scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        # Evaluate
        model, test_loader, criterion, device, metrics_calculator
        metrics = evaluate(
            model, test_loader, criterion, device, metrics_calculator
        )
        
        epoch_time = time.time() - start_time

        wandb.log(metrics)
        test_loss = metrics['test_loss']
        test_acc = metrics['top1_acc']

        global_step = (epoch + 1) * len(train_loader) 
        
        # Log metrics to wandb
        metrics_dict = {
            "step": global_step,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time(s)": epoch_time,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
                
        #class_accs = per_class_acc * 100  # Convert to percentages
        # In your evaluate() function, ensure this calculation is correct:
        # metrics_dict.update({
        #     "class_acc_worst5": torch.mean(torch.topk(class_accs, k=5, largest=False)[0]).item(),
        #     "class_acc_best5": torch.mean(torch.topk(class_accs, k=5, largest=True)[0]).item(),  # Make sure this is logging
        #     "class_acc_spread": class_accs.std().item()
        # })
    
        wandb.log(metrics_dict)
      
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Time taken: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
        #print(metrics_report)
        
        # Save if best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"New best model! (Test Acc: {test_acc:.2f}%)")
            # Store the model state without computing embeddings yet
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'num_classes': num_classes
            }

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:  # Every 5 epochs and last epoch
            wandb.run.summary.update({
                "best_test_acc": best_test_acc,
                "final_train_loss": train_loss,
                "epochs_to_best": epoch + 1,
                "throughput_avg": metrics_dict.get("samples_per_second", 0),
                "train_test_gap": abs(train_acc - test_acc)  # Track potential overfitting
            })

    print("\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    
    # Now compute class embeddings once at the end
    print("Computing final class embeddings...")
    class_embeddings = compute_class_embeddings(
        model, train_loader, num_classes, device
    )
    
    # Add embeddings to the best model state and save
    best_model_state['class_embeddings'] = class_embeddings
    torch.save(best_model_state, 'best_model.pt')

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
