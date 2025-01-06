import torch
import torch.nn.functional as F
from typing import Dict, Optional
import time


class ClassificationMetrics:
    """Unified metrics tracking for both training and evaluation."""
    
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device
        self.prev_predictions = None
        self.training_start_time = time.time()
        self.batch_start_time = time.time()
        
    def start_batch(self):
        """Mark the start of a new batch for timing purposes."""
        self.batch_start_time = time.time()
        return self.batch_start_time

    def compute_batch_metrics(self, 
                            logits: torch.Tensor,
                            targets: torch.Tensor,
                            loss: float,
                            batch_size: int,
                            optimizer: Optional[torch.optim.Optimizer] = None,
                            batch_times: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute metrics for training batches.
        
        Args:
            logits: Model outputs
            targets: Ground truth labels
            loss: Loss value for this batch
            batch_size: Number of samples in batch
            optimizer: Optional optimizer for learning rate tracking
            batch_times: Optional dict with timing info (data_transfer, forward_backward, optimizer)
        """
        metrics = {}
        
        # Basic batch metrics
        predictions = torch.argmax(logits, dim=1)
        correct = predictions.eq(targets).sum().item()
        accuracy = 100. * correct / batch_size
        
        # Timing metrics
        batch_time = time.time() - self.batch_start_time
        samples_per_second = batch_size / batch_time
        total_training_time = time.time() - self.training_start_time
        
        metrics.update({
            'batch_loss': loss,
            'batch_acc': accuracy,
            'batch_time': batch_time,
            #'samples_per_second': samples_per_second,
            'total_training_time': total_training_time
        })
        
        # Add detailed timing breakdown if provided
        # if batch_times:
        #     metrics.update({
        #         'data_transfer_time': batch_times.get('data_transfer', 0),
        #         'forward_backward_time': batch_times.get('forward_backward', 0),
        #         'optimizer_time': batch_times.get('optimizer', 0)
        #     })
        
        # Add learning rate if optimizer provided
        if optimizer:
            metrics['learning_rate'] = optimizer.param_groups[0]['lr']
            
        return metrics

    def compute_all_metrics(self, 
                          logits: torch.Tensor, 
                          targets: torch.Tensor,
                          model: Optional[torch.nn.Module] = None,
                          epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Compute comprehensive metrics for evaluation.
        
        Args:
            logits: Raw model outputs (pre-softmax)
            targets: Ground truth labels
            model: Optional model for computing gradient statistics
            epoch: Optional epoch number for tracking
        """
        probs = F.softmax(logits, dim=1)
        top1_pred = torch.argmax(logits, dim=1)
        
        metrics = {}
        
        # Core metrics
        metrics.update(self._compute_topk_accuracy(logits, targets))
        metrics.update(self._compute_per_class_metrics(top1_pred, targets))
        metrics.update(self._compute_distribution_metrics(probs, targets))
        metrics.update(self._compute_reliability_metrics(probs, targets, top1_pred))
        # metrics.update(self._compute_map(probs, targets))
        metrics.update(self._compute_logit_stats(logits))
        
        # Training dynamics metrics
        if self.prev_predictions is not None:
            metrics.update(self._compute_prediction_churn(top1_pred))
        self.prev_predictions = top1_pred.clone()
        
        # Add gradient stats if model is provided
        if model is not None and any(p.grad is not None for p in model.parameters()):
            metrics.update(self._compute_gradient_stats(model))
        
        if epoch is not None:
            metrics['epoch'] = epoch
            
        return metrics

    def _compute_logit_stats(self, logits: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about the model's logit outputs."""
        metrics = {}
        
        # Measure logit range and spread
        logit_mean = logits.mean().item()
        logit_std = logits.std().item()
        logit_max = logits.max().item()
        logit_min = logits.min().item()
        
        # Calculate dead logit percentage (logits that are always very negative)
        dead_logits = (logits < -10).float().mean().item() * 100
        
        # Measure logit stability
        logit_norm = torch.norm(logits, dim=1).mean().item()
        
        metrics.update({
            'logit_mean': logit_mean,
            'logit_std': logit_std,
            'logit_range': logit_max - logit_min,
            'dead_logits_pct': dead_logits,
            'logit_norm': logit_norm
        })
        
        return metrics
    
    def _compute_prediction_churn(self, current_preds: torch.Tensor) -> Dict[str, float]:
        """Compute prediction stability between epochs."""
        prediction_churn = (current_preds != self.prev_predictions).float().mean().item() * 100
        return {'prediction_churn': prediction_churn}
    
    def _compute_gradient_stats(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute gradient statistics."""
        metrics = {}
        
        total_norm = 0.0
        param_norm = 0.0
        grad_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm += param.data.norm().item() ** 2
                grad_norm = param.grad.data.norm().item() ** 2
                total_norm += grad_norm
                grad_count += 1
        
        if grad_count > 0:
            total_norm = (total_norm / grad_count) ** 0.5
            param_norm = (param_norm / grad_count) ** 0.5
            
            metrics.update({
                'grad_norm': total_norm,
                'param_norm': param_norm,
                'grad_to_param_ratio': total_norm / (param_norm + 1e-12)
            })
        
        return metrics
    
    def _compute_topk_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute top-k accuracy metrics."""
        metrics = {}
        topk = (1, 5)
        maxk = max(topk)
        
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            metrics[f'top{k}_acc'] = (correct_k * 100.0 / targets.size(0)).item()
            
        return metrics
    
    def _compute_per_class_metrics(self,
                                predictions: torch.Tensor,
                                targets: torch.Tensor) -> Dict[str, float]:
        """Compute per-class accuracy metrics and summary statistics."""
        # Initialize per-class correct predictions and counts
        per_class_correct = torch.zeros(self.num_classes, device=self.device)
        per_class_total = torch.zeros(self.num_classes, device=self.device)
        
        for cls in range(self.num_classes):
            cls_mask = targets == cls
            per_class_correct[cls] = (predictions[cls_mask] == cls).sum()
            per_class_total[cls] = cls_mask.sum()
        
        # Compute per-class accuracies (avoiding division by zero)
        valid_classes = per_class_total > 0
        per_class_acc = torch.zeros(self.num_classes, device=self.device)
        per_class_acc[valid_classes] = per_class_correct[valid_classes] / per_class_total[valid_classes]
        
        # Compute summary statistics
        metrics = {
            'mean_per_class_acc': per_class_acc[valid_classes].mean().item() * 100,
            'worst5_acc': per_class_acc[valid_classes].topk(5, largest=False)[0].mean().item() * 100,
            'best5_acc': per_class_acc[valid_classes].topk(5, largest=True)[0].mean().item() * 100,
            #'acc_std': per_class_acc[valid_classes].std().item() * 100,
            'median_acc': per_class_acc[valid_classes].median().item() * 100
        }
        
        return metrics
    
    def _compute_distribution_metrics(self,
                                   probs: torch.Tensor,
                                   targets: torch.Tensor) -> Dict[str, float]:
        """Compute metrics related to prediction distribution."""
        metrics = {}
        
        # Prediction entropy (uncertainty metric)
        pred_entropy = (-probs * torch.log(probs + 1e-12)).sum(1).mean()
        metrics['pred_entropy'] = pred_entropy.item()
        
        # Compute macro F1 score
        f1_scores = []
        for cls in range(self.num_classes):
            cls_preds = (torch.argmax(probs, dim=1) == cls)
            cls_targets = (targets == cls)
            tp = (cls_preds & cls_targets).sum().float()
            fp = (cls_preds & ~cls_targets).sum().float()
            fn = (~cls_preds & cls_targets).sum().float()
            
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
            f1_scores.append(f1)
        
        metrics['macro_f1'] = torch.tensor(f1_scores).mean().item()
        
        return metrics
    
    def _compute_reliability_metrics(self,
                                  probs: torch.Tensor,
                                  targets: torch.Tensor,
                                  predictions: torch.Tensor) -> Dict[str, float]:
        """Compute reliability and calibration metrics."""
        metrics = {}
        
        # Expected Calibration Error (ECE)
        confidences, _ = torch.max(probs, dim=1)
        accuracies = predictions.eq(targets)
        
        # Use 15 bins for ECE calculation
        bin_boundaries = torch.linspace(0, 1, 16)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = torch.zeros(1, device=self.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        metrics['ece'] = ece.item()
        
        # Average confidence and its gap with accuracy
        avg_confidence = confidences.mean().item()
        accuracy = accuracies.float().mean().item()
        metrics['avg_confidence'] = avg_confidence
        metrics['confidence_gap'] = abs(avg_confidence - accuracy)
        
        return metrics

    def _compute_map(self, probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute Mean Average Precision using PyTorch."""
        # Convert targets to one-hot encoding
        target_onehot = torch.zeros_like(probs)
        target_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Sort probabilities in descending order
        sorted_probs, sort_indices = torch.sort(probs, dim=0, descending=True)
        sorted_targets = target_onehot[sort_indices, torch.arange(self.num_classes).unsqueeze(0)]
        
        # Calculate cumulative TP
        tp_cumsum = torch.cumsum(sorted_targets, dim=0)
        
        # Calculate precision at each position
        positions = torch.arange(1, len(sorted_targets) + 1, device=self.device).float()
        precisions = tp_cumsum / positions.unsqueeze(1)
        
        # Only consider positions where there are true positives
        relevant_positions = sorted_targets == 1
        ap_per_class = (precisions * relevant_positions).sum(0) / (relevant_positions.sum(0) + 1e-12)
        
        # Average over all classes
        mean_ap = ap_per_class.mean().item()
        
        return {'mean_ap': mean_ap}
    
    def get_progress_bar_stats(self, 
                             loss: float,
                             logits: torch.Tensor,
                             targets: torch.Tensor,
                             batch_size: int) -> Dict[str, str]:
        """Get formatted statistics for tqdm progress bar."""
        batch_time = time.time() - self.batch_start_time
        correct = torch.argmax(logits, dim=1).eq(targets).sum().item()
        
        return {
            'loss': f'{loss:.3f}',
            'acc': f'{100. * correct / batch_size:.2f}%',
            'batch_time': f'{batch_time:.3f}s',
            'samples/sec': f'{batch_size / batch_time:.1f}',
            'total_time': f'{time.time() - self.training_start_time:.0f}s'
        }

    def reset_timing(self):
        """Reset timing counters."""
        self.training_start_time = time.time()
        self.batch_start_time = time.time()