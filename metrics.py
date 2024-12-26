import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List

class ClassificationMetrics:
    """Utility class for computing comprehensive classification metrics."""
    
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device
        
    def compute_all_metrics(self, 
                          logits: torch.Tensor, 
                          targets: torch.Tensor,
                          epoch: int = None) -> Dict[str, float]:
        """Compute all classification metrics in one pass."""
        
        # Get predictions and probabilities
        probs = F.softmax(logits, dim=1)
        top1_pred = torch.argmax(logits, dim=1)
        
        # Basic accuracy metrics
        metrics = {}
        metrics.update(self._compute_topk_accuracy(logits, targets))
        metrics.update(self._compute_per_class_metrics(top1_pred, targets))
        metrics.update(self._compute_distribution_metrics(probs, targets))
        metrics.update(self._compute_reliability_metrics(probs, targets, top1_pred))
        
        if epoch is not None:
            # Add epoch-specific metrics
            metrics["epoch"] = epoch
            
        return metrics
    
    def _compute_topk_accuracy(self, 
                             logits: torch.Tensor, 
                             targets: torch.Tensor) -> Dict[str, float]:
        """Compute top-k accuracy metrics."""
        metrics = {}
        
        # For 1000+ classes, we typically care about top-1 and top-5
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
        
        # Compute per-class accuracies
        valid_classes = per_class_total > 0
        per_class_acc = torch.zeros(self.num_classes, device=self.device)
        per_class_acc[valid_classes] = per_class_correct[valid_classes] / per_class_total[valid_classes]
        
        # Compute summary statistics
        metrics = {
            'mean_per_class_acc': per_class_acc[valid_classes].mean().item() * 100,
            'worst5_acc': per_class_acc[valid_classes].topk(5, largest=False)[0].mean().item() * 100,
            'best5_acc': per_class_acc[valid_classes].topk(5, largest=True)[0].mean().item() * 100,
            'acc_std': per_class_acc[valid_classes].std().item() * 100,
            'median_acc': per_class_acc[valid_classes].median().item() * 100,
            'empty_classes': (~valid_classes).sum().item()
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