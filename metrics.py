import torch
import torch.nn.functional as F
from typing import Dict, Optional


class ClassificationMetrics:
    """Utility class for computing comprehensive classification metrics."""
    
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device
        self.prev_predictions = None
        
    def compute_all_metrics(self, 
                          logits: torch.Tensor, 
                          targets: torch.Tensor,
                          model: Optional[torch.nn.Module] = None,
                          epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Compute all classification metrics in one pass.
        
        Args:
            logits: Raw model outputs (pre-softmax)
            targets: Ground truth labels
            model: Optional model for computing gradient statistics
            epoch: Optional epoch number for tracking
            
        Returns:
            Dictionary containing all computed metrics
        """
        probs = F.softmax(logits, dim=1)
        top1_pred = torch.argmax(logits, dim=1)
        
        metrics = {}
        
        # Core metrics
        metrics.update(self._compute_topk_accuracy(logits, targets))
        metrics.update(self._compute_per_class_metrics(top1_pred, targets))
        metrics.update(self._compute_distribution_metrics(probs, targets))
        metrics.update(self._compute_reliability_metrics(probs, targets, top1_pred))
        metrics.update(self._compute_map(probs, targets))
        
        # Training dynamics metrics
        metrics.update(self._compute_logit_stats(logits))
        if self.prev_predictions is not None:
            metrics.update(self._compute_prediction_churn(top1_pred))
        self.prev_predictions = top1_pred.clone()
        
        if model is not None and any(p.grad is not None for p in model.parameters()):
            metrics.update(self._compute_gradient_stats(model))
        
        if epoch is not None:
            metrics["epoch"] = epoch
            
        return metrics
    
    def _compute_topk_accuracy(self, 
                             logits: torch.Tensor, 
                             targets: torch.Tensor) -> Dict[str, float]:
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
        
        # Only compute statistics for classes that appeared in this batch
        valid_acc = per_class_acc[valid_classes]
        
        if len(valid_acc) > 0:
            metrics = {
                'mean_per_class_acc': valid_acc.mean().item() * 100,
                'median_acc': valid_acc.median().item() * 100,
            }
            
            # Only compute worst5 and best5 if we have enough classes
            if len(valid_acc) >= 5:
                metrics.update({
                    'worst5_acc': valid_acc.topk(5, largest=False)[0].mean().item() * 100,
                    'best5_acc': valid_acc.topk(5, largest=True)[0].mean().item() * 100,
                })
        else:
            metrics = {
                'mean_per_class_acc': 0.0,
                'median_acc': 0.0,
                'acc_std': 0.0,
                'worst5_acc': 0.0,
                'best5_acc': 0.0,
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
        
        metrics['macro_f1'] = torch.stack(f1_scores).mean().item()
        
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
        
        bin_boundaries = torch.linspace(0, 1, 16, device=self.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = torch.zeros(1, device=self.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower) & confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # avg_confidence = confidences.mean().item()
        # accuracy = accuracies.float().mean().item()
        
        metrics.update({
            'ece': ece.item(),
            # 'avg_confidence': avg_confidence,
            # 'confidence_gap': abs(avg_confidence - accuracy)
        })
        
        return metrics
    
    def _compute_map(self, probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute Mean Average Precision."""
        target_onehot = torch.zeros_like(probs)
        target_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        sorted_probs, sort_indices = torch.sort(probs, dim=0, descending=True)
        sorted_targets = target_onehot[sort_indices, torch.arange(self.num_classes).unsqueeze(0)]
        
        tp_cumsum = torch.cumsum(sorted_targets, dim=0)
        positions = torch.arange(1, len(sorted_targets) + 1, device=self.device).float()
        precisions = tp_cumsum / positions.unsqueeze(1)
        
        relevant_positions = sorted_targets == 1
        ap_per_class = (precisions * relevant_positions).sum(0) / (relevant_positions.sum(0) + 1e-12)
        mean_ap = ap_per_class.mean().item()
        
        return {'mean_ap': mean_ap}
    
    def _compute_logit_stats(self, logits: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about the model's logit outputs."""
        metrics = {}
        
        logit_mean = logits.mean().item()

        # Calculate percentage of very negative logits (potential dead units)
        dead_logits = (logits < -10).float().mean().item() * 100
        logit_norm = torch.norm(logits, dim=1).mean().item()
        
        metrics.update({
            'logit_mean': logit_mean,
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
            
            return {
                'grad_norm': total_norm,
                'param_norm': param_norm,
                'grad_to_param_ratio': total_norm / (param_norm + 1e-12)
            }
        
        return {}