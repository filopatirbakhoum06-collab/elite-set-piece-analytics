"""
Metrics module for Elite Set-Piece Analytics.
وحدة المقاييس

This module provides evaluation metrics for model performance assessment.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss,
)


class Metrics:
    """
    Evaluation metrics calculator for set-piece prediction models.
    
    Provides comprehensive metrics for:
    - Classification performance
    - Probability calibration
    - Model comparison
    
    Example:
        >>> metrics = Metrics()
        >>> results = metrics.calculate_all(y_true, y_pred, y_proba)
        >>> print(results['accuracy'])
    """
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        return float(accuracy_score(y_true, y_pred))
    
    @staticmethod
    def calculate_precision(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        average: str = "weighted"
    ) -> float:
        """Calculate precision score."""
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def calculate_recall(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        average: str = "weighted"
    ) -> float:
        """Calculate recall score."""
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def calculate_f1(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        average: str = "weighted"
    ) -> float:
        """Calculate F1 score."""
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def calculate_auc_roc(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        multi_class: str = "ovr"
    ) -> float:
        """
        Calculate AUC-ROC score.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            multi_class: Strategy for multi-class ('ovr' or 'ovo')
            
        Returns:
            AUC-ROC score
        """
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_proba.ndim == 2:
                    y_proba = y_proba[:, 1]
                return float(roc_auc_score(y_true, y_proba))
            else:
                # Multi-class classification
                return float(roc_auc_score(
                    y_true, y_proba, multi_class=multi_class, average="weighted"
                ))
        except ValueError:
            return 0.0
    
    @staticmethod
    def calculate_log_loss(
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> float:
        """Calculate log loss (cross-entropy loss)."""
        try:
            return float(log_loss(y_true, y_proba))
        except ValueError:
            return float('inf')
    
    @staticmethod
    def calculate_confusion_matrix(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Generate a classification report."""
        return classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
    
    @classmethod
    def calculate_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary containing all metrics
        """
        results = {
            "accuracy": cls.calculate_accuracy(y_true, y_pred),
            "precision": cls.calculate_precision(y_true, y_pred, average),
            "recall": cls.calculate_recall(y_true, y_pred, average),
            "f1_score": cls.calculate_f1(y_true, y_pred, average),
            "confusion_matrix": cls.calculate_confusion_matrix(y_true, y_pred),
        }
        
        if y_proba is not None:
            results["auc_roc"] = cls.calculate_auc_roc(y_true, y_proba)
            results["log_loss"] = cls.calculate_log_loss(y_true, y_proba)
        
        return results
    
    @staticmethod
    def top_k_accuracy(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        k: int = 3
    ) -> float:
        """
        Calculate top-k accuracy.
        
        Useful for first receiver prediction where the top-k candidates matter.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities (n_samples, n_classes)
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy score
        """
        if y_proba.ndim == 1:
            return float(accuracy_score(y_true, (y_proba > 0.5).astype(int)))
        
        # Get indices of top-k predictions for each sample
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        
        # Check if true label is in top-k predictions
        correct = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
        
        return float(np.mean(correct))
