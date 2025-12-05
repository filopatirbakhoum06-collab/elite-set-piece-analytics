"""
First Receiver Prediction Model for Elite Set-Piece Analytics.
نموذج التنبؤ بالمستلم الأول

This module implements a machine learning model to predict the most likely
first receiver of a set-piece delivery, using an ensemble of XGBoost
and optional neural network models.

Example usage:
    >>> from src.models.receiver_predictor import ReceiverPredictor
    >>> predictor = ReceiverPredictor()
    >>> predictor.fit(X_train, y_train)
    >>> predictions = predictor.predict(X_test)
    >>> probabilities = predictor.predict_proba(X_test)
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from ..utils.config import Config
from ..utils.metrics import Metrics

logger = logging.getLogger(__name__)


class ReceiverPredictor:
    """
    First receiver prediction model for set-piece analytics.
    نموذج التنبؤ بالمستلم الأول للكرات الثابتة
    
    This model predicts which player is most likely to receive the ball
    first from a set-piece delivery (corner kick, free kick, etc.).
    
    The model uses an ensemble approach combining:
    - XGBoost classifier for gradient boosting
    - Optional neural network for deep learning predictions
    
    Attributes:
        n_classes: Number of unique receiver classes
        feature_names: List of feature column names
        
    Example:
        >>> predictor = ReceiverPredictor()
        >>> predictor.fit(X_train, y_train)
        >>> probs = predictor.predict_proba(X_test)
        >>> top_k = predictor.get_top_k_receivers(X_test, k=3)
    """
    
    def __init__(
        self,
        xgb_params: Optional[Dict[str, Any]] = None,
        use_neural_network: bool = False,
        ensemble_weights: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize the receiver predictor.
        
        Args:
            xgb_params: XGBoost hyperparameters (uses defaults if None)
            use_neural_network: Whether to use neural network in ensemble
            ensemble_weights: Weights for (xgb, nn) predictions
        """
        self.config = Config()
        
        # Model parameters
        self.xgb_params = xgb_params or self.config.XGBOOST_PARAMS.copy()
        self.use_neural_network = use_neural_network
        self.ensemble_weights = ensemble_weights or (0.7, 0.3) if use_neural_network else (1.0, 0.0)
        
        # Preprocessors
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Models
        self.xgb_model: Optional[XGBClassifier] = None
        self.nn_model = None  # Optional neural network
        
        # State
        self.is_fitted = False
        self.n_classes: int = 0
        self.feature_names: List[str] = []
        self.classes_: Optional[np.ndarray] = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple] = None,
        verbose: bool = True
    ) -> 'ReceiverPredictor':
        """
        Fit the receiver prediction model.
        تدريب نموذج التنبؤ
        
        Args:
            X: Training features
            y: Training labels (receiver identifiers)
            eval_set: Optional validation set (X_val, y_val)
            verbose: Whether to print training progress
            
        Returns:
            self: The fitted model
        """
        self.logger.info("Fitting ReceiverPredictor...")
        
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.n_classes = len(self.label_encoder.classes_)
        self.classes_ = self.label_encoder.classes_
        
        self.logger.info(f"Number of classes (receivers): {self.n_classes}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit XGBoost model
        self.xgb_model = XGBClassifier(
            **self.xgb_params,
            objective='multi:softprob',
            num_class=self.n_classes,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # Prepare evaluation set if provided
        xgb_eval_set = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            X_val_scaled = self.scaler.transform(X_val)
            y_val_encoded = self.label_encoder.transform(y_val)
            xgb_eval_set = [(X_val_scaled, y_val_encoded)]
        
        # Fit XGBoost
        self.xgb_model.fit(
            X_scaled, y_encoded,
            eval_set=xgb_eval_set,
            verbose=verbose
        )
        
        # Fit neural network if enabled
        if self.use_neural_network:
            self._fit_neural_network(X_scaled, y_encoded)
        
        self.is_fitted = True
        self.logger.info("ReceiverPredictor fitting complete.")
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict the most likely receiver.
        التنبؤ بالمستلم الأكثر احتمالاً
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predicted receiver identifiers
        """
        self._check_fitted()
        
        proba = self.predict_proba(X)
        predicted_indices = np.argmax(proba, axis=1)
        
        return self.label_encoder.inverse_transform(predicted_indices)
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict probability distribution over receivers.
        التنبؤ باحتمالات جميع المستلمين
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        self._check_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        # XGBoost predictions
        xgb_proba = self.xgb_model.predict_proba(X_scaled)
        
        # Combine with neural network if enabled
        if self.use_neural_network and self.nn_model is not None:
            nn_proba = self._predict_neural_network(X_scaled)
            proba = (
                self.ensemble_weights[0] * xgb_proba +
                self.ensemble_weights[1] * nn_proba
            )
        else:
            proba = xgb_proba
        
        return proba
    
    def get_top_k_receivers(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        k: int = 3
    ) -> List[List[Tuple[Any, float]]]:
        """
        Get the top-k most likely receivers with probabilities.
        الحصول على أعلى k مستلمين محتملين
        
        Args:
            X: Features to predict on
            k: Number of top receivers to return
            
        Returns:
            List of lists, each containing (receiver_id, probability) tuples
        """
        proba = self.predict_proba(X)
        
        results = []
        for sample_proba in proba:
            top_indices = np.argsort(sample_proba)[-k:][::-1]
            top_receivers = [
                (self.label_encoder.inverse_transform([idx])[0], sample_proba[idx])
                for idx in top_indices
            ]
            results.append(top_receivers)
        
        return results
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        تقييم أداء النموذج
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        self._check_fitted()
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        y_encoded = self.label_encoder.transform(y)
        
        metrics = Metrics.calculate_all(y_encoded, 
                                        self.label_encoder.transform(y_pred), 
                                        y_proba)
        
        # Add top-k accuracy
        metrics['top_3_accuracy'] = Metrics.top_k_accuracy(y_encoded, y_proba, k=3)
        metrics['top_5_accuracy'] = Metrics.top_k_accuracy(y_encoded, y_proba, k=5)
        
        return metrics
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        التحقق المتقاطع
        
        Args:
            X: Features
            y: Labels
            n_folds: Number of folds
            
        Returns:
            Dictionary with cross-validation results
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        # Temporary model for CV
        cv_model = XGBClassifier(
            **self.xgb_params,
            objective='multi:softprob',
            num_class=len(np.unique(y_encoded)),
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                            random_state=self.config.RANDOM_SEED)
        
        scores = cross_val_score(cv_model, X_scaled, y_encoded, 
                                cv=cv, scoring='accuracy')
        
        return {
            'mean_accuracy': float(scores.mean()),
            'std_accuracy': float(scores.std()),
            'fold_scores': scores.tolist()
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        الحصول على أهمية الميزات
        
        Returns:
            DataFrame with feature importance rankings
        """
        self._check_fitted()
        
        importance = self.xgb_model.feature_importances_
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Args:
            path: File path for saving
        """
        self._check_fitted()
        
        model_data = {
            'xgb_model': self.xgb_model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'n_classes': self.n_classes,
            'feature_names': self.feature_names,
            'ensemble_weights': self.ensemble_weights,
            'xgb_params': self.xgb_params,
        }
        
        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ReceiverPredictor':
        """
        Load a model from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded ReceiverPredictor instance
        """
        model_data = joblib.load(path)
        
        predictor = cls(
            xgb_params=model_data['xgb_params'],
            ensemble_weights=model_data['ensemble_weights']
        )
        
        predictor.xgb_model = model_data['xgb_model']
        predictor.label_encoder = model_data['label_encoder']
        predictor.scaler = model_data['scaler']
        predictor.n_classes = model_data['n_classes']
        predictor.feature_names = model_data['feature_names']
        predictor.classes_ = predictor.label_encoder.classes_
        predictor.is_fitted = True
        
        return predictor
    
    def _check_fitted(self) -> None:
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
    
    def _fit_neural_network(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """Fit optional neural network component."""
        # Placeholder for neural network implementation
        # Can be extended with TensorFlow/PyTorch
        self.logger.info("Neural network training not implemented yet")
        self.nn_model = None
    
    def _predict_neural_network(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """Get predictions from neural network."""
        if self.nn_model is None:
            return np.zeros((X.shape[0], self.n_classes))
        
        # Placeholder for neural network prediction
        return np.zeros((X.shape[0], self.n_classes))


def create_training_data(
    events: pd.DataFrame,
    features: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create training data from events and features.
    
    Args:
        events: Set-piece events with receiver information
        features: Pre-computed features
        
    Returns:
        Tuple of (X, y) for training
    """
    # Merge features with events
    if 'event_id' in events.columns and 'event_id' in features.columns:
        data = events.merge(features, on='event_id')
    else:
        data = pd.concat([events.reset_index(drop=True), 
                         features.reset_index(drop=True)], axis=1)
    
    # Identify target column
    target_col = 'first_receiver' if 'first_receiver' in data.columns else 'receiver_id'
    
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Separate features and target
    feature_cols = [col for col in data.columns 
                   if col not in ['event_id', 'match_id', target_col, 'set_piece_type']]
    
    X = data[feature_cols]
    y = data[target_col]
    
    return X, y
