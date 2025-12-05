"""
Outcome Prediction Model for Elite Set-Piece Analytics.
نموذج التنبؤ بالنتائج

This module implements models to predict the outcome of set-pieces,
including goal probability, shot probability, and chance quality.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor

from ..utils.config import Config
from ..utils.metrics import Metrics

logger = logging.getLogger(__name__)


class OutcomePredictor:
    """
    Outcome prediction model for set-pieces.
    نموذج التنبؤ بنتائج الكرات الثابتة
    
    Predicts various outcomes from set-piece situations:
    - Goal probability (xG)
    - Shot probability
    - Chance quality rating
    
    Example:
        >>> predictor = OutcomePredictor()
        >>> predictor.fit(X_train, y_train)
        >>> xg = predictor.predict_goal_probability(X_test)
    """
    
    OUTCOME_TYPES = ['goal', 'shot', 'chance_quality']
    
    def __init__(
        self,
        outcome_type: str = 'goal',
        xgb_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the outcome predictor.
        
        Args:
            outcome_type: Type of outcome to predict ('goal', 'shot', 'chance_quality')
            xgb_params: XGBoost hyperparameters
        """
        if outcome_type not in self.OUTCOME_TYPES:
            raise ValueError(f"outcome_type must be one of {self.OUTCOME_TYPES}")
        
        self.outcome_type = outcome_type
        self.config = Config()
        self.xgb_params = xgb_params or self.config.XGBOOST_PARAMS.copy()
        
        # Preprocessor
        self.scaler = StandardScaler()
        
        # Model - use classifier for goal/shot, regressor for chance_quality
        if outcome_type == 'chance_quality':
            self.model = XGBRegressor(**self.xgb_params)
        else:
            self.model = XGBClassifier(
                **self.xgb_params,
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        self.is_fitted = False
        self.feature_names: List[str] = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[tuple] = None,
        verbose: bool = True
    ) -> 'OutcomePredictor':
        """
        Fit the outcome prediction model.
        
        Args:
            X: Training features
            y: Training labels (0/1 for goal/shot, continuous for quality)
            eval_set: Optional validation set
            verbose: Whether to print progress
            
        Returns:
            self
        """
        self.logger.info(f"Fitting OutcomePredictor for {self.outcome_type}...")
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare eval set
        xgb_eval_set = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            X_val_scaled = self.scaler.transform(X_val)
            xgb_eval_set = [(X_val_scaled, y_val)]
        
        self.model.fit(X_scaled, y, eval_set=xgb_eval_set, verbose=verbose)
        
        self.is_fitted = True
        self.logger.info("OutcomePredictor fitting complete.")
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict outcomes.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted values
        """
        self._check_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict probability of positive outcome (goal/shot).
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability values
        """
        self._check_fitted()
        
        if self.outcome_type == 'chance_quality':
            # For regression, just return predictions
            return self.predict(X)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X_scaled)
        
        # Return probability of positive class
        return proba[:, 1]
    
    def predict_goal_probability(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict expected goals (xG) probability.
        
        Args:
            X: Features
            
        Returns:
            xG values (0-1)
        """
        if self.outcome_type != 'goal':
            raise ValueError("This method requires outcome_type='goal'")
        
        return self.predict_proba(X)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        self._check_fitted()
        
        predictions = self.predict(X)
        
        if self.outcome_type == 'chance_quality':
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            return {
                'mse': float(mean_squared_error(y, predictions)),
                'mae': float(mean_absolute_error(y, predictions)),
                'r2': float(r2_score(y, predictions)),
            }
        else:
            # Classification metrics
            proba = self.predict_proba(X)
            return Metrics.calculate_all(y, predictions, proba.reshape(-1, 1))
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        self._check_fitted()
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'outcome_type': self.outcome_type,
            'feature_names': self.feature_names,
            'xgb_params': self.xgb_params,
        }
        
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'OutcomePredictor':
        """Load model from disk."""
        model_data = joblib.load(path)
        
        predictor = cls(
            outcome_type=model_data['outcome_type'],
            xgb_params=model_data['xgb_params']
        )
        
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.is_fitted = True
        
        return predictor
    
    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
