"""
Elite Set-Piece Analytics - First Receiver Predictor
متنبئ المستلم الأول للكرات الثابتة

This module predicts which player will be the first receiver of a set piece.

Models:
- XGBoost Classifier (primary)
- Random Forest (baseline)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, top_k_accuracy_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class FirstReceiverPredictor:
    """
    Predict which player will be the first receiver of a set piece
    التنبؤ بأي لاعب سيكون المستلم الأول للكرة الثابتة
    
    Attributes:
        model: The trained classifier
        scaler: Feature scaler
        label_encoder: Label encoder for target classes
        feature_columns: List of feature column names
        is_fitted: Whether the model has been trained
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        n_estimators: int = 100,
        max_depth: int = 6,
        random_state: int = 42
    ):
        """
        Initialize the predictor
        تهيئة المتنبئ
        
        Args:
            model_type: 'xgboost' or 'random_forest'
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_fitted = False
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the underlying model"""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        else:
            if self.model_type == 'xgboost':
                print("⚠️ XGBoost not available, using Random Forest")
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features from set piece data
        تحضير الميزات من بيانات الكرات الثابتة
        
        Args:
            df: DataFrame with set piece data
        
        Returns:
            Tuple of (feature DataFrame, feature column names)
        """
        # Select numerical features
        numerical_cols = [
            'x', 'y', 'minute', 'score_diff',
            'n_attackers_in_box', 'n_defenders_in_box'
        ]
        
        # Add computed features if available
        optional_numerical = [
            'distance_to_goal', 'angle_to_goal', 'danger_index',
            'fatigue_factor', 'score_pressure', 'header_probability',
            'height_advantage', 'delivery_speed'
        ]
        
        for col in optional_numerical:
            if col in df.columns:
                numerical_cols.append(col)
        
        # Start with numerical features
        features = df[numerical_cols].copy()
        
        # Add one-hot encoded categorical features
        categorical_cols = ['type', 'side']
        if 'delivery_type' in df.columns:
            categorical_cols.append('delivery_type')
        
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col].fillna('unknown'), prefix=col)
                features = pd.concat([features, dummies], axis=1)
        
        # Fill any missing values
        features = features.fillna(0)
        
        return features, features.columns.tolist()
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model
        تدريب النموذج
        
        Args:
            X: Feature DataFrame
            y: Target Series (first receiver index/id)
            validation_split: Proportion for validation
        
        Returns:
            Dictionary with training metrics
        """
        print("🚀 Training First Receiver Predictor...")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y.astype(str))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded,
            test_size=validation_split,
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        print(f"  Number of classes: {len(self.label_encoder.classes_)}")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_f1_weighted': f1_score(y_val, y_val_pred, average='weighted'),
        }
        
        # Top-3 accuracy if more than 3 classes
        if len(self.label_encoder.classes_) > 3:
            y_val_proba = self.model.predict_proba(X_val)
            metrics['val_top3_accuracy'] = top_k_accuracy_score(y_val, y_val_proba, k=3)
        
        print(f"\n✅ Training complete!")
        print(f"   Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"   Val Accuracy: {metrics['val_accuracy']:.2%}")
        if 'val_top3_accuracy' in metrics:
            print(f"   Val Top-3 Accuracy: {metrics['val_top3_accuracy']:.2%}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict first receiver
        التنبؤ بالمستلم الأول
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predicted receiver indices
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        # Ensure correct features
        X_aligned = self._align_features(X)
        
        # Scale
        X_scaled = self.scaler.transform(X_aligned)
        
        # Predict
        y_pred_encoded = self.model.predict(X_scaled)
        
        # Decode
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for each receiver
        التنبؤ باحتمالات كل مستلم
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of probabilities (shape: n_samples x n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        X_aligned = self._align_features(X)
        X_scaled = self.scaler.transform(X_aligned)
        
        return self.model.predict_proba(X_scaled)
    
    def predict_top_k(self, X: pd.DataFrame, k: int = 3) -> List[List[Tuple[Any, float]]]:
        """
        Get top-k predictions with probabilities
        الحصول على أفضل k تنبؤات مع احتمالاتها
        
        Args:
            X: Feature DataFrame
            k: Number of top predictions
        
        Returns:
            List of lists of (receiver, probability) tuples
        """
        probas = self.predict_proba(X)
        classes = self.label_encoder.classes_
        
        results = []
        for row_proba in probas:
            top_indices = np.argsort(row_proba)[-k:][::-1]
            top_k = [(classes[idx], row_proba[idx]) for idx in top_indices]
            results.append(top_k)
        
        return results
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align input features with training features"""
        # Add missing columns with zeros
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Select only training columns in correct order
        return X[self.feature_columns]
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test data
        تقييم النموذج على بيانات الاختبار
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        y_pred = self.predict(X_test)
        
        # Encode for metrics
        y_test_encoded = self.label_encoder.transform(y_test.astype(str))
        y_pred_encoded = self.label_encoder.transform(y_pred.astype(str))
        
        metrics = {
            'accuracy': accuracy_score(y_test_encoded, y_pred_encoded),
            'f1_weighted': f1_score(y_test_encoded, y_pred_encoded, average='weighted'),
            'precision_weighted': precision_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
        }
        
        # Top-3 accuracy
        if len(self.label_encoder.classes_) > 3:
            y_proba = self.predict_proba(X_test)
            metrics['top3_accuracy'] = top_k_accuracy_score(y_test_encoded, y_proba, k=3)
        
        print("📊 Evaluation Results:")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        print(f"   F1 Score: {metrics['f1_weighted']:.3f}")
        if 'top3_accuracy' in metrics:
            print(f"   Top-3 Accuracy: {metrics['top3_accuracy']:.2%}")
        
        return metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation
        إجراء التحقق المتبادل
        
        Args:
            X: Features
            y: Targets
            cv: Number of folds
        
        Returns:
            Dictionary with CV metrics
        """
        print(f"🔄 Running {cv}-fold cross-validation...")
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y.astype(str))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validate
        scores = cross_val_score(
            self.model, X_scaled, y_encoded,
            cv=cv, scoring='accuracy'
        )
        
        metrics = {
            'cv_mean_accuracy': scores.mean(),
            'cv_std_accuracy': scores.std(),
            'cv_scores': scores.tolist()
        }
        
        print(f"   Mean Accuracy: {metrics['cv_mean_accuracy']:.2%} (±{metrics['cv_std_accuracy']:.2%})")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores
        الحصول على درجات أهمية الميزات
        
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str):
        """
        Save model to file
        حفظ النموذج إلى ملف
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_dict, filepath)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FirstReceiverPredictor':
        """
        Load model from file
        تحميل النموذج من ملف
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded FirstReceiverPredictor
        """
        save_dict = joblib.load(filepath)
        
        predictor = cls(model_type=save_dict['model_type'])
        predictor.model = save_dict['model']
        predictor.scaler = save_dict['scaler']
        predictor.label_encoder = save_dict['label_encoder']
        predictor.feature_columns = save_dict['feature_columns']
        predictor.is_fitted = save_dict['is_fitted']
        
        print(f"📂 Model loaded from {filepath}")
        return predictor


def train_receiver_predictor(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> Tuple[FirstReceiverPredictor, Dict]:
    """
    Convenience function to train a receiver predictor
    دالة ملائمة لتدريب متنبئ المستلم
    
    Args:
        df: Set piece DataFrame
        save_path: Optional path to save the model
    
    Returns:
        Tuple of (trained predictor, metrics)
    """
    # Initialize predictor
    predictor = FirstReceiverPredictor()
    
    # Prepare features
    X, feature_cols = predictor.prepare_features(df)
    y = df['first_receiver_idx'].fillna(0).astype(int)
    
    # Train
    metrics = predictor.train(X, y)
    
    # Save if path provided
    if save_path:
        predictor.save(save_path)
    
    return predictor, metrics


if __name__ == "__main__":
    # Test the receiver predictor
    import sys
    sys.path.append('..')
    from data.loaders import load_wyscout_data
    from data.extractors import extract_set_pieces
    from features.spatial import calculate_spatial_features
    from features.temporal import calculate_temporal_features
    from features.physical import calculate_physical_features
    
    print("=" * 60)
    print("Testing First Receiver Predictor")
    print("=" * 60)
    
    # Load and prepare data
    print("\n📥 Loading data...")
    df = load_wyscout_data()
    set_pieces = extract_set_pieces(df)
    
    # Add features
    print("\n🔧 Engineering features...")
    set_pieces = calculate_spatial_features(set_pieces)
    set_pieces = calculate_temporal_features(set_pieces)
    set_pieces = calculate_physical_features(set_pieces)
    
    # Train model
    print("\n🚀 Training model...")
    predictor, metrics = train_receiver_predictor(set_pieces)
    
    # Show feature importance
    print("\n📊 Top 10 Feature Importances:")
    importance_df = predictor.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))
