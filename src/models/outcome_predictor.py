"""
Elite Set-Piece Analytics - Outcome Predictor
متنبئ نتيجة الكرة الثابتة

This module predicts the outcome of a set piece:
- Goal
- Shot on target
- Possession retained
- Possession lost
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class OutcomePredictor:
    """
    Predict the outcome of a set piece
    التنبؤ بنتيجة الكرة الثابتة
    
    Outcomes:
    - goal: Set piece leads to a goal
    - shot: Set piece leads to a shot (no goal)
    - possession_retained: Team keeps the ball
    - possession_lost: Opponent gains possession
    - clearance: Ball is cleared by defense
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
        binary_mode: bool = False
    ):
        """
        Initialize the predictor
        تهيئة المتنبئ
        
        Args:
            model_type: 'xgboost', 'random_forest', or 'gradient_boosting'
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            binary_mode: If True, predict success (goal/shot) vs non-success
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.binary_mode = binary_mode
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_fitted = False
        
        # Class names
        if binary_mode:
            self.class_names = ['not_successful', 'successful']
        else:
            self.class_names = ['goal', 'shot', 'possession_retained', 'possession_lost', 'clearance']
        
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
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'  # Handle imbalanced outcomes
            )
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features from set piece data
        تحضير الميزات من بيانات الكرات الثابتة
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
            'height_advantage', 'delivery_speed', 'centroid_distance',
            'attacker_spread_x', 'attacker_spread_y'
        ]
        
        for col in optional_numerical:
            if col in df.columns:
                numerical_cols.append(col)
        
        # Start with numerical features
        features = df[numerical_cols].copy()
        
        # Add one-hot encoded categorical features
        categorical_cols = ['type', 'side', 'game_state']
        if 'delivery_type' in df.columns:
            categorical_cols.append('delivery_type')
        
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col].fillna('unknown'), prefix=col)
                features = pd.concat([features, dummies], axis=1)
        
        # Fill any missing values
        features = features.fillna(0)
        
        return features, features.columns.tolist()
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Prepare target variable
        تحضير المتغير الهدف
        """
        if self.binary_mode:
            # Binary: success (goal or shot) vs non-success
            return df['outcome'].isin(['goal', 'shot']).astype(int)
        else:
            return df['outcome']
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model
        تدريب النموذج
        """
        print("🚀 Training Outcome Predictor...")
        
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
        print(f"  Classes: {self.label_encoder.classes_}")
        
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
        
        # AUC for binary classification
        if self.binary_mode or len(self.label_encoder.classes_) == 2:
            y_val_proba = self.model.predict_proba(X_val)[:, 1]
            metrics['val_auc'] = roc_auc_score(y_val, y_val_proba)
        
        print(f"\n✅ Training complete!")
        print(f"   Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"   Val Accuracy: {metrics['val_accuracy']:.2%}")
        if 'val_auc' in metrics:
            print(f"   Val AUC: {metrics['val_auc']:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict outcome
        التنبؤ بالنتيجة
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        X_aligned = self._align_features(X)
        X_scaled = self.scaler.transform(X_aligned)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for each outcome
        التنبؤ باحتمالات كل نتيجة
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        X_aligned = self._align_features(X)
        X_scaled = self.scaler.transform(X_aligned)
        
        return self.model.predict_proba(X_scaled)
    
    def predict_success_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of successful outcome (goal or shot)
        التنبؤ باحتمال النتيجة الناجحة
        """
        probas = self.predict_proba(X)
        classes = self.label_encoder.classes_
        
        # Sum probabilities of goal and shot
        success_prob = np.zeros(len(X))
        for i, cls in enumerate(classes):
            if cls in ['goal', 'shot', 'True', '1']:
                success_prob += probas[:, i]
        
        return success_prob
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align input features with training features"""
        X_copy = X.copy()
        for col in self.feature_columns:
            if col not in X_copy.columns:
                X_copy[col] = 0
        return X_copy[self.feature_columns]
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test data
        تقييم النموذج على بيانات الاختبار
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
        
        print("📊 Evaluation Results:")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        print(f"   F1 Score: {metrics['f1_weighted']:.3f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_outcome_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed outcome probabilities
        الحصول على احتمالات النتائج التفصيلية
        """
        probas = self.predict_proba(X)
        classes = self.label_encoder.classes_
        
        return pd.DataFrame(probas, columns=classes)
    
    def save(self, filepath: str):
        """Save model to file"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'binary_mode': self.binary_mode,
            'is_fitted': self.is_fitted
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_dict, filepath)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'OutcomePredictor':
        """Load model from file"""
        save_dict = joblib.load(filepath)
        
        predictor = cls(
            model_type=save_dict['model_type'],
            binary_mode=save_dict.get('binary_mode', False)
        )
        predictor.model = save_dict['model']
        predictor.scaler = save_dict['scaler']
        predictor.label_encoder = save_dict['label_encoder']
        predictor.feature_columns = save_dict['feature_columns']
        predictor.is_fitted = save_dict['is_fitted']
        
        print(f"📂 Model loaded from {filepath}")
        return predictor


class GoalProbabilityModel:
    """
    Specialized model to predict goal probability
    نموذج متخصص للتنبؤ باحتمالية الهدف
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the goal probability model"""
        # Target: is it a goal?
        y_binary = (y == 'goal').astype(int)
        
        self.feature_columns = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_binary,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_binary
        )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'auc': roc_auc_score(y_val, y_val_proba),
            'accuracy': accuracy_score(y_val, (y_val_proba > 0.5).astype(int))
        }
        
        return metrics
    
    def predict_goal_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of scoring a goal"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_aligned = X.copy()
        for col in self.feature_columns:
            if col not in X_aligned.columns:
                X_aligned[col] = 0
        X_aligned = X_aligned[self.feature_columns]
        
        X_scaled = self.scaler.transform(X_aligned)
        return self.model.predict_proba(X_scaled)[:, 1]


def train_outcome_predictor(
    df: pd.DataFrame,
    binary_mode: bool = False,
    save_path: Optional[str] = None
) -> Tuple[OutcomePredictor, Dict]:
    """
    Convenience function to train an outcome predictor
    دالة ملائمة لتدريب متنبئ النتيجة
    """
    predictor = OutcomePredictor(binary_mode=binary_mode)
    
    X, feature_cols = predictor.prepare_features(df)
    y = predictor.prepare_target(df)
    
    metrics = predictor.train(X, y)
    
    if save_path:
        predictor.save(save_path)
    
    return predictor, metrics


if __name__ == "__main__":
    # Test the outcome predictor
    import sys
    sys.path.append('..')
    from data.loaders import load_wyscout_data
    from data.extractors import extract_set_pieces
    from features.spatial import calculate_spatial_features
    from features.temporal import calculate_temporal_features
    from features.physical import calculate_physical_features
    
    print("=" * 60)
    print("Testing Outcome Predictor")
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
    
    # Train multi-class model
    print("\n🚀 Training multi-class model...")
    predictor_multi, metrics_multi = train_outcome_predictor(set_pieces, binary_mode=False)
    
    # Train binary model
    print("\n🚀 Training binary model...")
    predictor_binary, metrics_binary = train_outcome_predictor(set_pieces, binary_mode=True)
    
    # Show feature importance
    print("\n📊 Top 10 Feature Importances (Binary Model):")
    importance_df = predictor_binary.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))
