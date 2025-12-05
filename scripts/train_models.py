#!/usr/bin/env python
"""
Elite Set-Piece Analytics - Model Training Script
سكريبت تدريب النماذج

This script trains the first receiver prediction model and other
analytical models for the set-piece analytics platform.

Usage:
    python scripts/train_models.py --config config.yaml
    python scripts/train_models.py --model receiver
    python scripts/train_models.py --model all
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.config import Config
from models.receiver_predictor import ReceiverPredictor
from models.outcome_predictor import OutcomePredictor
from models.pattern_analyzer import PatternAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> tuple:
    """
    Load and prepare training data.
    
    Args:
        data_path: Path to processed data
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Loading training data from {data_path}")
    
    # Check if data exists
    if not data_path.exists():
        logger.warning("Data directory does not exist. Using synthetic data for demo.")
        return create_synthetic_data()
    
    # Load features and labels
    features_file = data_path / "features.csv"
    labels_file = data_path / "labels.csv"
    
    if features_file.exists() and labels_file.exists():
        X = pd.read_csv(features_file)
        y = pd.read_csv(labels_file)['receiver_id']
    else:
        logger.warning("Data files not found. Using synthetic data for demo.")
        return create_synthetic_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED,
        stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def create_synthetic_data() -> tuple:
    """
    Create synthetic data for demonstration purposes.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Creating synthetic training data...")
    
    np.random.seed(Config.RANDOM_SEED)
    n_samples = 1000
    n_features = 15
    n_classes = 10  # 10 possible receivers
    
    # Generate random features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some structure to features
    X['distance_to_goal'] = np.random.uniform(5, 40, n_samples)
    X['angle_to_goal'] = np.random.uniform(0, 90, n_samples)
    X['player_density'] = np.random.uniform(0, 10, n_samples)
    
    # Generate labels with some correlation to features
    probs = np.abs(X['distance_to_goal'] - 20) / 20  # Closer to 20m = higher prob
    probs = probs / probs.sum()  # Normalize
    y = pd.Series(
        np.random.choice(range(n_classes), size=n_samples),
        name='receiver_id'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED
    )
    
    return X_train, X_test, y_train, y_test


def train_receiver_predictor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_path: Path
) -> ReceiverPredictor:
    """
    Train the first receiver prediction model.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        save_path: Path to save the model
        
    Returns:
        Trained model
    """
    logger.info("Training ReceiverPredictor...")
    
    # Initialize model
    model = ReceiverPredictor()
    
    # Train
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    logger.info("=" * 50)
    logger.info("Model Evaluation Results:")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    if 'top_3_accuracy' in metrics:
        logger.info(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    
    # Feature importance
    importance = model.get_feature_importance()
    logger.info("\nTop 10 Features:")
    for idx, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")
    
    return model


def train_outcome_predictor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_path: Path
) -> OutcomePredictor:
    """
    Train the outcome prediction model.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels (0/1 for goal)
        y_test: Test labels
        save_path: Path to save the model
        
    Returns:
        Trained model
    """
    logger.info("Training OutcomePredictor...")
    
    # Create binary labels (for demo, random)
    y_train_binary = (np.random.rand(len(y_train)) > 0.9).astype(int)
    y_test_binary = (np.random.rand(len(y_test)) > 0.9).astype(int)
    
    # Initialize model
    model = OutcomePredictor(outcome_type='goal')
    
    # Train
    model.fit(X_train, y_train_binary, eval_set=(X_test, y_test_binary), verbose=True)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test_binary)
    
    logger.info("Outcome Predictor Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
    
    # Save model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")
    
    return model


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train set-piece analytics models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["receiver", "outcome", "pattern", "all"],
        default="all",
        help="Which model to train"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to save trained models"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir) if args.data_dir else Config.PROCESSED_DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else Config.MODELS_DIR
    
    logger.info("=" * 60)
    logger.info("Elite Set-Piece Analytics - Model Training")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models to train: {args.model}")
    logger.info("")
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data(data_dir)
    
    # Train models
    if args.model in ["receiver", "all"]:
        train_receiver_predictor(
            X_train, X_test, y_train, y_test,
            output_dir / "receiver_predictor.joblib"
        )
    
    if args.model in ["outcome", "all"]:
        train_outcome_predictor(
            X_train, X_test, y_train, y_test,
            output_dir / "outcome_predictor.joblib"
        )
    
    if args.model in ["pattern", "all"]:
        logger.info("Training PatternAnalyzer...")
        analyzer = PatternAnalyzer(n_clusters=10)
        analyzer.fit(X_train)
        logger.info(f"PatternAnalyzer fitted with {analyzer.n_clusters} clusters")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
