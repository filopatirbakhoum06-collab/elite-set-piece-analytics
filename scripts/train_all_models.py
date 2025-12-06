#!/usr/bin/env python
"""
Elite Set-Piece Analytics - Train All Models
تدريب جميع النماذج

Usage:
    python scripts/train_all_models.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_wyscout_data
from src.data.extractors import extract_set_pieces, calculate_success_metrics
from src.features.spatial import calculate_spatial_features
from src.features.temporal import calculate_temporal_features
from src.features.physical import calculate_physical_features
from src.models.receiver_predictor import FirstReceiverPredictor, train_receiver_predictor
from src.models.outcome_predictor import OutcomePredictor, train_outcome_predictor


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("⚽ Elite Set-Piece Analytics - Model Training Pipeline")
    print("=" * 60)
    
    # 1. Load data
    print("\n📥 Step 1: Loading data...")
    data = load_wyscout_data()
    print(f"   Loaded {len(data)} events")
    
    # 2. Extract set pieces
    print("\n⚽ Step 2: Extracting set pieces...")
    set_pieces = extract_set_pieces(data)
    print(f"   Found {len(set_pieces)} set pieces")
    
    # Show initial metrics
    metrics = calculate_success_metrics(set_pieces)
    print(f"\n   📊 Initial Metrics:")
    print(f"      Goal Rate: {metrics['goal_rate']:.2f}%")
    print(f"      Success Rate: {metrics['success_rate']:.2f}%")
    
    # 3. Feature engineering
    print("\n🔧 Step 3: Engineering features...")
    
    print("   3.1 Spatial features...")
    set_pieces = calculate_spatial_features(set_pieces)
    
    print("   3.2 Temporal features...")
    set_pieces = calculate_temporal_features(set_pieces)
    
    print("   3.3 Physical features...")
    set_pieces = calculate_physical_features(set_pieces)
    
    print(f"   ✅ Total features: {len(set_pieces.columns)} columns")
    
    # 4. Train First Receiver Predictor
    print("\n🤖 Step 4: Training First Receiver Predictor...")
    
    receiver_model = FirstReceiverPredictor(model_type='xgboost')
    X_receiver, _ = receiver_model.prepare_features(set_pieces)
    y_receiver = set_pieces['first_receiver_idx'].fillna(0).astype(int)
    
    receiver_metrics = receiver_model.train(X_receiver, y_receiver)
    
    # Save model
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    receiver_model.save(str(model_dir / "receiver_predictor_v1.pkl"))
    
    # Show feature importance
    print("\n   📊 Top 10 Important Features:")
    importance = receiver_model.get_feature_importance()
    for i, row in importance.head(10).iterrows():
        print(f"      {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # 5. Train Outcome Predictor
    print("\n🤖 Step 5: Training Outcome Predictor...")
    
    # Binary model
    print("   5.1 Binary model (success vs non-success)...")
    outcome_model_binary = OutcomePredictor(model_type='random_forest', binary_mode=True)
    X_outcome, _ = outcome_model_binary.prepare_features(set_pieces)
    y_outcome_binary = outcome_model_binary.prepare_target(set_pieces)
    
    binary_metrics = outcome_model_binary.train(X_outcome, y_outcome_binary)
    outcome_model_binary.save(str(model_dir / "outcome_predictor_binary_v1.pkl"))
    
    # Multi-class model
    print("\n   5.2 Multi-class model...")
    outcome_model_multi = OutcomePredictor(model_type='random_forest', binary_mode=False)
    X_outcome_multi, _ = outcome_model_multi.prepare_features(set_pieces)
    y_outcome_multi = outcome_model_multi.prepare_target(set_pieces)
    
    multi_metrics = outcome_model_multi.train(X_outcome_multi, y_outcome_multi)
    outcome_model_multi.save(str(model_dir / "outcome_predictor_multi_v1.pkl"))
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    print("\n📊 Model Performance Summary:")
    print("-" * 40)
    
    print("\n1. First Receiver Predictor:")
    print(f"   - Train Accuracy: {receiver_metrics['train_accuracy']:.2%}")
    print(f"   - Val Accuracy: {receiver_metrics['val_accuracy']:.2%}")
    if 'val_top3_accuracy' in receiver_metrics:
        print(f"   - Val Top-3 Accuracy: {receiver_metrics['val_top3_accuracy']:.2%}")
    
    print("\n2. Outcome Predictor (Binary):")
    print(f"   - Train Accuracy: {binary_metrics['train_accuracy']:.2%}")
    print(f"   - Val Accuracy: {binary_metrics['val_accuracy']:.2%}")
    if 'val_auc' in binary_metrics:
        print(f"   - Val AUC: {binary_metrics['val_auc']:.3f}")
    
    print("\n3. Outcome Predictor (Multi-class):")
    print(f"   - Train Accuracy: {multi_metrics['train_accuracy']:.2%}")
    print(f"   - Val Accuracy: {multi_metrics['val_accuracy']:.2%}")
    
    print("\n💾 Models saved to:")
    print(f"   - {model_dir / 'receiver_predictor_v1.pkl'}")
    print(f"   - {model_dir / 'outcome_predictor_binary_v1.pkl'}")
    print(f"   - {model_dir / 'outcome_predictor_multi_v1.pkl'}")
    
    print("\n🎉 Ready for predictions!")
    
    return receiver_model, outcome_model_binary, outcome_model_multi


if __name__ == "__main__":
    main()
