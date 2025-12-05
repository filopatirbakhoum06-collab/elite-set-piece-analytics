"""
Tests for machine learning models.
اختبارات نماذج التعلم الآلي
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for proper imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Set up package for relative imports
os.chdir(str(Path(__file__).parent.parent))


class TestReceiverPredictor:
    """Tests for ReceiverPredictor class."""
    
    def test_initialization(self):
        """Test that ReceiverPredictor initializes correctly."""
        from src.models.receiver_predictor import ReceiverPredictor
        
        predictor = ReceiverPredictor()
        assert predictor is not None
        assert predictor.is_fitted is False
    
    def test_fit_and_predict(self):
        """Test basic fit and predict functionality."""
        from src.models.receiver_predictor import ReceiverPredictor
        
        # Create synthetic data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.choice(['player_1', 'player_2', 'player_3'], 100))
        
        predictor = ReceiverPredictor()
        predictor.fit(X, y, verbose=False)
        
        assert predictor.is_fitted is True
        
        predictions = predictor.predict(X[:10])
        assert len(predictions) == 10
    
    def test_predict_proba(self):
        """Test probability prediction."""
        from src.models.receiver_predictor import ReceiverPredictor
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.choice(['player_1', 'player_2', 'player_3'], 100))
        
        predictor = ReceiverPredictor()
        predictor.fit(X, y, verbose=False)
        
        probabilities = predictor.predict_proba(X[:10])
        
        assert probabilities.shape == (10, 3)  # 10 samples, 3 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_get_top_k_receivers(self):
        """Test top-k receiver prediction."""
        from src.models.receiver_predictor import ReceiverPredictor
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.choice(['player_1', 'player_2', 'player_3'], 100))
        
        predictor = ReceiverPredictor()
        predictor.fit(X, y, verbose=False)
        
        top_k = predictor.get_top_k_receivers(X[:5], k=2)
        
        assert len(top_k) == 5
        assert all(len(receivers) == 2 for receivers in top_k)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        from src.models.receiver_predictor import ReceiverPredictor
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10), 
                        columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(np.random.choice(['player_1', 'player_2', 'player_3'], 100))
        
        predictor = ReceiverPredictor()
        predictor.fit(X, y, verbose=False)
        
        importance = predictor.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == 10
    
    def test_not_fitted_raises_error(self):
        """Test that predict raises error if not fitted."""
        from src.models.receiver_predictor import ReceiverPredictor
        
        predictor = ReceiverPredictor()
        X = pd.DataFrame(np.random.randn(10, 5))
        
        with pytest.raises(RuntimeError):
            predictor.predict(X)


class TestOutcomePredictor:
    """Tests for OutcomePredictor class."""
    
    def test_initialization(self):
        """Test that OutcomePredictor initializes correctly."""
        from src.models.outcome_predictor import OutcomePredictor
        
        predictor = OutcomePredictor(outcome_type='goal')
        assert predictor is not None
        assert predictor.outcome_type == 'goal'
        assert predictor.is_fitted is False
    
    def test_invalid_outcome_type(self):
        """Test that invalid outcome type raises error."""
        from src.models.outcome_predictor import OutcomePredictor
        
        with pytest.raises(ValueError):
            OutcomePredictor(outcome_type='invalid')
    
    def test_fit_and_predict_proba(self):
        """Test fit and predict_proba for goal prediction."""
        from src.models.outcome_predictor import OutcomePredictor
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        y = np.random.randint(0, 2, 100)
        
        predictor = OutcomePredictor(outcome_type='goal')
        predictor.fit(X, y, verbose=False)
        
        proba = predictor.predict_proba(X[:10])
        
        assert len(proba) == 10
        assert all(0 <= p <= 1 for p in proba)


class TestPatternAnalyzer:
    """Tests for PatternAnalyzer class."""
    
    def test_initialization(self):
        """Test that PatternAnalyzer initializes correctly."""
        from src.models.pattern_analyzer import PatternAnalyzer
        
        analyzer = PatternAnalyzer(n_clusters=5)
        assert analyzer is not None
        assert analyzer.n_clusters == 5
        assert analyzer.is_fitted is False
    
    def test_fit_and_cluster(self):
        """Test fit and clustering."""
        from src.models.pattern_analyzer import PatternAnalyzer
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        
        analyzer = PatternAnalyzer(n_clusters=5)
        analyzer.fit(X)
        
        assert analyzer.is_fitted is True
        
        clusters = analyzer.cluster_set_pieces(X)
        
        assert len(clusters) == 100
        assert all(0 <= c < 5 for c in clusters)
    
    def test_find_similar(self):
        """Test finding similar set-pieces."""
        from src.models.pattern_analyzer import PatternAnalyzer
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        
        analyzer = PatternAnalyzer(n_clusters=5)
        analyzer.fit(X)
        
        query = X.iloc[[0]]
        similar = analyzer.find_similar(query, X, n=5)
        
        assert len(similar) == 5
        assert similar[0][0] == 0  # First result should be the query itself


class TestMetrics:
    """Tests for Metrics class."""
    
    def test_calculate_accuracy(self):
        """Test accuracy calculation."""
        from src.utils.metrics import Metrics
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        accuracy = Metrics.calculate_accuracy(y_true, y_pred)
        
        assert accuracy == pytest.approx(0.8)
    
    def test_top_k_accuracy(self):
        """Test top-k accuracy calculation."""
        from src.utils.metrics import Metrics
        
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [0.7, 0.2, 0.1],  # Correct (class 0)
            [0.3, 0.4, 0.3],  # Correct (class 1)
            [0.4, 0.3, 0.3],  # Wrong (class 0, not 2)
        ])
        
        top_1 = Metrics.top_k_accuracy(y_true, y_proba, k=1)
        top_2 = Metrics.top_k_accuracy(y_true, y_proba, k=2)
        
        assert top_1 == pytest.approx(2/3, rel=0.1)
        assert top_2 >= top_1
    
    def test_calculate_all(self):
        """Test all metrics calculation."""
        from src.utils.metrics import Metrics
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = Metrics.calculate_all(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
