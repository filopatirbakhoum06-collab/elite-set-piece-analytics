"""
Elite Set-Piece Analytics - Tests
اختبارات تحليلات الكرات الثابتة

Run with: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_wyscout_data, _generate_sample_wyscout_data
from src.data.extractors import (
    extract_set_pieces, extract_corners, extract_free_kicks,
    calculate_success_metrics, get_first_receiver_data
)
from src.data.preprocessors import (
    clean_data, normalize_coordinates, handle_outliers, split_data
)


class TestDataLoaders:
    """Test data loading functionality"""
    
    def test_data_loading_returns_dataframe(self):
        """Test that data loads and returns a DataFrame"""
        data = load_wyscout_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
    
    def test_data_has_required_columns(self):
        """Test that loaded data has required columns"""
        data = load_wyscout_data()
        required_columns = ['event_id', 'match_id', 'team', 'type', 'x', 'y', 'outcome']
        for col in required_columns:
            assert col in data.columns, f"Missing column: {col}"
    
    def test_sample_data_generation(self):
        """Test sample data generator"""
        data = _generate_sample_wyscout_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 100  # Should generate substantial data
    
    def test_data_types_are_correct(self):
        """Test that data types are as expected"""
        data = load_wyscout_data()
        assert data['x'].dtype in [np.float64, np.int64, float, int]
        assert data['y'].dtype in [np.float64, np.int64, float, int]
        assert data['minute'].dtype in [np.int64, np.int32, int]


class TestDataExtractors:
    """Test data extraction functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Load sample data for testing"""
        return load_wyscout_data()
    
    def test_set_piece_extraction(self, sample_data):
        """Test set piece extraction"""
        set_pieces = extract_set_pieces(sample_data)
        assert isinstance(set_pieces, pd.DataFrame)
        assert len(set_pieces) > 0
        assert 'type' in set_pieces.columns
    
    def test_set_piece_types_are_valid(self, sample_data):
        """Test that only valid set piece types are extracted"""
        set_pieces = extract_set_pieces(sample_data)
        valid_types = ['corner', 'free_kick', 'throw_in', 'penalty']
        for sp_type in set_pieces['type'].unique():
            assert sp_type in valid_types, f"Invalid type: {sp_type}"
    
    def test_corner_extraction(self, sample_data):
        """Test corner kick extraction"""
        set_pieces = extract_set_pieces(sample_data)
        corners = extract_corners(set_pieces)
        
        assert all(corners['type'] == 'corner')
        assert 'is_near_post' in corners.columns
    
    def test_free_kick_extraction(self, sample_data):
        """Test free kick extraction"""
        set_pieces = extract_set_pieces(sample_data)
        free_kicks = extract_free_kicks(set_pieces)
        
        assert all(free_kicks['type'] == 'free_kick')
        assert 'is_direct_shot_range' in free_kicks.columns
    
    def test_success_metrics_calculation(self, sample_data):
        """Test success metrics calculation"""
        set_pieces = extract_set_pieces(sample_data)
        metrics = calculate_success_metrics(set_pieces)
        
        assert 'success_rate' in metrics
        assert 'goal_rate' in metrics
        assert 0 <= metrics['success_rate'] <= 100
        assert 0 <= metrics['goal_rate'] <= 100
    
    def test_first_receiver_data_preparation(self, sample_data):
        """Test first receiver data preparation"""
        set_pieces = extract_set_pieces(sample_data)
        X, y = get_first_receiver_data(set_pieces)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)


class TestPreprocessors:
    """Test data preprocessing functionality"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'x': [10, 20, 30, np.nan, 50],
            'y': [5, 15, 25, 35, 45],
            'category': ['A', 'B', None, 'A', 'B'],
            'value': [1, 2, 3, 4, 1000]  # 1000 is an outlier
        })
    
    def test_clean_data_handles_missing(self, sample_df):
        """Test that clean_data handles missing values"""
        cleaned = clean_data(sample_df)
        assert cleaned['x'].isnull().sum() == 0
        assert cleaned['category'].isnull().sum() == 0
    
    def test_normalize_coordinates(self, sample_df):
        """Test coordinate normalization"""
        normalized = normalize_coordinates(sample_df, 'x', 'y')
        
        assert 'x_meters' in normalized.columns
        assert 'y_meters' in normalized.columns
    
    def test_handle_outliers_iqr(self, sample_df):
        """Test outlier handling with IQR method"""
        cleaned = handle_outliers(sample_df, columns=['value'], method='iqr')
        
        # The outlier (1000) should be capped
        assert cleaned['value'].max() < 1000
    
    def test_split_data_proportions(self):
        """Test that data split maintains proportions"""
        X = pd.DataFrame({'a': range(100), 'b': range(100)})
        y = pd.Series([0] * 50 + [1] * 50)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.2, val_size=0.1
        )
        
        # Check sizes
        assert len(X_train) == 70
        assert len(X_val) == 10
        assert len(X_test) == 20


class TestFeatures:
    """Test feature engineering"""
    
    @pytest.fixture
    def sample_set_pieces(self):
        """Load and extract sample set pieces"""
        from src.features.spatial import calculate_spatial_features
        from src.features.temporal import calculate_temporal_features
        from src.features.physical import calculate_physical_features
        
        data = load_wyscout_data()
        set_pieces = extract_set_pieces(data)
        return set_pieces
    
    def test_spatial_features(self, sample_set_pieces):
        """Test spatial feature calculation"""
        from src.features.spatial import calculate_spatial_features
        
        with_features = calculate_spatial_features(sample_set_pieces)
        
        assert 'goal_distance' in with_features.columns
        assert 'goal_angle' in with_features.columns
        assert 'danger_index' in with_features.columns
    
    def test_temporal_features(self, sample_set_pieces):
        """Test temporal feature calculation"""
        from src.features.temporal import calculate_temporal_features
        
        with_features = calculate_temporal_features(sample_set_pieces)
        
        assert 'time_since_last_sp' in with_features.columns
        assert 'fatigue_factor' in with_features.columns
        assert 'score_pressure' in with_features.columns
    
    def test_physical_features(self, sample_set_pieces):
        """Test physical feature calculation"""
        from src.features.physical import calculate_physical_features
        
        with_features = calculate_physical_features(sample_set_pieces)
        
        assert 'header_probability' in with_features.columns
        assert 'height_advantage' in with_features.columns


class TestModels:
    """Test model functionality"""
    
    @pytest.fixture
    def training_data(self):
        """Prepare training data"""
        from src.features.spatial import calculate_spatial_features
        
        data = load_wyscout_data()
        set_pieces = extract_set_pieces(data)
        set_pieces = calculate_spatial_features(set_pieces)
        return set_pieces
    
    def test_receiver_predictor_training(self, training_data):
        """Test receiver predictor can be trained"""
        from src.models.receiver_predictor import FirstReceiverPredictor
        
        predictor = FirstReceiverPredictor(n_estimators=10, max_depth=3)
        X, _ = predictor.prepare_features(training_data)
        y = training_data['first_receiver_idx'].fillna(0).astype(int)
        
        metrics = predictor.train(X, y)
        
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
        assert predictor.is_fitted
    
    def test_outcome_predictor_training(self, training_data):
        """Test outcome predictor can be trained"""
        from src.models.outcome_predictor import OutcomePredictor
        
        predictor = OutcomePredictor(n_estimators=10, max_depth=3, binary_mode=True)
        X, _ = predictor.prepare_features(training_data)
        y = predictor.prepare_target(training_data)
        
        metrics = predictor.train(X, y)
        
        assert 'train_accuracy' in metrics
        assert predictor.is_fitted
    
    def test_model_prediction(self, training_data):
        """Test that trained model can make predictions"""
        from src.models.receiver_predictor import FirstReceiverPredictor
        
        predictor = FirstReceiverPredictor(n_estimators=10, max_depth=3)
        X, _ = predictor.prepare_features(training_data)
        y = training_data['first_receiver_idx'].fillna(0).astype(int)
        
        predictor.train(X, y)
        
        # Make predictions
        predictions = predictor.predict(X.head(10))
        
        assert len(predictions) == 10


class TestVisualization:
    """Test visualization functionality"""
    
    def test_draw_pitch(self):
        """Test pitch drawing"""
        from src.visualization.pitch import draw_pitch
        import matplotlib.pyplot as plt
        
        fig, ax = draw_pitch()
        
        assert fig is not None
        assert ax is not None
        
        plt.close()
    
    def test_plot_heatmap(self):
        """Test heatmap plotting"""
        from src.visualization.pitch import plot_heatmap
        import matplotlib.pyplot as plt
        
        data = load_wyscout_data()
        set_pieces = extract_set_pieces(data)
        
        fig, ax = plot_heatmap(set_pieces)
        
        assert fig is not None
        
        plt.close()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
