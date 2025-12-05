"""
Tests for data loading and processing modules.
اختبارات تحميل ومعالجة البيانات

This module contains unit tests for:
- Data loaders (Wyscout, StatsBomb, Metrica)
- Set-piece extractors
- Data preprocessors
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for proper imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Set up package for relative imports
os.chdir(str(Path(__file__).parent.parent))


class TestWyscoutLoader:
    """Tests for WyscoutLoader class."""
    
    def test_loader_initialization(self):
        """Test that WyscoutLoader initializes correctly."""
        from src.data.loaders import WyscoutLoader
        
        loader = WyscoutLoader()
        assert loader is not None
        assert hasattr(loader, 'wyscout_dir')
    
    def test_event_types_mapping(self):
        """Test that event type mappings are correct."""
        from src.data.loaders import WyscoutLoader
        
        loader = WyscoutLoader()
        assert 8 in loader.EVENT_TYPES
        assert loader.EVENT_TYPES[8] == "Pass"
        assert loader.EVENT_TYPES[9] == "Shot"
    
    def test_load_events_returns_dataframe(self):
        """Test that load_events returns a DataFrame (empty if no data)."""
        from src.data.loaders import WyscoutLoader
        
        loader = WyscoutLoader()
        events = loader.load_events(match_id=99999)  # Non-existent match
        
        assert isinstance(events, pd.DataFrame)


class TestStatsBombLoader:
    """Tests for StatsBombLoader class."""
    
    def test_loader_initialization(self):
        """Test that StatsBombLoader initializes correctly."""
        from src.data.loaders import StatsBombLoader
        
        loader = StatsBombLoader(use_api=False)
        assert loader is not None
        assert hasattr(loader, 'use_api')
        assert loader.use_api is False
    
    def test_load_events_returns_dataframe(self):
        """Test that load_events returns a DataFrame."""
        from src.data.loaders import StatsBombLoader
        
        loader = StatsBombLoader(use_api=False)
        events = loader.load_events(match_id=99999)  # Non-existent match
        
        assert isinstance(events, pd.DataFrame)


class TestMetricaLoader:
    """Tests for MetricaLoader class."""
    
    def test_loader_initialization(self):
        """Test that MetricaLoader initializes correctly."""
        from src.data.loaders import MetricaLoader
        
        loader = MetricaLoader()
        assert loader is not None
        assert hasattr(loader, 'SAMPLE_MATCHES')
    
    def test_sample_matches_exist(self):
        """Test that sample match definitions exist."""
        from src.data.loaders import MetricaLoader
        
        loader = MetricaLoader()
        assert 1 in loader.SAMPLE_MATCHES
        assert loader.SAMPLE_MATCHES[1] == "Sample_Game_1"


class TestUnifiedDataLoader:
    """Tests for UnifiedDataLoader class."""
    
    def test_loader_initialization(self):
        """Test that UnifiedDataLoader initializes correctly."""
        from src.data.loaders import UnifiedDataLoader
        
        loader = UnifiedDataLoader()
        assert loader is not None
        assert 'wyscout' in loader.loaders
        assert 'statsbomb' in loader.loaders
        assert 'metrica' in loader.loaders
    
    def test_invalid_source_raises_error(self):
        """Test that invalid source raises ValueError."""
        from src.data.loaders import UnifiedDataLoader
        
        loader = UnifiedDataLoader()
        with pytest.raises(ValueError):
            loader.load_events(source="invalid_source", match_id=123)
    
    def test_get_set_pieces_filters_correctly(self):
        """Test that get_set_pieces filters set-piece events."""
        from src.data.loaders import UnifiedDataLoader
        
        loader = UnifiedDataLoader()
        
        # Create sample events
        events = pd.DataFrame({
            'event_type': ['Pass', 'Corner', 'Shot', 'Free Kick', 'Dribble'],
            'x': [50, 0, 90, 30, 40],
            'y': [34, 68, 34, 34, 34]
        })
        
        set_pieces = loader.get_set_pieces(events)
        
        assert len(set_pieces) == 2  # Corner and Free Kick
        assert 'Corner' in set_pieces['event_type'].values or 'corner' in set_pieces['event_type'].str.lower().values


class TestSetPieceExtractor:
    """Tests for SetPieceExtractor class."""
    
    def test_extractor_initialization(self):
        """Test that SetPieceExtractor initializes correctly."""
        from src.data.extractors import SetPieceExtractor
        
        extractor = SetPieceExtractor()
        assert extractor is not None
        assert extractor.pitch_length == 105.0
        assert extractor.pitch_width == 68.0
    
    def test_extract_corners(self):
        """Test corner kick extraction."""
        from src.data.extractors import SetPieceExtractor
        
        extractor = SetPieceExtractor()
        
        events = pd.DataFrame({
            'event_type': ['Corner', 'Pass', 'corner kick', 'Shot'],
            'x': [0, 50, 0, 90],
            'y': [0, 34, 68, 34]
        })
        
        corners = extractor.extract_corners(events)
        
        assert isinstance(corners, pd.DataFrame)
        assert len(corners) == 2
        assert all(corners['set_piece_type'] == 'corner')
    
    def test_extract_free_kicks(self):
        """Test free kick extraction."""
        from src.data.extractors import SetPieceExtractor
        
        extractor = SetPieceExtractor()
        
        events = pd.DataFrame({
            'event_type': ['Free Kick', 'Pass', 'freekick', 'Shot'],
            'x': [25, 50, 35, 90],
            'y': [30, 34, 40, 34]
        })
        
        free_kicks = extractor.extract_free_kicks(events)
        
        assert isinstance(free_kicks, pd.DataFrame)
        assert len(free_kicks) == 2
        assert all(free_kicks['set_piece_type'] == 'free_kick')
    
    def test_extract_all_combines_set_pieces(self):
        """Test that extract_all combines all set-piece types."""
        from src.data.extractors import SetPieceExtractor
        
        extractor = SetPieceExtractor()
        
        events = pd.DataFrame({
            'event_type': ['Corner', 'Free Kick', 'Throw-in', 'Penalty', 'Pass'],
            'x': [0, 25, 50, 11, 50],
            'y': [0, 30, 0, 34, 34],
            'timestamp': [1, 2, 3, 4, 5]
        })
        
        all_set_pieces = extractor.extract_all(events)
        
        assert isinstance(all_set_pieces, pd.DataFrame)
        assert len(all_set_pieces) >= 4  # At least 4 set-pieces


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_preprocessor_initialization(self):
        """Test that DataPreprocessor initializes correctly."""
        from src.data.extractors import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
    
    def test_clean_data_removes_duplicates(self):
        """Test that clean_data removes duplicates."""
        from src.data.extractors import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'event_type': ['Pass', 'Pass', 'Shot'],
            'x': [50, 50, 90],
            'y': [34, 34, 34],
            'timestamp': [1, 1, 2]
        })
        
        cleaned = preprocessor.clean_data(df)
        
        assert len(cleaned) == 2  # Duplicate removed
    
    def test_validate_coordinates_clips_values(self):
        """Test that validate_coordinates clips out-of-bounds values."""
        from src.data.extractors import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'x': [-10, 50, 120],
            'y': [-5, 34, 80]
        })
        
        validated = preprocessor.validate_coordinates(df)
        
        assert validated['x'].min() >= 0
        assert validated['x'].max() <= 105
        assert validated['y'].min() >= 0
        assert validated['y'].max() <= 68


class TestConfig:
    """Tests for Config class."""
    
    def test_config_pitch_dimensions(self):
        """Test that pitch dimensions are correct."""
        from src.utils.config import Config
        
        assert Config.PITCH_LENGTH == 105.0
        assert Config.PITCH_WIDTH == 68.0
        assert Config.GOAL_WIDTH == 7.32
    
    def test_config_get_pitch_dimensions(self):
        """Test get_pitch_dimensions method."""
        from src.utils.config import Config
        
        dimensions = Config.get_pitch_dimensions()
        
        assert isinstance(dimensions, dict)
        assert 'length' in dimensions
        assert 'width' in dimensions
        assert dimensions['length'] == 105.0
    
    def test_config_get_all_features(self):
        """Test get_all_features method."""
        from src.utils.config import Config
        
        features = Config.get_all_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert 'distance_to_goal' in features


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
