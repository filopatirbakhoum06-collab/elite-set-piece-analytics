"""
Tests for feature engineering modules.
اختبارات هندسة الميزات
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for proper imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Set up package for relative imports
os.chdir(str(Path(__file__).parent.parent))


class TestSpatialFeatures:
    """Tests for SpatialFeatures class."""
    
    def test_initialization(self):
        """Test that SpatialFeatures initializes correctly."""
        from src.features.spatial import SpatialFeatures
        
        spatial = SpatialFeatures()
        assert spatial is not None
        assert spatial.pitch_length == 105.0
        assert spatial.pitch_width == 68.0
    
    def test_distance_to_goal(self):
        """Test distance to goal calculation."""
        from src.features.spatial import SpatialFeatures
        
        spatial = SpatialFeatures()
        
        # From center of pitch
        distance = spatial.distance_to_goal(52.5, 34, attacking_right=True)
        
        assert distance > 0
        assert distance == pytest.approx(52.5, rel=0.1)
    
    def test_angle_to_goal(self):
        """Test angle to goal calculation."""
        from src.features.spatial import SpatialFeatures
        
        spatial = SpatialFeatures()
        
        # Directly in front of goal
        angle = spatial.angle_to_goal(90, 34, attacking_right=True)
        
        assert 0 <= angle <= 90
    
    def test_classify_zone(self):
        """Test zone classification."""
        from src.features.spatial import SpatialFeatures
        
        spatial = SpatialFeatures()
        
        zone = spatial.classify_zone(10, 34)
        assert 'defensive' in zone
        
        zone = spatial.classify_zone(90, 34)
        assert 'attacking' in zone
    
    def test_is_in_penalty_area(self):
        """Test penalty area detection."""
        from src.features.spatial import SpatialFeatures
        
        spatial = SpatialFeatures()
        
        # Inside penalty area
        assert spatial.is_in_penalty_area(100, 34, attacking_right=True)
        
        # Outside penalty area
        assert not spatial.is_in_penalty_area(50, 34, attacking_right=True)
    
    def test_player_density(self):
        """Test player density calculation."""
        from src.features.spatial import SpatialFeatures
        
        spatial = SpatialFeatures()
        
        positions = pd.DataFrame({
            'x': [50, 52, 55, 60, 70],
            'y': [34, 35, 36, 34, 34]
        })
        
        density = spatial.player_density(50, 34, positions, radius=5)
        
        assert density >= 2  # At least 2 players within 5m


class TestTemporalFeatures:
    """Tests for TemporalFeatures class."""
    
    def test_initialization(self):
        """Test that TemporalFeatures initializes correctly."""
        from src.features.temporal import TemporalFeatures
        
        temporal = TemporalFeatures()
        assert temporal is not None
        assert temporal.MATCH_DURATION == 90 * 60
    
    def test_time_until_end(self):
        """Test time until end calculation."""
        from src.features.temporal import TemporalFeatures
        
        temporal = TemporalFeatures()
        
        # First half, 30 minutes in
        time_remaining = temporal.time_until_end(30 * 60, period=1)
        
        assert time_remaining == pytest.approx(15 * 60)  # 15 minutes left
    
    def test_time_pressure(self):
        """Test time pressure calculation."""
        from src.features.temporal import TemporalFeatures
        
        temporal = TemporalFeatures()
        
        # Early in game, level score
        pressure_early = temporal.time_pressure(10 * 60, score_diff=0)
        
        # Late in game, behind
        pressure_late = temporal.time_pressure(85 * 60, score_diff=-1)
        
        assert pressure_late > pressure_early


class TestPhysicalFeatures:
    """Tests for PhysicalFeatures class."""
    
    def test_initialization(self):
        """Test that PhysicalFeatures initializes correctly."""
        from src.features.physical import PhysicalFeatures
        
        physical = PhysicalFeatures(frame_rate=25.0)
        assert physical is not None
        assert physical.frame_rate == 25.0
    
    def test_calculate_speed(self):
        """Test speed calculation."""
        from src.features.physical import PhysicalFeatures
        
        physical = PhysicalFeatures(frame_rate=25.0)
        
        # Moving 1 meter per frame
        x = pd.Series([0, 1, 2, 3, 4])
        y = pd.Series([0, 0, 0, 0, 0])
        
        speed = physical.calculate_speed(x, y, smooth=False)
        
        # Speed should be 25 m/s (1m per frame * 25 fps)
        assert speed.iloc[1:].mean() == pytest.approx(25.0, rel=0.1)
    
    def test_classify_movement(self):
        """Test movement classification."""
        from src.features.physical import PhysicalFeatures
        
        physical = PhysicalFeatures()
        
        assert physical.classify_movement(1.0) == "standing"
        assert physical.classify_movement(3.0) == "walking"
        assert physical.classify_movement(5.0) == "jogging"
        assert physical.classify_movement(6.5) == "running"
        assert physical.classify_movement(8.0) == "sprinting"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
