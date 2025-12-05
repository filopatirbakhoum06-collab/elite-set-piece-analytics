"""
Physical feature engineering for Elite Set-Piece Analytics.
هندسة الميزات الفيزيائية

This module provides functions to calculate physical features
from tracking data, including speed, acceleration, and direction.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.config import Config


class PhysicalFeatures:
    """
    Calculator for physical features from tracking data.
    حاسب الميزات الفيزيائية
    
    This class computes movement-related features like speed,
    acceleration, and direction from position data.
    """
    
    def __init__(self, frame_rate: float = 25.0):
        """
        Initialize the physical features calculator.
        
        Args:
            frame_rate: Tracking data frame rate (Hz)
        """
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate  # Time between frames
    
    def calculate_speed(
        self,
        x: pd.Series,
        y: pd.Series,
        smooth: bool = True,
        window: int = 5
    ) -> pd.Series:
        """
        Calculate speed from position data.
        حساب السرعة
        
        Args:
            x: X coordinates over time
            y: Y coordinates over time
            smooth: Whether to apply smoothing
            window: Smoothing window size
            
        Returns:
            Speed in m/s
        """
        # Calculate displacements
        dx = x.diff()
        dy = y.diff()
        
        # Calculate speed
        distance = np.sqrt(dx ** 2 + dy ** 2)
        speed = distance / self.dt
        
        if smooth:
            speed = speed.rolling(window=window, min_periods=1).mean()
        
        return speed.fillna(0)
    
    def calculate_acceleration(
        self,
        speed: pd.Series,
        smooth: bool = True,
        window: int = 5
    ) -> pd.Series:
        """
        Calculate acceleration from speed data.
        حساب التسارع
        
        Args:
            speed: Speed values over time
            smooth: Whether to apply smoothing
            window: Smoothing window size
            
        Returns:
            Acceleration in m/s²
        """
        acceleration = speed.diff() / self.dt
        
        if smooth:
            acceleration = acceleration.rolling(window=window, min_periods=1).mean()
        
        return acceleration.fillna(0)
    
    def calculate_direction(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> pd.Series:
        """
        Calculate movement direction.
        حساب الاتجاه
        
        Args:
            x: X coordinates over time
            y: Y coordinates over time
            
        Returns:
            Direction in degrees (0-360, where 0 is positive x-axis)
        """
        dx = x.diff()
        dy = y.diff()
        
        direction = np.degrees(np.arctan2(dy, dx))
        direction = (direction + 360) % 360  # Normalize to 0-360
        
        return direction.fillna(0)
    
    def calculate_direction_change(
        self,
        direction: pd.Series
    ) -> pd.Series:
        """
        Calculate rate of direction change.
        
        Args:
            direction: Direction values in degrees
            
        Returns:
            Direction change in degrees per frame
        """
        change = direction.diff().abs()
        
        # Handle wrap-around (e.g., 359 -> 1 degrees)
        change = change.apply(lambda x: min(x, 360 - x) if pd.notna(x) else 0)
        
        return change.fillna(0)
    
    def calculate_distance_covered(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> float:
        """
        Calculate total distance covered.
        
        Args:
            x: X coordinates over time
            y: Y coordinates over time
            
        Returns:
            Total distance in meters
        """
        dx = x.diff()
        dy = y.diff()
        
        distances = np.sqrt(dx ** 2 + dy ** 2)
        
        return float(distances.sum())
    
    def add_physical_features(
        self,
        tracking_data: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y'
    ) -> pd.DataFrame:
        """
        Add all physical features to tracking DataFrame.
        
        Args:
            tracking_data: DataFrame with position data
            x_col: Name of X coordinate column
            y_col: Name of Y coordinate column
            
        Returns:
            DataFrame with physical features added
        """
        result = tracking_data.copy()
        
        if x_col not in result.columns or y_col not in result.columns:
            return result
        
        x = result[x_col]
        y = result[y_col]
        
        result['speed'] = self.calculate_speed(x, y)
        result['acceleration'] = self.calculate_acceleration(result['speed'])
        result['direction'] = self.calculate_direction(x, y)
        result['direction_change'] = self.calculate_direction_change(result['direction'])
        
        return result
    
    def classify_movement(
        self,
        speed: float
    ) -> str:
        """
        Classify movement type based on speed.
        
        Args:
            speed: Speed in m/s
            
        Returns:
            Movement classification string
        """
        if speed < 2:
            return "standing"
        elif speed < 4:
            return "walking"
        elif speed < 5.5:
            return "jogging"
        elif speed < 7:
            return "running"
        else:
            return "sprinting"
    
    def calculate_high_intensity_distance(
        self,
        x: pd.Series,
        y: pd.Series,
        threshold: float = 5.5
    ) -> float:
        """
        Calculate distance covered at high intensity.
        
        Args:
            x: X coordinates over time
            y: Y coordinates over time
            threshold: Speed threshold for high intensity (m/s)
            
        Returns:
            High intensity distance in meters
        """
        speed = self.calculate_speed(x, y, smooth=False)
        
        dx = x.diff()
        dy = y.diff()
        distances = np.sqrt(dx ** 2 + dy ** 2)
        
        high_intensity_mask = speed >= threshold
        
        return float(distances[high_intensity_mask].sum())
