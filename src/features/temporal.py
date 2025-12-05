"""
Temporal feature engineering for Elite Set-Piece Analytics.
هندسة الميزات الزمنية

This module provides functions to calculate temporal features
from event sequences and tracking data.
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from ..utils.config import Config


class TemporalFeatures:
    """
    Calculator for temporal features in football analytics.
    حاسب الميزات الزمنية
    
    This class provides methods to compute time-based features
    useful for sequence analysis and prediction.
    """
    
    MATCH_DURATION = 90 * 60  # 90 minutes in seconds
    HALF_DURATION = 45 * 60  # 45 minutes in seconds
    
    def __init__(self):
        """Initialize the temporal features calculator."""
        self.config = Config()
    
    def time_until_end(
        self,
        current_time: float,
        period: int = 2,
        extra_time: float = 0
    ) -> float:
        """
        Calculate time remaining until end of half/match.
        
        Args:
            current_time: Current match time in seconds
            period: Match period (1 or 2)
            extra_time: Additional time in seconds
            
        Returns:
            Time remaining in seconds
        """
        if period == 1:
            return max(0, self.HALF_DURATION + extra_time - current_time)
        else:
            return max(0, self.MATCH_DURATION + extra_time - current_time)
    
    def time_pressure(
        self,
        current_time: float,
        score_diff: int = 0
    ) -> float:
        """
        Calculate time pressure factor.
        
        Higher values indicate more time pressure (late in game, behind in score).
        
        Args:
            current_time: Current match time in seconds
            score_diff: Score difference (positive = leading)
            
        Returns:
            Time pressure factor (0-1)
        """
        time_factor = current_time / self.MATCH_DURATION
        
        if score_diff < 0:
            # Behind: pressure increases with time
            return min(1.0, time_factor * (1 + abs(score_diff) * 0.1))
        elif score_diff > 0:
            # Ahead: pressure decreases
            return max(0.0, time_factor * (1 - score_diff * 0.1))
        else:
            # Tied: moderate pressure
            return time_factor * 0.5
    
    def calculate_event_duration(
        self,
        events: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.Series:
        """
        Calculate duration between consecutive events.
        
        Args:
            events: DataFrame with event data
            timestamp_col: Name of timestamp column
            
        Returns:
            Series with event durations
        """
        if timestamp_col not in events.columns:
            return pd.Series([0] * len(events))
        
        durations = events[timestamp_col].diff()
        durations = durations.fillna(0)
        
        return durations
    
    def sequence_position(
        self,
        events: pd.DataFrame,
        possession_col: str = 'possession_id'
    ) -> pd.Series:
        """
        Calculate position in possession sequence.
        
        Args:
            events: DataFrame with event data
            possession_col: Name of possession identifier column
            
        Returns:
            Series with sequence positions (1, 2, 3, ...)
        """
        if possession_col not in events.columns:
            return pd.Series(range(1, len(events) + 1))
        
        return events.groupby(possession_col).cumcount() + 1
    
    def calculate_tempo(
        self,
        events: pd.DataFrame,
        window: int = 5,
        timestamp_col: str = 'timestamp'
    ) -> pd.Series:
        """
        Calculate game tempo (events per minute).
        
        Args:
            events: DataFrame with event data
            window: Rolling window size
            timestamp_col: Name of timestamp column
            
        Returns:
            Series with tempo values
        """
        if timestamp_col not in events.columns or len(events) < 2:
            return pd.Series([0] * len(events))
        
        durations = events[timestamp_col].diff()
        
        # Events per minute = 60 / average duration between events
        rolling_avg = durations.rolling(window=window, min_periods=1).mean()
        tempo = 60 / rolling_avg.replace(0, np.inf)
        
        return tempo.clip(0, 100)  # Cap at 100 events per minute
    
    def add_temporal_features(
        self,
        events: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        period_col: str = 'period'
    ) -> pd.DataFrame:
        """
        Add all temporal features to events DataFrame.
        
        Args:
            events: DataFrame with event data
            timestamp_col: Name of timestamp column
            period_col: Name of period column
            
        Returns:
            DataFrame with temporal features added
        """
        result = events.copy()
        
        if timestamp_col in result.columns:
            result['event_duration'] = self.calculate_event_duration(
                result, timestamp_col
            )
            result['tempo'] = self.calculate_tempo(result, 5, timestamp_col)
        
        # Add time until end
        if timestamp_col in result.columns and period_col in result.columns:
            result['time_until_end'] = result.apply(
                lambda row: self.time_until_end(
                    row[timestamp_col],
                    row.get(period_col, 2)
                ),
                axis=1
            )
        
        return result
