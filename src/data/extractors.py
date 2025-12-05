"""
Set-piece extraction pipeline for Elite Set-Piece Analytics.
خط أنابيب استخراج الكرات الثابتة

This module provides functionality to extract and classify different types
of set-pieces from event data, including:
- Corner kicks
- Free kicks
- Throw-ins
- Penalty kicks

Example usage:
    >>> from src.data.extractors import SetPieceExtractor
    >>> extractor = SetPieceExtractor()
    >>> corners = extractor.extract_corners(events)
    >>> free_kicks = extractor.extract_free_kicks(events)
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from ..utils.config import Config

logger = logging.getLogger(__name__)


class SetPieceExtractor:
    """
    Extractor for set-piece events from football event data.
    مستخرج الكرات الثابتة
    
    This class provides methods to identify and extract various set-piece
    situations from event data, categorizing them by type and context.
    
    Attributes:
        pitch_length: Length of the pitch in meters
        pitch_width: Width of the pitch in meters
        
    Example:
        >>> extractor = SetPieceExtractor()
        >>> all_set_pieces = extractor.extract_all(events_df)
        >>> corners = all_set_pieces[all_set_pieces['set_piece_type'] == 'corner']
    """
    
    # Set-piece type identifiers
    SET_PIECE_TYPES = {
        "corner": ["corner", "corner kick"],
        "free_kick": ["free kick", "free_kick", "freekick"],
        "throw_in": ["throw-in", "throw in", "throw_in", "throwin"],
        "penalty": ["penalty", "penalty kick"],
        "goal_kick": ["goal kick", "goal_kick", "goalkick"],
    }
    
    # Corner kick zones (relative to pitch dimensions)
    CORNER_ZONES = {
        "left_attacking": {"x_range": (0, 2), "y_range": (0, 2)},
        "right_attacking": {"x_range": (0, 2), "y_range": (66, 68)},
    }
    
    def __init__(
        self, 
        pitch_length: float = Config.PITCH_LENGTH,
        pitch_width: float = Config.PITCH_WIDTH
    ):
        """
        Initialize the set-piece extractor.
        
        Args:
            pitch_length: Length of the pitch in meters
            pitch_width: Width of the pitch in meters
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_all(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all set-pieces from event data.
        
        Args:
            events: DataFrame containing event data
            
        Returns:
            DataFrame containing all set-piece events with classifications
        """
        if events.empty:
            self.logger.warning("Empty events DataFrame provided")
            return pd.DataFrame()
        
        # Extract each type
        corners = self.extract_corners(events)
        free_kicks = self.extract_free_kicks(events)
        throw_ins = self.extract_throw_ins(events)
        penalties = self.extract_penalties(events)
        goal_kicks = self.extract_goal_kicks(events)
        
        # Combine all set-pieces
        all_set_pieces = pd.concat([
            corners, free_kicks, throw_ins, penalties, goal_kicks
        ], ignore_index=True)
        
        if not all_set_pieces.empty:
            all_set_pieces = all_set_pieces.sort_values(
                by=["match_id", "period", "timestamp"] 
                if all(c in all_set_pieces.columns for c in ["match_id", "period", "timestamp"])
                else ["timestamp"] if "timestamp" in all_set_pieces.columns
                else []
            ).reset_index(drop=True)
        
        self.logger.info(f"Extracted {len(all_set_pieces)} total set-pieces")
        return all_set_pieces
    
    def extract_corners(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract corner kick events.
        استخراج ركلات الركنية
        
        Args:
            events: DataFrame containing event data
            
        Returns:
            DataFrame containing corner kick events
        """
        corners = self._filter_by_type(events, self.SET_PIECE_TYPES["corner"])
        
        if corners.empty:
            # Try to identify corners by position
            corners = self._identify_corners_by_position(events)
        
        if not corners.empty:
            corners["set_piece_type"] = "corner"
            corners = self._classify_corner_side(corners)
            corners = self._add_corner_context(corners, events)
        
        self.logger.info(f"Extracted {len(corners)} corner kicks")
        return corners
    
    def extract_free_kicks(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract free kick events.
        استخراج الركلات الحرة
        
        Args:
            events: DataFrame containing event data
            
        Returns:
            DataFrame containing free kick events
        """
        free_kicks = self._filter_by_type(events, self.SET_PIECE_TYPES["free_kick"])
        
        if not free_kicks.empty:
            free_kicks["set_piece_type"] = "free_kick"
            free_kicks = self._classify_free_kick_zone(free_kicks)
            free_kicks = self._classify_free_kick_type(free_kicks)
        
        self.logger.info(f"Extracted {len(free_kicks)} free kicks")
        return free_kicks
    
    def extract_throw_ins(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract throw-in events.
        استخراج رميات التماس
        
        Args:
            events: DataFrame containing event data
            
        Returns:
            DataFrame containing throw-in events
        """
        throw_ins = self._filter_by_type(events, self.SET_PIECE_TYPES["throw_in"])
        
        if not throw_ins.empty:
            throw_ins["set_piece_type"] = "throw_in"
            throw_ins = self._classify_throw_in_zone(throw_ins)
        
        self.logger.info(f"Extracted {len(throw_ins)} throw-ins")
        return throw_ins
    
    def extract_penalties(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract penalty kick events.
        استخراج ركلات الجزاء
        
        Args:
            events: DataFrame containing event data
            
        Returns:
            DataFrame containing penalty kick events
        """
        penalties = self._filter_by_type(events, self.SET_PIECE_TYPES["penalty"])
        
        if not penalties.empty:
            penalties["set_piece_type"] = "penalty"
            penalties = self._add_penalty_context(penalties, events)
        
        self.logger.info(f"Extracted {len(penalties)} penalties")
        return penalties
    
    def extract_goal_kicks(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract goal kick events.
        استخراج ركلات المرمى
        
        Args:
            events: DataFrame containing event data
            
        Returns:
            DataFrame containing goal kick events
        """
        goal_kicks = self._filter_by_type(events, self.SET_PIECE_TYPES["goal_kick"])
        
        if not goal_kicks.empty:
            goal_kicks["set_piece_type"] = "goal_kick"
        
        self.logger.info(f"Extracted {len(goal_kicks)} goal kicks")
        return goal_kicks
    
    def _filter_by_type(
        self, 
        events: pd.DataFrame, 
        type_keywords: List[str]
    ) -> pd.DataFrame:
        """Filter events by type keywords."""
        if "event_type" not in events.columns:
            return pd.DataFrame()
        
        mask = events["event_type"].astype(str).str.lower().str.contains(
            "|".join(type_keywords), na=False
        )
        
        return events[mask].copy()
    
    def _identify_corners_by_position(
        self, 
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """Identify corners by position when not explicitly labeled."""
        if "x" not in events.columns or "y" not in events.columns:
            return pd.DataFrame()
        
        # Corner positions are at the corners of the pitch
        corner_threshold = 3  # meters from corner
        
        corner_mask = (
            ((events["x"] < corner_threshold) | 
             (events["x"] > self.pitch_length - corner_threshold)) &
            ((events["y"] < corner_threshold) | 
             (events["y"] > self.pitch_width - corner_threshold))
        )
        
        return events[corner_mask].copy()
    
    def _classify_corner_side(self, corners: pd.DataFrame) -> pd.DataFrame:
        """Classify corners as left or right side."""
        if "y" not in corners.columns:
            return corners
        
        mid_y = self.pitch_width / 2
        corners["corner_side"] = np.where(
            corners["y"] < mid_y, "left", "right"
        )
        
        return corners
    
    def _add_corner_context(
        self, 
        corners: pd.DataFrame, 
        all_events: pd.DataFrame
    ) -> pd.DataFrame:
        """Add context information to corner kicks."""
        # This would add information about subsequent events, receivers, etc.
        # Implementation depends on the specific data format
        corners["corner_type"] = "standard"  # placeholder
        return corners
    
    def _classify_free_kick_zone(
        self, 
        free_kicks: pd.DataFrame
    ) -> pd.DataFrame:
        """Classify free kicks by zone on the pitch."""
        if "x" not in free_kicks.columns:
            return free_kicks
        
        # Define zones based on distance from goal
        conditions = [
            free_kicks["x"] < 20,
            (free_kicks["x"] >= 20) & (free_kicks["x"] < 35),
            free_kicks["x"] >= 35,
        ]
        choices = ["dangerous", "promising", "deep"]
        
        free_kicks["zone"] = np.select(conditions, choices, default="unknown")
        
        return free_kicks
    
    def _classify_free_kick_type(
        self, 
        free_kicks: pd.DataFrame
    ) -> pd.DataFrame:
        """Classify free kicks as direct or indirect."""
        if "x" not in free_kicks.columns or "y" not in free_kicks.columns:
            return free_kicks
        
        # Direct free kicks are typically within shooting range
        shooting_range = 30  # meters from goal
        
        distance_to_goal = np.sqrt(
            (free_kicks["x"] - 0) ** 2 + 
            (free_kicks["y"] - self.pitch_width / 2) ** 2
        )
        
        free_kicks["free_kick_type"] = np.where(
            distance_to_goal < shooting_range, "direct", "indirect"
        )
        
        return free_kicks
    
    def _classify_throw_in_zone(
        self, 
        throw_ins: pd.DataFrame
    ) -> pd.DataFrame:
        """Classify throw-ins by zone."""
        if "x" not in throw_ins.columns:
            return throw_ins
        
        # Zones: defensive third, middle third, attacking third
        third_length = self.pitch_length / 3
        
        conditions = [
            throw_ins["x"] < third_length,
            (throw_ins["x"] >= third_length) & (throw_ins["x"] < 2 * third_length),
            throw_ins["x"] >= 2 * third_length,
        ]
        choices = ["defensive", "middle", "attacking"]
        
        throw_ins["zone"] = np.select(conditions, choices, default="unknown")
        
        return throw_ins
    
    def _add_penalty_context(
        self, 
        penalties: pd.DataFrame,
        all_events: pd.DataFrame
    ) -> pd.DataFrame:
        """Add context information to penalties."""
        # Add information about penalty outcome
        penalties["penalty_outcome"] = "unknown"  # placeholder
        return penalties
    
    def get_set_piece_sequences(
        self, 
        events: pd.DataFrame,
        set_piece_idx: int,
        window_before: int = 3,
        window_after: int = 10
    ) -> pd.DataFrame:
        """
        Get the sequence of events around a set-piece.
        
        Args:
            events: Full events DataFrame
            set_piece_idx: Index of the set-piece event
            window_before: Number of events before the set-piece
            window_after: Number of events after the set-piece
            
        Returns:
            DataFrame containing the event sequence
        """
        start_idx = max(0, set_piece_idx - window_before)
        end_idx = min(len(events), set_piece_idx + window_after + 1)
        
        return events.iloc[start_idx:end_idx].copy()
    
    def calculate_set_piece_statistics(
        self, 
        set_pieces: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate statistics for set-pieces.
        
        Args:
            set_pieces: DataFrame containing set-piece events
            
        Returns:
            Dictionary containing statistics
        """
        if set_pieces.empty:
            return {}
        
        stats = {
            "total_count": len(set_pieces),
            "by_type": set_pieces["set_piece_type"].value_counts().to_dict()
            if "set_piece_type" in set_pieces.columns else {},
        }
        
        if "zone" in set_pieces.columns:
            stats["by_zone"] = set_pieces["zone"].value_counts().to_dict()
        
        return stats


class DataPreprocessor:
    """
    Preprocessor for football event data.
    معالج البيانات
    
    This class handles data cleaning, validation, and transformation
    to prepare data for feature engineering and modeling.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = self.clean_data(df)
        df = self.validate_coordinates(df)
        df = self.add_derived_columns(df)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and handle missing values."""
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Handle missing values in critical columns
        critical_cols = ["event_type", "x", "y", "timestamp"]
        for col in critical_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        return df
    
    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix coordinate values."""
        if "x" in df.columns:
            df["x"] = df["x"].clip(0, self.config.PITCH_LENGTH)
        if "y" in df.columns:
            df["y"] = df["y"].clip(0, self.config.PITCH_WIDTH)
        if "end_x" in df.columns:
            df["end_x"] = df["end_x"].clip(0, self.config.PITCH_LENGTH)
        if "end_y" in df.columns:
            df["end_y"] = df["end_y"].clip(0, self.config.PITCH_WIDTH)
        
        return df
    
    def add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns."""
        if "x" in df.columns and "y" in df.columns:
            # Distance from center
            df["distance_from_center"] = np.sqrt(
                (df["x"] - self.config.PITCH_LENGTH / 2) ** 2 +
                (df["y"] - self.config.PITCH_WIDTH / 2) ** 2
            )
            
            # Distance to nearest goal
            goal_1_x, goal_1_y = 0, self.config.PITCH_WIDTH / 2
            goal_2_x, goal_2_y = self.config.PITCH_LENGTH, self.config.PITCH_WIDTH / 2
            
            dist_1 = np.sqrt((df["x"] - goal_1_x) ** 2 + (df["y"] - goal_1_y) ** 2)
            dist_2 = np.sqrt((df["x"] - goal_2_x) ** 2 + (df["y"] - goal_2_y) ** 2)
            
            df["distance_to_goal"] = np.minimum(dist_1, dist_2)
        
        return df
