"""
Spatial feature engineering for Elite Set-Piece Analytics.
هندسة الميزات المكانية

This module provides functions to calculate spatial features for
set-piece analysis, including:
- Distance calculations
- Angle calculations
- Zone classifications
- Player density metrics

Example usage:
    >>> from src.features.spatial import SpatialFeatures
    >>> spatial = SpatialFeatures()
    >>> features = spatial.calculate_all_features(positions_df)
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.spatial import distance

from ..utils.config import Config


class SpatialFeatures:
    """
    Calculator for spatial features in football analytics.
    حاسب الميزات المكانية
    
    This class provides methods to compute various spatial features
    that are useful for set-piece analysis and first receiver prediction.
    
    Attributes:
        pitch_length: Length of the pitch in meters
        pitch_width: Width of the pitch in meters
        
    Example:
        >>> spatial = SpatialFeatures()
        >>> dist = spatial.distance_to_goal(45.0, 30.0)
        >>> angle = spatial.angle_to_goal(45.0, 30.0)
    """
    
    def __init__(
        self,
        pitch_length: float = Config.PITCH_LENGTH,
        pitch_width: float = Config.PITCH_WIDTH
    ):
        """
        Initialize the spatial features calculator.
        
        Args:
            pitch_length: Length of the pitch in meters
            pitch_width: Width of the pitch in meters
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        
        # Goal positions (center of each goal)
        self.goal_left = (0, pitch_width / 2)
        self.goal_right = (pitch_length, pitch_width / 2)
        
        # Penalty spot positions
        self.penalty_left = (11, pitch_width / 2)
        self.penalty_right = (pitch_length - 11, pitch_width / 2)
    
    def distance_to_goal(
        self, 
        x: float, 
        y: float, 
        attacking_right: bool = True
    ) -> float:
        """
        Calculate distance from a point to the attacking goal.
        حساب المسافة إلى المرمى
        
        Args:
            x: X coordinate
            y: Y coordinate
            attacking_right: If True, attacking the right goal
            
        Returns:
            Distance to goal in meters
        """
        goal = self.goal_right if attacking_right else self.goal_left
        return np.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2)
    
    def angle_to_goal(
        self, 
        x: float, 
        y: float, 
        attacking_right: bool = True
    ) -> float:
        """
        Calculate angle to goal from a point.
        حساب الزاوية إلى المرمى
        
        The angle is measured from the line to the goal center,
        with 0 being directly in front and 90 being from the side.
        
        Args:
            x: X coordinate
            y: Y coordinate
            attacking_right: If True, attacking the right goal
            
        Returns:
            Angle to goal in degrees (0-180)
        """
        goal = self.goal_right if attacking_right else self.goal_left
        
        # Calculate angle using atan2
        dx = goal[0] - x
        dy = goal[1] - y
        
        angle_rad = np.arctan2(abs(dy), abs(dx))
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def visible_goal_angle(
        self,
        x: float,
        y: float,
        attacking_right: bool = True
    ) -> float:
        """
        Calculate the visible angle of the goal from a position.
        زاوية المرمى المرئية
        
        This is the angle subtended by the goal posts as seen
        from the player's position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            attacking_right: If True, attacking the right goal
            
        Returns:
            Visible goal angle in degrees
        """
        goal_width = Config.GOAL_WIDTH
        goal_x = self.pitch_length if attacking_right else 0
        goal_center_y = self.pitch_width / 2
        
        # Goal post positions
        post_left_y = goal_center_y - goal_width / 2
        post_right_y = goal_center_y + goal_width / 2
        
        # Angles to each post
        angle_left = np.arctan2(post_left_y - y, goal_x - x)
        angle_right = np.arctan2(post_right_y - y, goal_x - x)
        
        # Visible angle is the difference
        visible_angle = abs(angle_right - angle_left)
        
        return np.degrees(visible_angle)
    
    def distance_between_points(
        self,
        x1: float, y1: float,
        x2: float, y2: float
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            x1, y1: First point coordinates
            x2, y2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def player_density(
        self,
        x: float,
        y: float,
        all_positions: pd.DataFrame,
        radius: float = 5.0
    ) -> int:
        """
        Calculate the number of players within a radius.
        كثافة اللاعبين
        
        Args:
            x: Reference point X coordinate
            y: Reference point Y coordinate
            all_positions: DataFrame with 'x' and 'y' columns
            radius: Radius to consider in meters
            
        Returns:
            Number of players within the radius
        """
        if all_positions.empty or 'x' not in all_positions.columns:
            return 0
        
        distances = np.sqrt(
            (all_positions['x'] - x) ** 2 +
            (all_positions['y'] - y) ** 2
        )
        
        return int((distances <= radius).sum())
    
    def nearest_player_distance(
        self,
        x: float,
        y: float,
        player_positions: pd.DataFrame
    ) -> float:
        """
        Find distance to nearest player.
        المسافة إلى أقرب لاعب
        
        Args:
            x: Reference point X coordinate
            y: Reference point Y coordinate
            player_positions: DataFrame with 'x' and 'y' columns
            
        Returns:
            Distance to nearest player in meters
        """
        if player_positions.empty or 'x' not in player_positions.columns:
            return float('inf')
        
        distances = np.sqrt(
            (player_positions['x'] - x) ** 2 +
            (player_positions['y'] - y) ** 2
        )
        
        # Exclude distance 0 (the player itself)
        distances = distances[distances > 0.1]
        
        return float(distances.min()) if len(distances) > 0 else float('inf')
    
    def classify_zone(self, x: float, y: float) -> str:
        """
        Classify a position into a pitch zone.
        تصنيف المنطقة
        
        Zones are divided into:
        - defensive_third: x < 35
        - middle_third: 35 <= x < 70
        - attacking_third: x >= 70
        
        Combined with left/center/right based on y position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Zone classification string
        """
        # Vertical zones
        if x < self.pitch_length / 3:
            zone_x = "defensive"
        elif x < 2 * self.pitch_length / 3:
            zone_x = "middle"
        else:
            zone_x = "attacking"
        
        # Horizontal zones
        if y < self.pitch_width / 3:
            zone_y = "left"
        elif y < 2 * self.pitch_width / 3:
            zone_y = "center"
        else:
            zone_y = "right"
        
        return f"{zone_x}_{zone_y}"
    
    def is_in_penalty_area(
        self,
        x: float,
        y: float,
        attacking_right: bool = True
    ) -> bool:
        """
        Check if a position is in the penalty area.
        هل الموقع في منطقة الجزاء
        
        Args:
            x: X coordinate
            y: Y coordinate
            attacking_right: If True, check attacking penalty area
            
        Returns:
            True if in penalty area
        """
        penalty_length = Config.PENALTY_AREA_LENGTH
        penalty_width = Config.PENALTY_AREA_WIDTH
        
        center_y = self.pitch_width / 2
        y_min = center_y - penalty_width / 2
        y_max = center_y + penalty_width / 2
        
        if attacking_right:
            return (x >= self.pitch_length - penalty_length) and (y_min <= y <= y_max)
        else:
            return (x <= penalty_length) and (y_min <= y <= y_max)
    
    def is_in_goal_area(
        self,
        x: float,
        y: float,
        attacking_right: bool = True
    ) -> bool:
        """
        Check if a position is in the goal area (6-yard box).
        
        Args:
            x: X coordinate
            y: Y coordinate
            attacking_right: If True, check attacking goal area
            
        Returns:
            True if in goal area
        """
        goal_area_length = Config.GOAL_AREA_LENGTH
        goal_area_width = Config.GOAL_AREA_WIDTH
        
        center_y = self.pitch_width / 2
        y_min = center_y - goal_area_width / 2
        y_max = center_y + goal_area_width / 2
        
        if attacking_right:
            return (x >= self.pitch_length - goal_area_length) and (y_min <= y <= y_max)
        else:
            return (x <= goal_area_length) and (y_min <= y <= y_max)
    
    def calculate_all_features(
        self,
        positions: pd.DataFrame,
        ball_x: float,
        ball_y: float,
        team_positions: Optional[pd.DataFrame] = None,
        opponent_positions: Optional[pd.DataFrame] = None,
        attacking_right: bool = True
    ) -> pd.DataFrame:
        """
        Calculate all spatial features for player positions.
        حساب جميع الميزات المكانية
        
        Args:
            positions: DataFrame with player positions (x, y columns)
            ball_x: Ball X coordinate
            ball_y: Ball Y coordinate
            team_positions: Positions of teammates (optional)
            opponent_positions: Positions of opponents (optional)
            attacking_right: Direction of attack
            
        Returns:
            DataFrame with all spatial features added
        """
        result = positions.copy()
        
        if 'x' not in result.columns or 'y' not in result.columns:
            return result
        
        # Distance and angle to goal
        result['distance_to_goal'] = result.apply(
            lambda row: self.distance_to_goal(row['x'], row['y'], attacking_right),
            axis=1
        )
        
        result['angle_to_goal'] = result.apply(
            lambda row: self.angle_to_goal(row['x'], row['y'], attacking_right),
            axis=1
        )
        
        result['visible_goal_angle'] = result.apply(
            lambda row: self.visible_goal_angle(row['x'], row['y'], attacking_right),
            axis=1
        )
        
        # Distance to ball
        result['distance_to_ball'] = result.apply(
            lambda row: self.distance_between_points(row['x'], row['y'], ball_x, ball_y),
            axis=1
        )
        
        # Zone classification
        result['zone'] = result.apply(
            lambda row: self.classify_zone(row['x'], row['y']),
            axis=1
        )
        
        # In penalty area
        result['in_penalty_area'] = result.apply(
            lambda row: self.is_in_penalty_area(row['x'], row['y'], attacking_right),
            axis=1
        )
        
        # Player density (using all positions)
        all_positions = positions
        result['density_5m'] = result.apply(
            lambda row: self.player_density(row['x'], row['y'], all_positions, 5.0),
            axis=1
        )
        result['density_10m'] = result.apply(
            lambda row: self.player_density(row['x'], row['y'], all_positions, 10.0),
            axis=1
        )
        
        # Nearest teammate and opponent distances
        if team_positions is not None and not team_positions.empty:
            result['nearest_teammate_dist'] = result.apply(
                lambda row: self.nearest_player_distance(row['x'], row['y'], team_positions),
                axis=1
            )
        
        if opponent_positions is not None and not opponent_positions.empty:
            result['nearest_opponent_dist'] = result.apply(
                lambda row: self.nearest_player_distance(row['x'], row['y'], opponent_positions),
                axis=1
            )
        
        return result


def calculate_distance_matrix(positions: pd.DataFrame) -> np.ndarray:
    """
    Calculate pairwise distances between all players.
    
    Args:
        positions: DataFrame with 'x' and 'y' columns
        
    Returns:
        NxN distance matrix
    """
    if positions.empty or 'x' not in positions.columns:
        return np.array([])
    
    coords = positions[['x', 'y']].values
    return distance.cdist(coords, coords, 'euclidean')


def calculate_convex_hull_area(positions: pd.DataFrame) -> float:
    """
    Calculate the area of the convex hull formed by player positions.
    
    This is useful for measuring team shape/compactness.
    
    Args:
        positions: DataFrame with 'x' and 'y' columns
        
    Returns:
        Area of convex hull in square meters
    """
    from scipy.spatial import ConvexHull
    
    if len(positions) < 3:
        return 0.0
    
    try:
        coords = positions[['x', 'y']].values
        hull = ConvexHull(coords)
        return float(hull.volume)  # In 2D, volume is area
    except Exception:
        return 0.0


def calculate_centroid(positions: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate the centroid of player positions.
    
    Args:
        positions: DataFrame with 'x' and 'y' columns
        
    Returns:
        Tuple of (x, y) centroid coordinates
    """
    if positions.empty or 'x' not in positions.columns:
        return (0.0, 0.0)
    
    return (
        float(positions['x'].mean()),
        float(positions['y'].mean())
    )
