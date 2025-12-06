"""
Elite Set-Piece Analytics - Spatial Features
الميزات المكانية لتحليلات الكرات الثابتة

This module calculates spatial features:
- Distance to goal
- Angle to goal
- Distance to nearest defender
- Player density in zones
- Formation compactness
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# Pitch dimensions (FIFA standard in meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32
GOAL_Y_MIN = (PITCH_WIDTH - GOAL_WIDTH) / 2
GOAL_Y_MAX = (PITCH_WIDTH + GOAL_WIDTH) / 2
GOAL_X = PITCH_LENGTH  # Goal line position


def calculate_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all spatial features for set pieces
    حساب جميع الميزات المكانية للكرات الثابتة
    
    Args:
        df: DataFrame with set piece data
    
    Returns:
        DataFrame with added spatial features
    """
    df_feat = df.copy()
    
    # Basic distance and angle features
    df_feat['goal_distance'] = calculate_distance_to_goal(
        df_feat['x'].values, 
        df_feat['y'].values
    )
    
    df_feat['goal_angle'] = calculate_angle_to_goal(
        df_feat['x'].values, 
        df_feat['y'].values
    )
    
    # Zone features
    df_feat['pitch_zone'] = df_feat.apply(
        lambda row: get_pitch_zone(row['x'], row['y']),
        axis=1
    )
    
    # Danger index (composite score)
    df_feat['danger_index'] = calculate_danger_index(
        df_feat['goal_distance'].values,
        df_feat['goal_angle'].values,
        df_feat['n_attackers_in_box'].values,
        df_feat['n_defenders_in_box'].values
    )
    
    # Player density features (from positions if available)
    if 'attacker_positions' in df_feat.columns:
        density_features = df_feat.apply(
            calculate_density_features, axis=1, result_type='expand'
        )
        for col in density_features.columns:
            df_feat[col] = density_features[col]
    
    return df_feat


def calculate_distance_to_goal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate distance from position to center of goal
    حساب المسافة من الموقع إلى مركز المرمى
    
    Args:
        x: X coordinates (0-100 scale)
        y: Y coordinates (0-100 scale)
    
    Returns:
        Array of distances
    """
    # Convert to meters
    x_meters = x / 100 * PITCH_LENGTH
    y_meters = y / 100 * PITCH_WIDTH
    
    # Goal center
    goal_x = PITCH_LENGTH
    goal_y = PITCH_WIDTH / 2
    
    # Calculate Euclidean distance
    distance = np.sqrt((goal_x - x_meters)**2 + (goal_y - y_meters)**2)
    
    return distance


def calculate_angle_to_goal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate angle to goal (visible goal width)
    حساب الزاوية إلى المرمى (عرض المرمى المرئي)
    
    Args:
        x: X coordinates (0-100 scale)
        y: Y coordinates (0-100 scale)
    
    Returns:
        Array of angles in degrees
    """
    # Convert to meters
    x_meters = x / 100 * PITCH_LENGTH
    y_meters = y / 100 * PITCH_WIDTH
    
    # Calculate angle to both goal posts
    dx = GOAL_X - x_meters
    
    # Angle to near post
    dy_near = GOAL_Y_MIN - y_meters
    angle_near = np.arctan2(dy_near, dx)
    
    # Angle to far post
    dy_far = GOAL_Y_MAX - y_meters
    angle_far = np.arctan2(dy_far, dx)
    
    # Total visible angle
    angle = np.abs(angle_far - angle_near)
    
    return np.degrees(angle)


def get_pitch_zone(x: float, y: float) -> str:
    """
    Determine the pitch zone for a position
    تحديد منطقة الملعب للموقع
    
    Args:
        x: X coordinate (0-100 scale)
        y: Y coordinate (0-100 scale)
    
    Returns:
        Zone name string
    """
    # Define zones
    if x < 33:
        x_zone = 'defensive'
    elif x < 66:
        x_zone = 'middle'
    else:
        x_zone = 'attacking'
    
    if y < 30:
        y_zone = 'left'
    elif y < 70:
        y_zone = 'center'
    else:
        y_zone = 'right'
    
    return f"{x_zone}_{y_zone}"


def calculate_danger_index(
    distance: np.ndarray,
    angle: np.ndarray,
    n_attackers: np.ndarray,
    n_defenders: np.ndarray
) -> np.ndarray:
    """
    Calculate a danger index for the set piece
    حساب مؤشر الخطورة للكرة الثابتة
    
    Combines multiple factors into a single score 0-100
    
    Args:
        distance: Distance to goal
        angle: Angle to goal
        n_attackers: Number of attackers in box
        n_defenders: Number of defenders in box
    
    Returns:
        Danger index array (0-100)
    """
    # Distance factor (closer = more dangerous)
    # Normalize: 0m = 100, 50m = 0
    distance_factor = np.clip(100 - (distance * 2), 0, 100)
    
    # Angle factor (wider angle = more dangerous)
    # Normalize: 0° = 0, 45° = 100
    angle_factor = np.clip(angle * 2.2, 0, 100)
    
    # Numerical advantage factor
    advantage = n_attackers - n_defenders
    advantage_factor = np.clip(50 + (advantage * 10), 0, 100)
    
    # Combine with weights
    danger_index = (
        0.40 * distance_factor +
        0.30 * angle_factor +
        0.30 * advantage_factor
    )
    
    return danger_index


def calculate_density_features(row) -> Dict:
    """
    Calculate player density features from position data
    حساب ميزات كثافة اللاعبين من بيانات المواقع
    
    Args:
        row: DataFrame row with player positions
    
    Returns:
        Dictionary of density features
    """
    features = {}
    
    # Get attacker positions
    attackers = row.get('attacker_positions', [])
    defenders = row.get('defender_positions', [])
    
    if not attackers or not isinstance(attackers, list):
        attackers = []
    if not defenders or not isinstance(defenders, list):
        defenders = []
    
    # Calculate attacker centroid
    if attackers:
        att_x = np.mean([p['x'] for p in attackers])
        att_y = np.mean([p['y'] for p in attackers])
        features['attacker_centroid_x'] = att_x
        features['attacker_centroid_y'] = att_y
        
        # Attacker spread (standard deviation)
        features['attacker_spread_x'] = np.std([p['x'] for p in attackers])
        features['attacker_spread_y'] = np.std([p['y'] for p in attackers])
    else:
        features['attacker_centroid_x'] = 50
        features['attacker_centroid_y'] = 50
        features['attacker_spread_x'] = 0
        features['attacker_spread_y'] = 0
    
    # Calculate defender centroid
    if defenders:
        def_x = np.mean([p['x'] for p in defenders])
        def_y = np.mean([p['y'] for p in defenders])
        features['defender_centroid_x'] = def_x
        features['defender_centroid_y'] = def_y
        
        # Defender compactness
        features['defender_spread_x'] = np.std([p['x'] for p in defenders])
        features['defender_spread_y'] = np.std([p['y'] for p in defenders])
    else:
        features['defender_centroid_x'] = 50
        features['defender_centroid_y'] = 50
        features['defender_spread_x'] = 0
        features['defender_spread_y'] = 0
    
    # Distance between centroids
    if attackers and defenders:
        features['centroid_distance'] = np.sqrt(
            (features['attacker_centroid_x'] - features['defender_centroid_x'])**2 +
            (features['attacker_centroid_y'] - features['defender_centroid_y'])**2
        )
    else:
        features['centroid_distance'] = 0
    
    # Density zones (count players in different areas)
    features['attackers_near_post'] = sum(
        1 for p in attackers if p['y'] < 35
    )
    features['attackers_far_post'] = sum(
        1 for p in attackers if p['y'] > 65
    )
    features['attackers_center'] = sum(
        1 for p in attackers if 35 <= p['y'] <= 65
    )
    
    return features


def calculate_distance_to_nearest_defender(
    attacker_pos: Dict,
    defender_positions: List[Dict]
) -> float:
    """
    Calculate distance from attacker to nearest defender
    حساب المسافة من المهاجم إلى أقرب مدافع
    
    Args:
        attacker_pos: Dict with 'x' and 'y' keys
        defender_positions: List of defender position dicts
    
    Returns:
        Distance to nearest defender
    """
    if not defender_positions:
        return 100.0  # No defenders, max distance
    
    att_x, att_y = attacker_pos['x'], attacker_pos['y']
    
    min_distance = float('inf')
    for defender in defender_positions:
        def_x, def_y = defender['x'], defender['y']
        distance = np.sqrt((att_x - def_x)**2 + (att_y - def_y)**2)
        min_distance = min(min_distance, distance)
    
    return min_distance


def calculate_free_space(
    position: Dict,
    all_player_positions: List[Dict],
    radius: float = 5.0
) -> float:
    """
    Calculate free space around a position
    حساب المساحة الحرة حول موقع معين
    
    Args:
        position: Position dict with 'x' and 'y'
        all_player_positions: List of all player positions
        radius: Radius to consider
    
    Returns:
        Free space score (higher = more space)
    """
    x, y = position['x'], position['y']
    
    players_in_radius = 0
    for player in all_player_positions:
        distance = np.sqrt((x - player['x'])**2 + (y - player['y'])**2)
        if distance < radius:
            players_in_radius += 1
    
    # Free space is inverse of density
    free_space = 100 / (1 + players_in_radius)
    
    return free_space


def get_zone_control(
    attacker_positions: List[Dict],
    defender_positions: List[Dict],
    zone_bounds: Tuple[float, float, float, float]
) -> Dict:
    """
    Calculate zone control metrics
    حساب مقاييس السيطرة على المنطقة
    
    Args:
        attacker_positions: List of attacker positions
        defender_positions: List of defender positions
        zone_bounds: Tuple of (x_min, x_max, y_min, y_max)
    
    Returns:
        Dict with zone control metrics
    """
    x_min, x_max, y_min, y_max = zone_bounds
    
    # Count players in zone
    attackers_in_zone = sum(
        1 for p in attacker_positions
        if x_min <= p['x'] <= x_max and y_min <= p['y'] <= y_max
    )
    
    defenders_in_zone = sum(
        1 for p in defender_positions
        if x_min <= p['x'] <= x_max and y_min <= p['y'] <= y_max
    )
    
    total = attackers_in_zone + defenders_in_zone
    
    return {
        'attackers_in_zone': attackers_in_zone,
        'defenders_in_zone': defenders_in_zone,
        'control_ratio': attackers_in_zone / total if total > 0 else 0.5,
        'numerical_advantage': attackers_in_zone - defenders_in_zone
    }


if __name__ == "__main__":
    # Test the spatial features
    import sys
    sys.path.append('..')
    from data.loaders import load_wyscout_data
    from data.extractors import extract_set_pieces
    
    print("=" * 50)
    print("Testing Spatial Features")
    print("=" * 50)
    
    df = load_wyscout_data()
    set_pieces = extract_set_pieces(df)
    
    # Calculate spatial features
    set_pieces_with_features = calculate_spatial_features(set_pieces)
    
    print(f"\nNew columns added:")
    new_cols = [col for col in set_pieces_with_features.columns 
                if col not in set_pieces.columns]
    print(f"  {new_cols}")
    
    print(f"\nDanger index stats:")
    print(f"  Mean: {set_pieces_with_features['danger_index'].mean():.2f}")
    print(f"  Std:  {set_pieces_with_features['danger_index'].std():.2f}")
    print(f"  Min:  {set_pieces_with_features['danger_index'].min():.2f}")
    print(f"  Max:  {set_pieces_with_features['danger_index'].max():.2f}")
