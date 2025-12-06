"""
Elite Set-Piece Analytics - Physical Features
الميزات الفيزيائية لتحليلات الكرات الثابتة

This module calculates physical features:
- Player height for headers
- Speed metrics
- Position-based features
- Physical attributes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


# Player physical profiles (simulated based on typical attributes by position)
POSITION_PROFILES = {
    'GK': {'height': 188, 'speed': 55, 'jumping': 70, 'strength': 65},
    'CB': {'height': 186, 'speed': 60, 'jumping': 75, 'strength': 80},
    'RB': {'height': 177, 'speed': 78, 'jumping': 68, 'strength': 68},
    'LB': {'height': 178, 'speed': 77, 'jumping': 67, 'strength': 67},
    'CDM': {'height': 182, 'speed': 68, 'jumping': 72, 'strength': 78},
    'CM': {'height': 180, 'speed': 72, 'jumping': 68, 'strength': 72},
    'CAM': {'height': 176, 'speed': 76, 'jumping': 65, 'strength': 65},
    'RW': {'height': 175, 'speed': 85, 'jumping': 65, 'strength': 58},
    'LW': {'height': 175, 'speed': 84, 'jumping': 64, 'strength': 57},
    'ST': {'height': 183, 'speed': 80, 'jumping': 78, 'strength': 75},
    'CF': {'height': 180, 'speed': 78, 'jumping': 75, 'strength': 72},
}


def calculate_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all physical features for set pieces
    حساب جميع الميزات الفيزيائية للكرات الثابتة
    
    Args:
        df: DataFrame with set piece data
    
    Returns:
        DataFrame with added physical features
    """
    df_feat = df.copy()
    
    # Team physical metrics (based on attackers in box)
    df_feat = add_team_physical_metrics(df_feat)
    
    # Delivery-specific physical features
    df_feat = add_delivery_physical_features(df_feat)
    
    # Header probability based on delivery
    df_feat['header_probability'] = df_feat.apply(
        calculate_header_probability, axis=1
    )
    
    return df_feat


def add_team_physical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add team-level physical metrics based on players in the box
    إضافة مقاييس فيزيائية على مستوى الفريق
    
    Args:
        df: DataFrame with set piece data
    
    Returns:
        DataFrame with team physical metrics
    """
    df_phys = df.copy()
    
    # Estimate physical metrics from attacker positions
    # In real implementation, this would use actual player data
    
    physical_metrics = df_phys.apply(
        estimate_team_physical_metrics, axis=1, result_type='expand'
    )
    
    for col in physical_metrics.columns:
        df_phys[col] = physical_metrics[col]
    
    return df_phys


def estimate_team_physical_metrics(row) -> Dict:
    """
    Estimate team physical metrics for a set piece
    تقدير المقاييس الفيزيائية للفريق
    
    Args:
        row: DataFrame row
    
    Returns:
        Dictionary of physical metrics
    """
    n_attackers = row.get('n_attackers_in_box', 5)
    n_defenders = row.get('n_defenders_in_box', 6)
    set_piece_type = row.get('type', 'corner')
    
    # Simulate player heights based on typical set piece lineups
    # Taller players are usually selected for set pieces
    
    # Attacker heights (biased toward tall players)
    np.random.seed(int(row.get('event_id', 0)) % 1000)
    attacker_heights = np.random.normal(184, 5, n_attackers)
    attacker_heights = np.clip(attacker_heights, 170, 200)
    
    # Defender heights
    defender_heights = np.random.normal(182, 5, n_defenders)
    defender_heights = np.clip(defender_heights, 168, 198)
    
    metrics = {
        'avg_attacker_height': np.mean(attacker_heights),
        'max_attacker_height': np.max(attacker_heights) if len(attacker_heights) > 0 else 180,
        'avg_defender_height': np.mean(defender_heights),
        'max_defender_height': np.max(defender_heights) if len(defender_heights) > 0 else 180,
        'height_advantage': np.mean(attacker_heights) - np.mean(defender_heights),
        'tallest_in_box': max(np.max(attacker_heights), np.max(defender_heights)) if len(attacker_heights) > 0 and len(defender_heights) > 0 else 185,
    }
    
    # Jumping ability (correlated with height but not perfectly)
    metrics['team_jumping'] = 70 + (metrics['avg_attacker_height'] - 180) * 0.5
    metrics['team_strength'] = 75 + n_attackers * 1.5
    
    return metrics


def add_delivery_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physical features related to delivery type
    إضافة ميزات فيزيائية متعلقة بنوع التسديدة
    
    Args:
        df: DataFrame with set piece data
    
    Returns:
        DataFrame with delivery physical features
    """
    df_del = df.copy()
    
    # Delivery speed estimates (m/s)
    delivery_speeds = {
        'inswinger': 22,
        'outswinger': 21,
        'driven': 28,
        'lofted': 18,
        'short': 8,
        'unknown': 20
    }
    
    df_del['delivery_speed'] = df_del['delivery_type'].fillna('unknown').map(
        lambda x: delivery_speeds.get(x, 20)
    )
    
    # Flight time estimate (based on distance and speed)
    df_del['estimated_flight_time'] = df_del.apply(
        estimate_flight_time, axis=1
    )
    
    # Arrival height estimate
    arrival_heights = {
        'inswinger': 2.5,
        'outswinger': 2.3,
        'driven': 1.5,
        'lofted': 2.8,
        'short': 1.0,
        'unknown': 2.0
    }
    
    df_del['estimated_arrival_height'] = df_del['delivery_type'].fillna('unknown').map(
        lambda x: arrival_heights.get(x, 2.0)
    )
    
    return df_del


def estimate_flight_time(row) -> float:
    """
    Estimate ball flight time based on delivery
    تقدير وقت طيران الكرة
    
    Args:
        row: DataFrame row
    
    Returns:
        Estimated flight time in seconds
    """
    delivery_type = row.get('delivery_type', 'unknown')
    distance = row.get('distance_to_goal', 30)
    
    # Simplified physics
    if delivery_type == 'short':
        return 1.0
    elif delivery_type == 'driven':
        return distance / 28  # Fast delivery
    elif delivery_type in ['lofted', 'inswinger', 'outswinger']:
        return distance / 18 + 0.5  # Higher trajectory adds time
    else:
        return distance / 20


def calculate_header_probability(row) -> float:
    """
    Calculate probability that the first touch will be a header
    حساب احتمالية أن تكون اللمسة الأولى رأسية
    
    Args:
        row: DataFrame row
    
    Returns:
        Header probability (0-1)
    """
    delivery_type = row.get('delivery_type', 'unknown')
    avg_height = row.get('avg_attacker_height', 180)
    arrival_height = row.get('estimated_arrival_height', 2.0)
    
    # Base probability by delivery type
    base_probs = {
        'inswinger': 0.65,
        'outswinger': 0.60,
        'lofted': 0.70,
        'driven': 0.35,
        'short': 0.05,
        'unknown': 0.50
    }
    
    base_prob = base_probs.get(delivery_type, 0.50)
    
    # Adjust by height (taller teams more likely to contest aerially)
    height_factor = (avg_height - 175) / 20  # 0 to 1 scale
    height_factor = np.clip(height_factor, 0, 1)
    
    # Adjust by arrival height
    if arrival_height > 2.0:
        arrival_factor = 1.1
    elif arrival_height < 1.5:
        arrival_factor = 0.7
    else:
        arrival_factor = 1.0
    
    probability = base_prob * (0.8 + 0.2 * height_factor) * arrival_factor
    
    return np.clip(probability, 0, 1)


def get_player_physical_profile(position: str) -> Dict:
    """
    Get physical profile for a player based on position
    الحصول على الملف الفيزيائي للاعب بناءً على مركزه
    
    Args:
        position: Player position code
    
    Returns:
        Dictionary with physical attributes
    """
    return POSITION_PROFILES.get(position, POSITION_PROFILES['CM'])


def calculate_aerial_duel_probability(
    attacker_profile: Dict,
    defender_profile: Dict,
    arrival_height: float
) -> float:
    """
    Calculate probability of winning an aerial duel
    حساب احتمالية الفوز في المبارزة الهوائية
    
    Args:
        attacker_profile: Attacker physical profile
        defender_profile: Defender physical profile
        arrival_height: Ball arrival height
    
    Returns:
        Win probability for attacker (0-1)
    """
    # Height advantage
    height_diff = attacker_profile['height'] - defender_profile['height']
    height_factor = 0.5 + (height_diff / 20)  # ±10cm = ±0.5
    
    # Jumping ability
    jump_diff = attacker_profile['jumping'] - defender_profile['jumping']
    jump_factor = 0.5 + (jump_diff / 40)
    
    # Strength factor
    strength_diff = attacker_profile['strength'] - defender_profile['strength']
    strength_factor = 0.5 + (strength_diff / 40)
    
    # Combine factors
    probability = (
        0.40 * height_factor +
        0.35 * jump_factor +
        0.25 * strength_factor
    )
    
    # Arrival height bonus (high ball favors taller/better jumpers)
    if arrival_height > 2.5:
        probability *= 1.1
    
    return np.clip(probability, 0.1, 0.9)


def calculate_run_speed_advantage(
    attacker_positions: List[Dict],
    defender_positions: List[Dict]
) -> float:
    """
    Calculate speed advantage based on starting positions
    حساب ميزة السرعة بناءً على المواقع البدائية
    
    Args:
        attacker_positions: List of attacker positions
        defender_positions: List of defender positions
    
    Returns:
        Speed advantage score
    """
    if not attacker_positions or not defender_positions:
        return 0
    
    # Calculate average distance to target zone (goal area)
    target_x, target_y = 95, 50  # Goal area center
    
    # Attackers' average distance
    att_distances = [
        np.sqrt((p['x'] - target_x)**2 + (p['y'] - target_y)**2)
        for p in attacker_positions
    ]
    avg_att_distance = np.mean(att_distances)
    
    # Defenders' average distance
    def_distances = [
        np.sqrt((p['x'] - target_x)**2 + (p['y'] - target_y)**2)
        for p in defender_positions
    ]
    avg_def_distance = np.mean(def_distances)
    
    # Speed advantage: negative distance is closer
    advantage = avg_def_distance - avg_att_distance
    
    return advantage


def calculate_fatigue_impact(minute: int, position: str) -> Dict:
    """
    Calculate fatigue impact on physical attributes
    حساب تأثير الإرهاق على السمات الفيزيائية
    
    Args:
        minute: Game minute
        position: Player position
    
    Returns:
        Dictionary with fatigue adjustments
    """
    # Base fatigue factor (0 to 0.2 reduction)
    if minute <= 45:
        fatigue = minute / 225  # Up to 20% reduction at 45 min
    else:
        fatigue = 0.2 + (minute - 45) / 225  # Additional fatigue in second half
    
    fatigue = min(fatigue, 0.3)  # Cap at 30% reduction
    
    # Position-specific fatigue (high intensity positions tire faster)
    position_multipliers = {
        'GK': 0.3,  # Low fatigue
        'CB': 0.8,
        'RB': 1.2,  # High running
        'LB': 1.2,
        'CDM': 1.0,
        'CM': 1.1,
        'CAM': 1.0,
        'RW': 1.3,  # Highest fatigue
        'LW': 1.3,
        'ST': 1.0,
        'CF': 1.0,
    }
    
    multiplier = position_multipliers.get(position, 1.0)
    adjusted_fatigue = fatigue * multiplier
    
    return {
        'speed_reduction': adjusted_fatigue * 0.15,  # Max 5% speed reduction
        'jumping_reduction': adjusted_fatigue * 0.10,  # Max 3% jumping reduction
        'strength_reduction': adjusted_fatigue * 0.08,  # Strength more stable
    }


if __name__ == "__main__":
    # Test the physical features
    import sys
    sys.path.append('..')
    from data.loaders import load_wyscout_data
    from data.extractors import extract_set_pieces
    
    print("=" * 50)
    print("Testing Physical Features")
    print("=" * 50)
    
    df = load_wyscout_data()
    set_pieces = extract_set_pieces(df)
    
    # Calculate physical features
    set_pieces_with_features = calculate_physical_features(set_pieces)
    
    print(f"\nNew columns added:")
    new_cols = [col for col in set_pieces_with_features.columns 
                if col not in set_pieces.columns]
    print(f"  {new_cols}")
    
    print(f"\nHeader probability stats:")
    print(f"  Mean: {set_pieces_with_features['header_probability'].mean():.3f}")
    print(f"  Std:  {set_pieces_with_features['header_probability'].std():.3f}")
    
    print(f"\nHeight advantage stats:")
    print(f"  Mean: {set_pieces_with_features['height_advantage'].mean():.2f} cm")
    print(f"  Range: {set_pieces_with_features['height_advantage'].min():.2f} to {set_pieces_with_features['height_advantage'].max():.2f}")
