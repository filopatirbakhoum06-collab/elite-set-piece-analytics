"""
Elite Set-Piece Analytics - Data Extractors
استخراج البيانات لتحليلات الكرات الثابتة

This module extracts set pieces from event data:
- Corners
- Free kicks
- Throw-ins
- Penalties
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


def extract_set_pieces(
    df: pd.DataFrame,
    set_piece_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract all set pieces from event data
    استخراج جميع الكرات الثابتة من بيانات الأحداث
    
    Args:
        df: DataFrame with event data
        set_piece_types: List of set piece types to extract (default: all)
    
    Returns:
        DataFrame with set piece data
    """
    if set_piece_types is None:
        set_piece_types = ['corner', 'free_kick', 'throw_in', 'penalty']
    
    # Filter for set pieces
    set_pieces = df[df['type'].isin(set_piece_types)].copy()
    
    # Add derived features
    set_pieces = _add_derived_features(set_pieces)
    
    print(f"✅ Extracted {len(set_pieces)} set pieces")
    print(f"   Types: {set_pieces['type'].value_counts().to_dict()}")
    
    return set_pieces


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to set piece data"""
    
    # Goal distance (assuming goal at x=100, y=50)
    df['distance_to_goal'] = np.sqrt(
        (100 - df['x'])**2 + (50 - df['y'])**2
    )
    
    # Angle to goal
    df['angle_to_goal'] = np.degrees(
        np.arctan2(50 - df['y'], 100 - df['x'])
    )
    
    # Is dangerous zone (within 35 units of goal)
    df['is_dangerous_zone'] = df['distance_to_goal'] < 35
    
    # Game phase
    df['game_phase'] = pd.cut(
        df['minute'],
        bins=[0, 15, 30, 45, 60, 75, 90, 120],
        labels=['early_1st', 'mid_1st', 'late_1st', 'early_2nd', 'mid_2nd', 'late_2nd', 'extra_time']
    )
    
    # Is winning, drawing, or losing
    df['game_state'] = df['score_diff'].apply(
        lambda x: 'winning' if x > 0 else ('losing' if x < 0 else 'drawing')
    )
    
    return df


def extract_corners(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract corner kicks specifically
    استخراج الركنيات بشكل خاص
    """
    corners = df[df['type'] == 'corner'].copy()
    
    # Corner-specific features
    corners['is_near_post'] = corners['y'].apply(lambda y: y < 35 or y > 65)
    corners['is_far_post'] = ~corners['is_near_post']
    
    # Determine corner zone target
    corners['target_zone'] = corners.apply(_determine_corner_target_zone, axis=1)
    
    return corners


def _determine_corner_target_zone(row) -> str:
    """Determine the target zone for a corner"""
    y = row.get('y', 50)
    
    # Simple zone classification
    if y < 30:
        return 'near_post'
    elif y > 70:
        return 'far_post'
    elif y < 50:
        return 'penalty_spot_near'
    else:
        return 'penalty_spot_far'


def extract_free_kicks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract free kicks specifically
    استخراج الركلات الحرة بشكل خاص
    """
    free_kicks = df[df['type'] == 'free_kick'].copy()
    
    # Free kick specific features
    free_kicks['is_direct_shot_range'] = free_kicks['distance_to_goal'] < 30
    free_kicks['is_crossing_range'] = (
        (free_kicks['distance_to_goal'] >= 30) & 
        (free_kicks['distance_to_goal'] < 50)
    )
    
    return free_kicks


def extract_throw_ins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract throw-ins specifically
    استخراج رميات التماس بشكل خاص
    """
    throw_ins = df[df['type'] == 'throw_in'].copy()
    
    # Throw-in specific features
    throw_ins['is_attacking_third'] = throw_ins['x'] > 66
    throw_ins['is_long_throw_position'] = (
        (throw_ins['x'] > 80) & 
        (throw_ins['y'] < 30) | (throw_ins['y'] > 70)
    )
    
    return throw_ins


def link_to_next_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Link each set piece to its next event (to find first receiver)
    ربط كل كرة ثابتة بالحدث التالي لها
    """
    # Sort by match and time
    df_sorted = df.sort_values(['match_id', 'minute', 'second']).reset_index(drop=True)
    
    # Get next event for each row
    df_sorted['next_event_type'] = df_sorted['type'].shift(-1)
    df_sorted['next_event_team'] = df_sorted['team'].shift(-1)
    df_sorted['next_event_outcome'] = df_sorted['outcome'].shift(-1)
    
    # Check if same match
    df_sorted['same_match'] = df_sorted['match_id'] == df_sorted['match_id'].shift(-1)
    
    # Only keep next event info if same match
    for col in ['next_event_type', 'next_event_team', 'next_event_outcome']:
        df_sorted.loc[~df_sorted['same_match'], col] = None
    
    df_sorted = df_sorted.drop('same_match', axis=1)
    
    return df_sorted


def calculate_success_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate success metrics for set pieces
    حساب مقاييس النجاح للكرات الثابتة
    """
    metrics = {}
    
    # Overall success rate (goal or shot)
    successful_outcomes = ['goal', 'shot']
    metrics['success_rate'] = (
        df['outcome'].isin(successful_outcomes).sum() / len(df) * 100
    )
    
    # Goal rate
    metrics['goal_rate'] = (
        (df['outcome'] == 'goal').sum() / len(df) * 100
    )
    
    # Shot rate
    metrics['shot_rate'] = (
        (df['outcome'] == 'shot').sum() / len(df) * 100
    )
    
    # Possession retained rate
    metrics['possession_retained_rate'] = (
        (df['outcome'] == 'possession_retained').sum() / len(df) * 100
    )
    
    # By type
    metrics['by_type'] = {}
    for sp_type in df['type'].unique():
        type_df = df[df['type'] == sp_type]
        metrics['by_type'][sp_type] = {
            'count': len(type_df),
            'goal_rate': (type_df['outcome'] == 'goal').sum() / len(type_df) * 100 if len(type_df) > 0 else 0,
            'success_rate': type_df['outcome'].isin(successful_outcomes).sum() / len(type_df) * 100 if len(type_df) > 0 else 0
        }
    
    return metrics


def get_first_receiver_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for first receiver prediction
    تحضير البيانات لتنبؤ المستلم الأول
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Only use set pieces where we have a first receiver
    valid_df = df[df['first_receiver_idx'].notna()].copy()
    
    # Extract features
    feature_columns = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal',
        'n_attackers_in_box', 'n_defenders_in_box',
        'minute', 'score_diff'
    ]
    
    # Add one-hot encoding for categorical features
    type_dummies = pd.get_dummies(valid_df['type'], prefix='type')
    delivery_dummies = pd.get_dummies(valid_df['delivery_type'].fillna('unknown'), prefix='delivery')
    side_dummies = pd.get_dummies(valid_df['side'], prefix='side')
    
    # Combine features
    X = pd.concat([
        valid_df[feature_columns],
        type_dummies,
        delivery_dummies,
        side_dummies
    ], axis=1)
    
    # Target
    y = valid_df['first_receiver_idx'].astype(int)
    
    return X, y


def summarize_set_pieces(df: pd.DataFrame) -> Dict:
    """
    Create a summary of set piece data
    إنشاء ملخص لبيانات الكرات الثابتة
    """
    summary = {
        'total_count': len(df),
        'unique_matches': df['match_id'].nunique(),
        'avg_per_match': len(df) / df['match_id'].nunique() if df['match_id'].nunique() > 0 else 0,
        'types': df['type'].value_counts().to_dict(),
        'outcomes': df['outcome'].value_counts().to_dict(),
        'teams': df['team'].value_counts().head(10).to_dict(),
        'metrics': calculate_success_metrics(df)
    }
    
    return summary


if __name__ == "__main__":
    # Test the extractors
    from loaders import load_wyscout_data
    
    print("=" * 50)
    print("Testing Data Extractors")
    print("=" * 50)
    
    df = load_wyscout_data()
    
    # Extract set pieces
    set_pieces = extract_set_pieces(df)
    print(f"\nExtracted {len(set_pieces)} set pieces")
    
    # Calculate metrics
    metrics = calculate_success_metrics(set_pieces)
    print(f"\nSuccess Metrics:")
    print(f"  Goal Rate: {metrics['goal_rate']:.2f}%")
    print(f"  Shot Rate: {metrics['shot_rate']:.2f}%")
    print(f"  Success Rate: {metrics['success_rate']:.2f}%")
    
    # Get first receiver data
    X, y = get_first_receiver_data(set_pieces)
    print(f"\nFirst Receiver Data:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
