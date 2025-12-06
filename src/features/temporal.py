"""
Elite Set-Piece Analytics - Temporal Features
الميزات الزمنية لتحليلات الكرات الثابتة

This module calculates temporal features:
- Time since last set piece
- Game minute context
- Score difference impact
- Fatigue indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all temporal features for set pieces
    حساب جميع الميزات الزمنية للكرات الثابتة
    
    Args:
        df: DataFrame with set piece data
    
    Returns:
        DataFrame with added temporal features
    """
    df_feat = df.copy()
    
    # Sort by match and time
    df_feat = df_feat.sort_values(['match_id', 'minute', 'second']).reset_index(drop=True)
    
    # Time since last set piece (same match)
    df_feat['time_since_last_sp'] = calculate_time_since_last(df_feat)
    
    # Game phase features
    df_feat['game_phase_numeric'] = df_feat['minute'].apply(get_game_phase_numeric)
    df_feat['is_first_half'] = df_feat['minute'] <= 45
    df_feat['is_extra_time'] = df_feat['minute'] > 90
    
    # Fatigue indicator (minutes played)
    df_feat['fatigue_factor'] = calculate_fatigue_factor(df_feat['minute'].values)
    
    # Score pressure
    df_feat['score_pressure'] = df_feat.apply(calculate_score_pressure, axis=1)
    
    # Set piece sequence features
    df_feat = add_sequence_features(df_feat)
    
    # Critical moment indicators
    df_feat['is_critical_moment'] = (
        (df_feat['minute'] >= 85) | 
        (df_feat['minute'].isin([44, 45, 89, 90])) |
        (df_feat['score_diff'].abs() <= 1)
    )
    
    return df_feat


def calculate_time_since_last(df: pd.DataFrame) -> pd.Series:
    """
    Calculate time since last set piece for the same team in same match
    حساب الوقت منذ آخر كرة ثابتة لنفس الفريق في نفس المباراة
    
    Args:
        df: DataFrame sorted by time
    
    Returns:
        Series with time differences in minutes
    """
    time_since = []
    
    last_sp_time = {}  # {(match_id, team): minute}
    
    for idx, row in df.iterrows():
        key = (row['match_id'], row['team'])
        current_minute = row['minute']
        
        if key in last_sp_time:
            diff = current_minute - last_sp_time[key]
        else:
            diff = current_minute  # First set piece
        
        time_since.append(diff)
        last_sp_time[key] = current_minute
    
    return pd.Series(time_since, index=df.index)


def get_game_phase_numeric(minute: int) -> int:
    """
    Convert game minute to numeric phase
    تحويل دقيقة المباراة إلى مرحلة رقمية
    
    Args:
        minute: Game minute
    
    Returns:
        Phase number (1-6)
    """
    if minute <= 15:
        return 1  # Early first half
    elif minute <= 30:
        return 2  # Mid first half
    elif minute <= 45:
        return 3  # Late first half
    elif minute <= 60:
        return 4  # Early second half
    elif minute <= 75:
        return 5  # Mid second half
    else:
        return 6  # Late game (most critical)


def calculate_fatigue_factor(minutes: np.ndarray) -> np.ndarray:
    """
    Calculate fatigue factor based on game minute
    حساب عامل الإرهاق بناءً على دقيقة المباراة
    
    Research shows fatigue significantly impacts:
    - Sprint speed (decreases ~10% in last 15 min)
    - Reaction time
    - Marking quality
    
    Args:
        minutes: Array of game minutes
    
    Returns:
        Fatigue factor array (0-1, higher = more fatigued)
    """
    # Fatigue accumulates non-linearly
    # First 45 min: gradual increase
    # Half-time: reset partially
    # Last 30 min: accelerated fatigue
    
    fatigue = np.zeros_like(minutes, dtype=float)
    
    for i, minute in enumerate(minutes):
        if minute <= 45:
            # First half: gradual accumulation
            fatigue[i] = minute / 90  # 0 to 0.5
        elif minute <= 60:
            # After half-time: partial recovery
            fatigue[i] = 0.4 + (minute - 45) / 90
        else:
            # Last 30 minutes: accelerated fatigue
            fatigue[i] = 0.55 + (minute - 60) / 60  # Faster accumulation
    
    # Cap at 1.0
    fatigue = np.clip(fatigue, 0, 1)
    
    return fatigue


def calculate_score_pressure(row) -> float:
    """
    Calculate pressure factor based on score and time
    حساب عامل الضغط بناءً على النتيجة والوقت
    
    Args:
        row: DataFrame row
    
    Returns:
        Pressure score (0-100)
    """
    minute = row.get('minute', 45)
    score_diff = row.get('score_diff', 0)
    
    # Time pressure (increases as game progresses)
    time_pressure = min(minute / 90, 1) * 50
    
    # Score pressure (higher when losing or drawing)
    if score_diff < 0:  # Losing
        score_pressure = 50 * (1 + abs(score_diff) / 3)  # Max ~67 when losing by 2+
    elif score_diff == 0:  # Drawing
        score_pressure = 40
    else:  # Winning
        score_pressure = 20 / (1 + score_diff)  # Decreases when winning by more
    
    # Combine
    total_pressure = min(time_pressure + score_pressure / 2, 100)
    
    return total_pressure


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features about set piece sequences
    إضافة ميزات حول تسلسل الكرات الثابتة
    
    Args:
        df: DataFrame sorted by time
    
    Returns:
        DataFrame with sequence features
    """
    df_seq = df.copy()
    
    # Number of set pieces so far in match (for each team)
    df_seq['sp_count_in_match'] = df_seq.groupby(['match_id', 'team']).cumcount() + 1
    
    # Total set pieces in match up to this point
    df_seq['total_sp_count'] = df_seq.groupby('match_id').cumcount() + 1
    
    # Set piece type sequence (was last SP same type?)
    df_seq['same_type_as_last'] = False
    
    last_type = {}  # {(match_id, team): type}
    same_type_list = []
    
    for idx, row in df_seq.iterrows():
        key = (row['match_id'], row['team'])
        current_type = row['type']
        
        if key in last_type:
            same_type_list.append(last_type[key] == current_type)
        else:
            same_type_list.append(False)
        
        last_type[key] = current_type
    
    df_seq['same_type_as_last'] = same_type_list
    
    # Rolling success rate (last 3 set pieces)
    df_seq['recent_success'] = calculate_rolling_success(df_seq)
    
    return df_seq


def calculate_rolling_success(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """
    Calculate rolling success rate for recent set pieces
    حساب معدل النجاح المتحرك للكرات الثابتة الأخيرة
    
    Args:
        df: DataFrame sorted by time
        window: Number of previous set pieces to consider
    
    Returns:
        Series with rolling success rate
    """
    # Define success as goal or shot
    df_temp = df.copy()
    df_temp['is_success'] = df_temp['outcome'].isin(['goal', 'shot'])
    
    # Group by match and team
    success_rates = []
    
    history = {}  # {(match_id, team): [success_list]}
    
    for idx, row in df_temp.iterrows():
        key = (row['match_id'], row['team'])
        
        if key not in history:
            history[key] = []
        
        # Calculate rate from history
        recent_history = history[key][-window:]
        if recent_history:
            rate = sum(recent_history) / len(recent_history)
        else:
            rate = 0.5  # Default
        
        success_rates.append(rate)
        
        # Add current result to history
        history[key].append(row['is_success'])
    
    return pd.Series(success_rates, index=df.index)


def get_match_context(df: pd.DataFrame, match_id: int) -> Dict:
    """
    Get temporal context for a specific match
    الحصول على السياق الزمني لمباراة معينة
    
    Args:
        df: Full DataFrame
        match_id: Match ID
    
    Returns:
        Dictionary with match context
    """
    match_df = df[df['match_id'] == match_id]
    
    if len(match_df) == 0:
        return {}
    
    context = {
        'total_set_pieces': len(match_df),
        'first_half_sp': len(match_df[match_df['minute'] <= 45]),
        'second_half_sp': len(match_df[match_df['minute'] > 45]),
        'extra_time_sp': len(match_df[match_df['minute'] > 90]),
        'avg_time_between': match_df['time_since_last_sp'].mean() if 'time_since_last_sp' in match_df.columns else None,
        'most_common_minute': match_df['minute'].mode().iloc[0] if len(match_df) > 0 else None,
        'critical_moment_sp': len(match_df[match_df['minute'] >= 85])
    }
    
    return context


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate momentum-based features
    حساب ميزات الزخم
    
    Args:
        df: DataFrame with set piece data
    
    Returns:
        DataFrame with momentum features
    """
    df_mom = df.copy()
    
    # Set pieces in last 10 minutes
    df_mom['sp_last_10_min'] = df_mom.apply(
        lambda row: count_recent_sp(df_mom, row, 10),
        axis=1
    )
    
    # Goals in last 10 minutes
    df_mom['goals_last_10_min'] = df_mom.apply(
        lambda row: count_recent_goals(df_mom, row, 10),
        axis=1
    )
    
    # Momentum score
    df_mom['momentum_score'] = df_mom['sp_last_10_min'] * 10 + df_mom['goals_last_10_min'] * 30
    
    return df_mom


def count_recent_sp(df: pd.DataFrame, row, window_minutes: int) -> int:
    """Count set pieces in the last N minutes for same team"""
    mask = (
        (df['match_id'] == row['match_id']) &
        (df['team'] == row['team']) &
        (df['minute'] >= row['minute'] - window_minutes) &
        (df['minute'] < row['minute'])
    )
    return mask.sum()


def count_recent_goals(df: pd.DataFrame, row, window_minutes: int) -> int:
    """Count goals in the last N minutes for same team"""
    mask = (
        (df['match_id'] == row['match_id']) &
        (df['team'] == row['team']) &
        (df['minute'] >= row['minute'] - window_minutes) &
        (df['minute'] < row['minute']) &
        (df['outcome'] == 'goal')
    )
    return mask.sum()


if __name__ == "__main__":
    # Test the temporal features
    import sys
    sys.path.append('..')
    from data.loaders import load_wyscout_data
    from data.extractors import extract_set_pieces
    
    print("=" * 50)
    print("Testing Temporal Features")
    print("=" * 50)
    
    df = load_wyscout_data()
    set_pieces = extract_set_pieces(df)
    
    # Calculate temporal features
    set_pieces_with_features = calculate_temporal_features(set_pieces)
    
    print(f"\nNew columns added:")
    new_cols = [col for col in set_pieces_with_features.columns 
                if col not in set_pieces.columns]
    print(f"  {new_cols}")
    
    print(f"\nFatigue factor stats:")
    print(f"  Mean: {set_pieces_with_features['fatigue_factor'].mean():.3f}")
    print(f"  Std:  {set_pieces_with_features['fatigue_factor'].std():.3f}")
    
    print(f"\nScore pressure stats:")
    print(f"  Mean: {set_pieces_with_features['score_pressure'].mean():.2f}")
    print(f"  Max:  {set_pieces_with_features['score_pressure'].max():.2f}")
