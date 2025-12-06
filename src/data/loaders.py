"""
Elite Set-Piece Analytics - Data Loaders
تحميل البيانات للتحليلات الكروية

This module handles loading data from various sources:
- Wyscout World Cup 2022 data (JSON format)
- StatsBomb open data (using statsbombpy)
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm

# Try to import statsbombpy, fall back gracefully if not available
try:
    from statsbombpy import sb
    STATSBOMB_AVAILABLE = True
except ImportError:
    STATSBOMB_AVAILABLE = False


def get_data_path() -> Path:
    """Get the path to the data directory"""
    # Get the project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root / "data"


def load_wyscout_data(
    data_path: Optional[str] = None,
    competition: str = "world_cup_2022"
) -> pd.DataFrame:
    """
    Load Wyscout World Cup 2022 data
    تحميل بيانات وايسكاوت لكأس العالم 2022
    
    Args:
        data_path: Optional path to the data file
        competition: Competition name (default: world_cup_2022)
    
    Returns:
        DataFrame with event data
    """
    if data_path and os.path.exists(data_path):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading Wyscout data from {data_path}: {e}")
    
    # If no file found, generate sample World Cup 2022 data
    # This simulates real Wyscout data structure
    print("📥 Generating sample World Cup 2022 data...")
    return _generate_sample_wyscout_data()


def _generate_sample_wyscout_data() -> pd.DataFrame:
    """
    Generate sample World Cup 2022 data for demonstration
    إنشاء بيانات نموذجية لكأس العالم 2022
    
    This creates realistic set-piece data based on actual World Cup statistics
    """
    np.random.seed(42)  # For reproducibility
    
    # World Cup 2022 teams
    teams = [
        "Argentina", "France", "Croatia", "Morocco", "Netherlands", 
        "England", "Brazil", "Portugal", "Japan", "Spain",
        "Germany", "Switzerland", "Australia", "Poland", "Senegal", "USA"
    ]
    
    # Event types
    set_piece_types = ["corner", "free_kick", "throw_in", "penalty", "goal_kick"]
    
    # Outcomes
    outcomes = ["goal", "shot", "possession_retained", "possession_lost", "clearance"]
    
    # Generate matches
    matches = []
    match_id = 1
    
    for i in range(len(teams)):
        for j in range(i+1, min(i+4, len(teams))):  # Each team plays 3-4 matches
            matches.append({
                'match_id': match_id,
                'home_team': teams[i],
                'away_team': teams[j],
                'stage': np.random.choice(['group_stage', 'round_of_16', 'quarter_final', 'semi_final', 'final'], 
                                         p=[0.6, 0.2, 0.1, 0.05, 0.05])
            })
            match_id += 1
    
    # Generate events for each match
    events = []
    event_id = 1
    
    for match in tqdm(matches, desc="Generating match data"):
        # 15-25 set pieces per match (realistic based on World Cup stats)
        n_set_pieces = np.random.randint(15, 26)
        
        for _ in range(n_set_pieces):
            # Random game minute (more set pieces later in game due to fatigue)
            minute = int(np.random.triangular(1, 45, 95))
            
            # Team and side
            team = np.random.choice([match['home_team'], match['away_team']])
            side = np.random.choice(['left', 'right'])
            
            # Set piece type (corners and free kicks most common)
            sp_type = np.random.choice(
                set_piece_types, 
                p=[0.35, 0.30, 0.25, 0.05, 0.05]
            )
            
            # Position on pitch (normalized 0-100)
            if sp_type == 'corner':
                x = np.random.choice([0, 100])  # Corner spots
                y = np.random.choice([0, 100]) if x == 0 else np.random.choice([0, 100])
            elif sp_type == 'free_kick':
                x = np.random.uniform(20, 95)
                y = np.random.uniform(0, 100)
            else:
                x = np.random.uniform(0, 100)
                y = np.random.uniform(0, 100)
            
            # Generate player positions (attacking team)
            n_attackers = np.random.randint(3, 8)
            attacker_positions = [
                {'x': np.random.uniform(70, 100), 'y': np.random.uniform(20, 80), 
                 'player_id': np.random.randint(1, 12)}
                for _ in range(n_attackers)
            ]
            
            # Generate defender positions
            n_defenders = np.random.randint(4, 10)
            defender_positions = [
                {'x': np.random.uniform(70, 100), 'y': np.random.uniform(15, 85),
                 'player_id': np.random.randint(1, 12)}
                for _ in range(n_defenders)
            ]
            
            # Delivery type for corners and free kicks
            delivery_type = np.random.choice(
                ['inswinger', 'outswinger', 'short', 'driven', 'lofted'],
                p=[0.35, 0.25, 0.15, 0.15, 0.10]
            )
            
            # Outcome (goals are rare ~3%)
            if sp_type in ['corner', 'free_kick']:
                outcome = np.random.choice(
                    outcomes,
                    p=[0.03, 0.15, 0.40, 0.35, 0.07]
                )
            else:
                outcome = np.random.choice(
                    outcomes,
                    p=[0.01, 0.05, 0.55, 0.35, 0.04]
                )
            
            # First receiver - usually one of the attackers
            first_receiver = np.random.randint(0, n_attackers) if outcome != 'clearance' else None
            
            # Create event
            event = {
                'event_id': event_id,
                'match_id': match['match_id'],
                'team': team,
                'opponent': match['away_team'] if team == match['home_team'] else match['home_team'],
                'minute': minute,
                'second': np.random.randint(0, 60),
                'type': sp_type,
                'x': round(x, 2),
                'y': round(y, 2),
                'side': side,
                'delivery_type': delivery_type if sp_type in ['corner', 'free_kick'] else None,
                'outcome': outcome,
                'attacker_positions': attacker_positions,
                'defender_positions': defender_positions,
                'first_receiver_idx': first_receiver,
                'n_attackers_in_box': n_attackers,
                'n_defenders_in_box': n_defenders,
                'stage': match['stage'],
                'score_diff': np.random.randint(-2, 3)  # Score difference at time of event
            }
            
            events.append(event)
            event_id += 1
    
    print(f"✅ Generated {len(events)} set-piece events from {len(matches)} matches")
    return pd.DataFrame(events)


def load_statsbomb_data(
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None
) -> pd.DataFrame:
    """
    Load StatsBomb open data
    تحميل بيانات ستاتسبومب المفتوحة
    
    Args:
        competition_id: StatsBomb competition ID (43 = World Cup)
        season_id: StatsBomb season ID (106 = 2022)
    
    Returns:
        DataFrame with event data
    """
    if not STATSBOMB_AVAILABLE:
        print("⚠️ statsbombpy not available, generating sample data...")
        return _generate_sample_statsbomb_data()
    
    try:
        # Default to World Cup 2022
        if competition_id is None:
            competition_id = 43  # FIFA World Cup
        if season_id is None:
            season_id = 106  # 2022
        
        print(f"📥 Loading StatsBomb data for competition {competition_id}, season {season_id}...")
        
        # Get matches
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        
        if len(matches) == 0:
            print("⚠️ No matches found, generating sample data...")
            return _generate_sample_statsbomb_data()
        
        # Get events for all matches
        all_events = []
        for _, match in tqdm(matches.iterrows(), total=len(matches), desc="Loading matches"):
            try:
                events = sb.events(match_id=match['match_id'])
                events['match_id'] = match['match_id']
                events['home_team'] = match['home_team']
                events['away_team'] = match['away_team']
                all_events.append(events)
            except Exception as e:
                print(f"  Warning: Could not load match {match['match_id']}: {e}")
        
        if all_events:
            df = pd.concat(all_events, ignore_index=True)
            print(f"✅ Loaded {len(df)} events from {len(matches)} matches")
            return df
        else:
            return _generate_sample_statsbomb_data()
            
    except Exception as e:
        print(f"⚠️ Error loading StatsBomb data: {e}")
        return _generate_sample_statsbomb_data()


def _generate_sample_statsbomb_data() -> pd.DataFrame:
    """Generate sample StatsBomb-style data for demonstration"""
    # Use the same generator as Wyscout for consistency
    return _generate_sample_wyscout_data()


def load_all_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load all available data sources and combine them
    تحميل جميع مصادر البيانات المتاحة ودمجها
    
    Args:
        use_cache: Whether to use cached data if available
    
    Returns:
        Combined DataFrame with all event data
    """
    cache_path = get_data_path() / "processed" / "combined_data.parquet"
    
    if use_cache and cache_path.exists():
        print("📥 Loading cached data...")
        return pd.read_parquet(cache_path)
    
    # Load from all sources
    all_data = []
    
    # Try Wyscout data
    print("\n📥 Loading Wyscout data...")
    wyscout_df = load_wyscout_data()
    wyscout_df['source'] = 'wyscout'
    all_data.append(wyscout_df)
    
    # Combine
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Cache if directory exists
    cache_dir = get_data_path() / "processed"
    if cache_dir.exists():
        try:
            combined_df.to_parquet(cache_path)
            print(f"💾 Cached data to {cache_path}")
        except Exception as e:
            print(f"⚠️ Could not cache data: {e}")
    
    return combined_df


def get_available_competitions() -> List[Dict]:
    """
    Get list of available competitions
    الحصول على قائمة المسابقات المتاحة
    """
    return [
        {'id': 1, 'name': 'World Cup 2022', 'source': 'wyscout'},
        {'id': 43, 'name': 'FIFA World Cup', 'source': 'statsbomb'},
    ]


def get_match_list(competition_id: int = 1) -> pd.DataFrame:
    """
    Get list of matches for a competition
    الحصول على قائمة المباريات لمسابقة معينة
    """
    # Load data and extract unique matches
    df = load_wyscout_data()
    
    matches = df.groupby('match_id').agg({
        'team': 'first',
        'opponent': 'first',
        'stage': 'first'
    }).reset_index()
    
    return matches


# Convenience functions
def quick_load() -> pd.DataFrame:
    """Quick load sample data for testing / تحميل سريع للبيانات النموذجية"""
    return load_wyscout_data()


if __name__ == "__main__":
    # Test the loaders
    print("=" * 50)
    print("Testing Data Loaders")
    print("=" * 50)
    
    df = load_wyscout_data()
    print(f"\nLoaded DataFrame shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSet piece types: {df['type'].value_counts().to_dict()}")
    print(f"\nOutcomes: {df['outcome'].value_counts().to_dict()}")
