"""
Elite Set-Piece Analytics - Feature Engineering Module
"""

from .spatial import (
    calculate_spatial_features,
    calculate_distance_to_goal,
    calculate_angle_to_goal,
    get_pitch_zone,
    calculate_danger_index,
    calculate_density_features
)

from .temporal import (
    calculate_temporal_features,
    calculate_time_since_last,
    get_game_phase_numeric,
    calculate_fatigue_factor,
    calculate_score_pressure
)

from .physical import (
    calculate_physical_features,
    calculate_header_probability,
    get_player_physical_profile,
    calculate_aerial_duel_probability
)

__all__ = [
    'calculate_spatial_features',
    'calculate_temporal_features',
    'calculate_physical_features',
    'calculate_danger_index',
    'calculate_fatigue_factor'
]
