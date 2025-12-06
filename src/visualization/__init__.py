"""
Elite Set-Piece Analytics - Visualization Module
"""

from .pitch import (
    draw_pitch,
    plot_set_piece,
    plot_heatmap,
    plot_team_comparison,
    plot_set_piece_distribution,
    plot_feature_importance,
    plot_minute_distribution,
    create_summary_dashboard,
    PITCH_LENGTH,
    PITCH_WIDTH
)

from .animations import (
    animate_set_piece,
    create_trajectory_animation,
    animate_prediction
)

__all__ = [
    'draw_pitch',
    'plot_set_piece',
    'plot_heatmap',
    'plot_team_comparison',
    'plot_minute_distribution',
    'create_summary_dashboard',
    'animate_set_piece'
]
