"""
Elite Set-Piece Analytics - Data Module
"""

from .loaders import (
    load_wyscout_data,
    load_statsbomb_data,
    load_all_data,
    quick_load,
    get_available_competitions,
    get_match_list
)

from .extractors import (
    extract_set_pieces,
    extract_corners,
    extract_free_kicks,
    extract_throw_ins,
    calculate_success_metrics,
    get_first_receiver_data,
    summarize_set_pieces
)

from .preprocessors import (
    clean_data,
    normalize_coordinates,
    handle_outliers,
    split_data,
    prepare_features,
    encode_categorical,
    create_training_pipeline
)

__all__ = [
    'load_wyscout_data',
    'load_statsbomb_data',
    'load_all_data',
    'quick_load',
    'extract_set_pieces',
    'extract_corners',
    'extract_free_kicks',
    'calculate_success_metrics',
    'clean_data',
    'split_data'
]
