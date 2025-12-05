"""
Configuration module for Elite Set-Piece Analytics.
ملف الإعدادات الرئيسي

This module contains all configuration settings used throughout the project,
including paths, pitch dimensions, model parameters, and visualization settings.
"""

from pathlib import Path
from typing import Dict, Any, List
import os


class Config:
    """
    Central configuration class for the Elite Set-Piece Analytics platform.
    
    This class provides access to all configuration settings including:
    - Project paths
    - Pitch dimensions
    - Model parameters
    - Feature engineering settings
    - Visualization settings
    - Random seeds for reproducibility
    
    Example:
        >>> config = Config()
        >>> print(config.pitch_length)
        105.0
        >>> print(config.RANDOM_SEED)
        42
    """
    
    # ==========================================================================
    # Project Paths - مسارات المشروع
    # ==========================================================================
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
    DOCS_DIR = PROJECT_ROOT / "docs"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # ==========================================================================
    # Pitch Dimensions - أبعاد الملعب (FIFA standard)
    # ==========================================================================
    
    PITCH_LENGTH = 105.0  # meters
    PITCH_WIDTH = 68.0    # meters
    
    # Goal dimensions
    GOAL_WIDTH = 7.32     # meters
    GOAL_HEIGHT = 2.44    # meters
    
    # Penalty area
    PENALTY_AREA_LENGTH = 16.5   # meters
    PENALTY_AREA_WIDTH = 40.3    # meters
    PENALTY_SPOT_DISTANCE = 11.0 # meters from goal line
    
    # Goal area (6-yard box)
    GOAL_AREA_LENGTH = 5.5   # meters
    GOAL_AREA_WIDTH = 18.3   # meters
    
    # Center circle
    CENTER_CIRCLE_RADIUS = 9.15  # meters
    
    # Corner arc
    CORNER_ARC_RADIUS = 1.0  # meters
    
    # ==========================================================================
    # Set-Piece Zone Definitions - تعريفات مناطق الكرات الثابتة
    # ==========================================================================
    
    SET_PIECE_ZONES: Dict[str, Dict[str, float]] = {
        "corner_left": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
        "corner_right": {"x_min": 0, "x_max": 1, "y_min": 67, "y_max": 68},
        "penalty_area": {"x_min": 0, "x_max": 16.5, "y_min": 13.85, "y_max": 54.15},
        "dangerous_free_kick": {"x_min": 16.5, "x_max": 30, "y_min": 10, "y_max": 58},
    }
    
    # ==========================================================================
    # Model Parameters - معاملات النموذج
    # ==========================================================================
    
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.15
    N_FOLDS = 5
    
    # XGBoost default parameters
    XGBOOST_PARAMS: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # LightGBM default parameters
    LIGHTGBM_PARAMS: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # Neural Network parameters
    NEURAL_NETWORK_PARAMS: Dict[str, Any] = {
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping_patience": 10,
    }
    
    # ==========================================================================
    # Feature Engineering Settings - إعدادات هندسة الميزات
    # ==========================================================================
    
    SPATIAL_FEATURES: List[str] = [
        "distance_to_goal",
        "angle_to_goal",
        "distance_to_ball",
        "player_density_5m",
        "player_density_10m",
        "nearest_opponent_distance",
        "nearest_teammate_distance",
        "x_position",
        "y_position",
        "zone_classification",
    ]
    
    TEMPORAL_FEATURES: List[str] = [
        "time_since_start",
        "time_until_end",
        "sequence_position",
        "event_duration",
        "time_pressure",
    ]
    
    PHYSICAL_FEATURES: List[str] = [
        "speed",
        "acceleration",
        "direction",
        "direction_change",
        "distance_covered",
    ]
    
    # ==========================================================================
    # Visualization Settings - إعدادات التصوير
    # ==========================================================================
    
    VIZ_SETTINGS: Dict[str, Any] = {
        "pitch_color": "#22312b",
        "line_color": "white",
        "home_team_color": "#3498db",
        "away_team_color": "#e74c3c",
        "ball_color": "yellow",
        "figsize": (12, 8),
        "dpi": 100,
        "font_family": "DejaVu Sans",
    }
    
    HEATMAP_SETTINGS: Dict[str, Any] = {
        "cmap": "hot",
        "alpha": 0.6,
        "bins": 25,
    }
    
    ANIMATION_SETTINGS: Dict[str, Any] = {
        "fps": 10,
        "interval": 100,
        "player_size": 100,
        "ball_size": 50,
    }
    
    # ==========================================================================
    # Data Source URLs - مصادر البيانات
    # ==========================================================================
    
    DATA_SOURCES: Dict[str, str] = {
        "wyscout_github": "https://github.com/statsbomb/wyscout-soccer-match-event-dataset",
        "statsbomb_open": "https://github.com/statsbomb/open-data",
        "metrica_sample": "https://github.com/metrica-sports/sample-data",
    }
    
    # ==========================================================================
    # Logging Configuration - إعدادات التسجيل
    # ==========================================================================
    
    LOGGING_CONFIG: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": True
            }
        }
    }
    
    # ==========================================================================
    # Class Methods - وظائف الفئة
    # ==========================================================================
    
    @classmethod
    def get_pitch_dimensions(cls) -> Dict[str, float]:
        """Get all pitch dimensions as a dictionary."""
        return {
            "length": cls.PITCH_LENGTH,
            "width": cls.PITCH_WIDTH,
            "goal_width": cls.GOAL_WIDTH,
            "goal_height": cls.GOAL_HEIGHT,
            "penalty_area_length": cls.PENALTY_AREA_LENGTH,
            "penalty_area_width": cls.PENALTY_AREA_WIDTH,
            "penalty_spot_distance": cls.PENALTY_SPOT_DISTANCE,
            "goal_area_length": cls.GOAL_AREA_LENGTH,
            "goal_area_width": cls.GOAL_AREA_WIDTH,
            "center_circle_radius": cls.CENTER_CIRCLE_RADIUS,
            "corner_arc_radius": cls.CORNER_ARC_RADIUS,
        }
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_all_features(cls) -> List[str]:
        """Get combined list of all feature names."""
        return cls.SPATIAL_FEATURES + cls.TEMPORAL_FEATURES + cls.PHYSICAL_FEATURES
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export all configuration as a dictionary."""
        return {
            "pitch": cls.get_pitch_dimensions(),
            "zones": cls.SET_PIECE_ZONES,
            "model_params": {
                "xgboost": cls.XGBOOST_PARAMS,
                "lightgbm": cls.LIGHTGBM_PARAMS,
                "neural_network": cls.NEURAL_NETWORK_PARAMS,
            },
            "features": {
                "spatial": cls.SPATIAL_FEATURES,
                "temporal": cls.TEMPORAL_FEATURES,
                "physical": cls.PHYSICAL_FEATURES,
            },
            "visualization": cls.VIZ_SETTINGS,
            "random_seed": cls.RANDOM_SEED,
        }


# Create a default instance
config = Config()
