"""
Elite Set-Piece Analytics - Models Module
"""

from .receiver_predictor import (
    FirstReceiverPredictor,
    train_receiver_predictor
)

from .outcome_predictor import (
    OutcomePredictor,
    GoalProbabilityModel,
    train_outcome_predictor
)

__all__ = [
    'FirstReceiverPredictor',
    'OutcomePredictor',
    'train_receiver_predictor',
    'train_outcome_predictor'
]
