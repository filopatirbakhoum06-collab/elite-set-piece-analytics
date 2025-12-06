"""
Elite Set-Piece Analytics

Advanced analytics platform for football set pieces.
تحليلات متقدمة للكرات الثابتة في كرة القدم

Author: Elite Analytics Team
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Elite Analytics Team'

from . import data
from . import features
from . import models
from . import visualization

__all__ = ['data', 'features', 'models', 'visualization']
