"""
Animation utilities for Elite Set-Piece Analytics.
أدوات الرسوم المتحركة

This module provides functionality to create animations of
set-piece events using tracking data.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

from ..utils.config import Config
from .pitch import PitchVisualizer


class AnimationCreator:
    """
    Creator for animated visualizations of set-pieces.
    منشئ الرسوم المتحركة للكرات الثابتة
    
    This class generates animated visualizations showing
    player and ball movement over time.
    
    Example:
        >>> creator = AnimationCreator()
        >>> anim = creator.create_animation(tracking_data)
        >>> anim.save('set_piece.gif', writer='pillow')
    """
    
    def __init__(
        self,
        pitch_viz: Optional[PitchVisualizer] = None,
        fps: int = 10,
        interval: int = 100
    ):
        """
        Initialize the animation creator.
        
        Args:
            pitch_viz: PitchVisualizer instance
            fps: Frames per second for output
            interval: Interval between frames in milliseconds
        """
        self.pitch_viz = pitch_viz or PitchVisualizer()
        self.fps = fps
        self.interval = interval
        self.config = Config()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_animation(
        self,
        tracking_data: pd.DataFrame,
        home_players: List[str],
        away_players: List[str],
        ball_columns: Tuple[str, str] = ('ball_x', 'ball_y'),
        title: str = "Set-Piece Animation"
    ) -> FuncAnimation:
        """
        Create an animation from tracking data.
        إنشاء رسوم متحركة من بيانات التتبع
        
        Args:
            tracking_data: DataFrame with tracking data (frame-by-frame)
            home_players: List of home player column prefixes
            away_players: List of away player column prefixes
            ball_columns: Tuple of (x_column, y_column) for ball
            title: Animation title
            
        Returns:
            FuncAnimation object
        """
        fig, ax = self.pitch_viz.draw_pitch()
        
        # Initialize scatter plots
        home_scatter = ax.scatter([], [], s=200, c='#3498db', edgecolors='white', 
                                 linewidths=2, zorder=5)
        away_scatter = ax.scatter([], [], s=200, c='#e74c3c', edgecolors='white', 
                                 linewidths=2, zorder=5)
        ball_scatter = ax.scatter([], [], s=100, c='yellow', edgecolors='black', 
                                 linewidths=2, zorder=10)
        
        self.pitch_viz.add_title(ax, title)
        
        def init():
            """Initialize animation."""
            home_scatter.set_offsets(np.empty((0, 2)))
            away_scatter.set_offsets(np.empty((0, 2)))
            ball_scatter.set_offsets(np.empty((0, 2)))
            return home_scatter, away_scatter, ball_scatter
        
        def update(frame):
            """Update frame."""
            row = tracking_data.iloc[frame]
            
            # Update home players
            home_positions = []
            for player in home_players:
                x_col = f"{player}_x"
                y_col = f"{player}_y"
                if x_col in row.index and y_col in row.index:
                    x, y = row[x_col], row[y_col]
                    if pd.notna(x) and pd.notna(y):
                        home_positions.append([x, y])
            
            if home_positions:
                home_scatter.set_offsets(np.array(home_positions))
            
            # Update away players
            away_positions = []
            for player in away_players:
                x_col = f"{player}_x"
                y_col = f"{player}_y"
                if x_col in row.index and y_col in row.index:
                    x, y = row[x_col], row[y_col]
                    if pd.notna(x) and pd.notna(y):
                        away_positions.append([x, y])
            
            if away_positions:
                away_scatter.set_offsets(np.array(away_positions))
            
            # Update ball
            ball_x_col, ball_y_col = ball_columns
            if ball_x_col in row.index and ball_y_col in row.index:
                ball_x, ball_y = row[ball_x_col], row[ball_y_col]
                if pd.notna(ball_x) and pd.notna(ball_y):
                    ball_scatter.set_offsets([[ball_x, ball_y]])
            
            return home_scatter, away_scatter, ball_scatter
        
        anim = FuncAnimation(
            fig, update, init_func=init,
            frames=len(tracking_data),
            interval=self.interval,
            blit=True
        )
        
        return anim
    
    def save_animation(
        self,
        animation: FuncAnimation,
        filename: str,
        writer: str = 'pillow',
        dpi: int = 100
    ) -> None:
        """
        Save animation to file.
        
        Args:
            animation: FuncAnimation object
            filename: Output filename
            writer: Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)
            dpi: Output DPI
        """
        self.logger.info(f"Saving animation to {filename}")
        animation.save(filename, writer=writer, dpi=dpi, fps=self.fps)
        self.logger.info("Animation saved successfully")
