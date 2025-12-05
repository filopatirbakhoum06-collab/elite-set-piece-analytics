"""
Pitch Visualization utilities for Elite Set-Piece Analytics.
أدوات تصوير الملعب

This module provides classes and functions for visualizing football
pitches, player positions, and tactical annotations.

Example usage:
    >>> from src.visualization.pitch import PitchVisualizer
    >>> viz = PitchVisualizer()
    >>> fig, ax = viz.draw_pitch()
    >>> viz.plot_players(ax, home_positions, away_positions)
    >>> plt.show()
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Rectangle, FancyArrow
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from ..utils.config import Config


class PitchVisualizer:
    """
    Football pitch visualization class.
    فئة تصوير ملعب كرة القدم
    
    This class provides methods to draw a standard football pitch
    and add various annotations like player positions, passes, and shots.
    
    Attributes:
        pitch_length: Length of pitch in meters
        pitch_width: Width of pitch in meters
        
    Example:
        >>> viz = PitchVisualizer()
        >>> fig, ax = viz.draw_pitch()
        >>> viz.plot_players(ax, home_df, away_df)
    """
    
    def __init__(
        self,
        pitch_length: float = Config.PITCH_LENGTH,
        pitch_width: float = Config.PITCH_WIDTH,
        figsize: Tuple[int, int] = (12, 8),
        pitch_color: str = "#22312b",
        line_color: str = "white",
        line_width: float = 2.0
    ):
        """
        Initialize the pitch visualizer.
        
        Args:
            pitch_length: Length of pitch in meters
            pitch_width: Width of pitch in meters
            figsize: Figure size (width, height)
            pitch_color: Background color of pitch
            line_color: Color of pitch markings
            line_width: Width of pitch lines
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.figsize = figsize
        self.pitch_color = pitch_color
        self.line_color = line_color
        self.line_width = line_width
        
        # Import configuration
        self.config = Config()
        
        # Pitch dimensions from config
        self.penalty_area_length = Config.PENALTY_AREA_LENGTH
        self.penalty_area_width = Config.PENALTY_AREA_WIDTH
        self.goal_area_length = Config.GOAL_AREA_LENGTH
        self.goal_area_width = Config.GOAL_AREA_WIDTH
        self.center_circle_radius = Config.CENTER_CIRCLE_RADIUS
        self.penalty_spot_distance = Config.PENALTY_SPOT_DISTANCE
        self.goal_width = Config.GOAL_WIDTH
        self.corner_arc_radius = Config.CORNER_ARC_RADIUS
    
    def draw_pitch(
        self,
        ax: Optional[plt.Axes] = None,
        orientation: str = 'horizontal'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Draw a football pitch.
        رسم ملعب كرة القدم
        
        Args:
            ax: Matplotlib axes (creates new if None)
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            Tuple of (figure, axes)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Set pitch background
        ax.set_facecolor(self.pitch_color)
        
        # Calculate coordinates
        length = self.pitch_length
        width = self.pitch_width
        
        # Outer boundary
        ax.plot([0, 0], [0, width], color=self.line_color, linewidth=self.line_width)
        ax.plot([0, length], [width, width], color=self.line_color, linewidth=self.line_width)
        ax.plot([length, length], [width, 0], color=self.line_color, linewidth=self.line_width)
        ax.plot([length, 0], [0, 0], color=self.line_color, linewidth=self.line_width)
        
        # Halfway line
        ax.plot([length/2, length/2], [0, width], color=self.line_color, linewidth=self.line_width)
        
        # Center circle
        center_circle = Circle(
            (length/2, width/2), 
            self.center_circle_radius,
            fill=False, 
            color=self.line_color, 
            linewidth=self.line_width
        )
        ax.add_patch(center_circle)
        
        # Center spot
        center_spot = Circle(
            (length/2, width/2), 
            0.4,
            fill=True, 
            color=self.line_color
        )
        ax.add_patch(center_spot)
        
        # Left penalty area
        left_penalty_y = (width - self.penalty_area_width) / 2
        ax.plot([0, self.penalty_area_length], [left_penalty_y, left_penalty_y], 
               color=self.line_color, linewidth=self.line_width)
        ax.plot([self.penalty_area_length, self.penalty_area_length], 
               [left_penalty_y, left_penalty_y + self.penalty_area_width],
               color=self.line_color, linewidth=self.line_width)
        ax.plot([self.penalty_area_length, 0], 
               [left_penalty_y + self.penalty_area_width, left_penalty_y + self.penalty_area_width],
               color=self.line_color, linewidth=self.line_width)
        
        # Right penalty area
        ax.plot([length, length - self.penalty_area_length], [left_penalty_y, left_penalty_y], 
               color=self.line_color, linewidth=self.line_width)
        ax.plot([length - self.penalty_area_length, length - self.penalty_area_length], 
               [left_penalty_y, left_penalty_y + self.penalty_area_width],
               color=self.line_color, linewidth=self.line_width)
        ax.plot([length - self.penalty_area_length, length], 
               [left_penalty_y + self.penalty_area_width, left_penalty_y + self.penalty_area_width],
               color=self.line_color, linewidth=self.line_width)
        
        # Left goal area
        left_goal_y = (width - self.goal_area_width) / 2
        ax.plot([0, self.goal_area_length], [left_goal_y, left_goal_y], 
               color=self.line_color, linewidth=self.line_width)
        ax.plot([self.goal_area_length, self.goal_area_length], 
               [left_goal_y, left_goal_y + self.goal_area_width],
               color=self.line_color, linewidth=self.line_width)
        ax.plot([self.goal_area_length, 0], 
               [left_goal_y + self.goal_area_width, left_goal_y + self.goal_area_width],
               color=self.line_color, linewidth=self.line_width)
        
        # Right goal area
        ax.plot([length, length - self.goal_area_length], [left_goal_y, left_goal_y], 
               color=self.line_color, linewidth=self.line_width)
        ax.plot([length - self.goal_area_length, length - self.goal_area_length], 
               [left_goal_y, left_goal_y + self.goal_area_width],
               color=self.line_color, linewidth=self.line_width)
        ax.plot([length - self.goal_area_length, length], 
               [left_goal_y + self.goal_area_width, left_goal_y + self.goal_area_width],
               color=self.line_color, linewidth=self.line_width)
        
        # Penalty spots
        left_penalty_spot = Circle(
            (self.penalty_spot_distance, width/2), 
            0.4,
            fill=True, 
            color=self.line_color
        )
        ax.add_patch(left_penalty_spot)
        
        right_penalty_spot = Circle(
            (length - self.penalty_spot_distance, width/2), 
            0.4,
            fill=True, 
            color=self.line_color
        )
        ax.add_patch(right_penalty_spot)
        
        # Penalty arcs
        left_arc = Arc(
            (self.penalty_spot_distance, width/2),
            height=2*self.center_circle_radius,
            width=2*self.center_circle_radius,
            angle=0,
            theta1=307,
            theta2=53,
            color=self.line_color,
            linewidth=self.line_width
        )
        ax.add_patch(left_arc)
        
        right_arc = Arc(
            (length - self.penalty_spot_distance, width/2),
            height=2*self.center_circle_radius,
            width=2*self.center_circle_radius,
            angle=0,
            theta1=127,
            theta2=233,
            color=self.line_color,
            linewidth=self.line_width
        )
        ax.add_patch(right_arc)
        
        # Corner arcs
        corners = [
            (0, 0, 0, 90),
            (0, width, 270, 360),
            (length, 0, 90, 180),
            (length, width, 180, 270),
        ]
        
        for x, y, theta1, theta2 in corners:
            arc = Arc(
                (x, y),
                height=2*self.corner_arc_radius,
                width=2*self.corner_arc_radius,
                angle=0,
                theta1=theta1,
                theta2=theta2,
                color=self.line_color,
                linewidth=self.line_width
            )
            ax.add_patch(arc)
        
        # Goals
        goal_y = (width - self.goal_width) / 2
        ax.plot([-2, -2], [goal_y, goal_y + self.goal_width], 
               color=self.line_color, linewidth=self.line_width*1.5)
        ax.plot([length+2, length+2], [goal_y, goal_y + self.goal_width], 
               color=self.line_color, linewidth=self.line_width*1.5)
        
        # Set axis properties
        ax.set_xlim(-5, length + 5)
        ax.set_ylim(-5, width + 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig, ax
    
    def plot_players(
        self,
        ax: plt.Axes,
        home_positions: pd.DataFrame,
        away_positions: Optional[pd.DataFrame] = None,
        home_color: str = "#3498db",
        away_color: str = "#e74c3c",
        player_size: int = 200,
        show_numbers: bool = False,
        alpha: float = 0.8
    ) -> None:
        """
        Plot player positions on the pitch.
        رسم مواقع اللاعبين
        
        Args:
            ax: Matplotlib axes
            home_positions: DataFrame with 'x', 'y' columns for home team
            away_positions: DataFrame with 'x', 'y' columns for away team
            home_color: Color for home team players
            away_color: Color for away team players
            player_size: Size of player markers
            show_numbers: Whether to show player numbers
            alpha: Transparency
        """
        # Plot home team
        if not home_positions.empty and 'x' in home_positions.columns:
            ax.scatter(
                home_positions['x'], 
                home_positions['y'],
                s=player_size,
                c=home_color,
                edgecolors='white',
                linewidths=2,
                alpha=alpha,
                zorder=5
            )
            
            if show_numbers and 'number' in home_positions.columns:
                for _, player in home_positions.iterrows():
                    ax.annotate(
                        str(int(player['number'])),
                        (player['x'], player['y']),
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white',
                        fontweight='bold',
                        zorder=6
                    )
        
        # Plot away team
        if away_positions is not None and not away_positions.empty and 'x' in away_positions.columns:
            ax.scatter(
                away_positions['x'], 
                away_positions['y'],
                s=player_size,
                c=away_color,
                edgecolors='white',
                linewidths=2,
                alpha=alpha,
                zorder=5
            )
            
            if show_numbers and 'number' in away_positions.columns:
                for _, player in away_positions.iterrows():
                    ax.annotate(
                        str(int(player['number'])),
                        (player['x'], player['y']),
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white',
                        fontweight='bold',
                        zorder=6
                    )
    
    def plot_ball(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        color: str = "yellow",
        size: int = 100
    ) -> None:
        """
        Plot ball position.
        
        Args:
            ax: Matplotlib axes
            x: X coordinate
            y: Y coordinate
            color: Ball color
            size: Ball marker size
        """
        ax.scatter(
            x, y,
            s=size,
            c=color,
            edgecolors='black',
            linewidths=2,
            zorder=10
        )
    
    def plot_pass(
        self,
        ax: plt.Axes,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        color: str = "white",
        width: float = 2.0,
        alpha: float = 0.7
    ) -> None:
        """
        Plot a pass arrow.
        
        Args:
            ax: Matplotlib axes
            start_x, start_y: Starting position
            end_x, end_y: Ending position
            color: Arrow color
            width: Arrow width
            alpha: Transparency
        """
        ax.annotate(
            '',
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle='-|>',
                color=color,
                lw=width,
                alpha=alpha
            ),
            zorder=4
        )
    
    def plot_shot(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        goal: bool = False,
        size: int = 300,
        goal_color: str = "lime",
        miss_color: str = "red"
    ) -> None:
        """
        Plot a shot marker.
        
        Args:
            ax: Matplotlib axes
            x, y: Shot position
            goal: Whether the shot was a goal
            size: Marker size
            goal_color: Color for goals
            miss_color: Color for misses
        """
        color = goal_color if goal else miss_color
        marker = '*' if goal else 'o'
        
        ax.scatter(
            x, y,
            s=size,
            c=color,
            marker=marker,
            edgecolors='black',
            linewidths=1,
            zorder=7
        )
    
    def add_title(
        self,
        ax: plt.Axes,
        title: str,
        subtitle: Optional[str] = None,
        fontsize: int = 14
    ) -> None:
        """
        Add title and subtitle to the plot.
        
        Args:
            ax: Matplotlib axes
            title: Main title
            subtitle: Optional subtitle
            fontsize: Font size for title
        """
        ax.set_title(
            title,
            fontsize=fontsize,
            fontweight='bold',
            color='white',
            pad=20
        )
        
        if subtitle:
            ax.text(
                self.pitch_length / 2,
                self.pitch_width + 8,
                subtitle,
                ha='center',
                fontsize=fontsize - 2,
                color='lightgray'
            )
    
    def add_legend(
        self,
        ax: plt.Axes,
        home_name: str = "Home",
        away_name: str = "Away",
        home_color: str = "#3498db",
        away_color: str = "#e74c3c"
    ) -> None:
        """
        Add team legend to the plot.
        
        Args:
            ax: Matplotlib axes
            home_name: Home team name
            away_name: Away team name
            home_color: Home team color
            away_color: Away team color
        """
        legend_elements = [
            mpatches.Patch(color=home_color, label=home_name),
            mpatches.Patch(color=away_color, label=away_name),
        ]
        
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            framealpha=0.7,
            facecolor='black',
            edgecolor='white',
            labelcolor='white'
        )


def draw_corner_kick_setup(
    viz: PitchVisualizer,
    corner_side: str = "right",
    attacking_positions: Optional[pd.DataFrame] = None,
    defending_positions: Optional[pd.DataFrame] = None,
    ball_position: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a corner kick setup visualization.
    
    Args:
        viz: PitchVisualizer instance
        corner_side: "left" or "right" corner
        attacking_positions: Positions of attacking players
        defending_positions: Positions of defending players
        ball_position: Position of the ball
        
    Returns:
        Figure and axes tuple
    """
    fig, ax = viz.draw_pitch()
    
    # Default ball position for corner
    if ball_position is None:
        if corner_side == "right":
            ball_position = (0, viz.pitch_width)
        else:
            ball_position = (0, 0)
    
    # Plot ball
    viz.plot_ball(ax, ball_position[0], ball_position[1])
    
    # Plot players if provided
    if attacking_positions is not None:
        viz.plot_players(ax, attacking_positions, None, home_color="#2ecc71")
    
    if defending_positions is not None:
        viz.plot_players(ax, pd.DataFrame(), defending_positions, away_color="#e74c3c")
    
    viz.add_title(ax, f"Corner Kick - {corner_side.capitalize()} Side")
    
    return fig, ax
