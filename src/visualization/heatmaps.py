"""
Heatmap utilities for Elite Set-Piece Analytics.
أدوات الخرائط الحرارية

This module provides functionality to create heatmaps for
visualizing player positions, shot locations, and pass patterns.
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from ..utils.config import Config
from .pitch import PitchVisualizer


class HeatmapGenerator:
    """
    Generator for football analytics heatmaps.
    منشئ الخرائط الحرارية
    
    Creates various types of heatmaps including:
    - Player position heatmaps
    - Shot location maps
    - Pass origin/destination maps
    
    Example:
        >>> generator = HeatmapGenerator()
        >>> fig, ax = generator.create_position_heatmap(positions_df)
    """
    
    def __init__(
        self,
        pitch_viz: Optional[PitchVisualizer] = None,
        cmap: str = 'hot',
        alpha: float = 0.6,
        bins: int = 25
    ):
        """
        Initialize the heatmap generator.
        
        Args:
            pitch_viz: PitchVisualizer instance
            cmap: Colormap name
            alpha: Transparency
            bins: Number of bins for histogram
        """
        self.pitch_viz = pitch_viz or PitchVisualizer()
        self.cmap = cmap
        self.alpha = alpha
        self.bins = bins
        self.config = Config()
    
    def create_position_heatmap(
        self,
        positions: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y',
        ax: Optional[plt.Axes] = None,
        title: str = "Position Heatmap",
        sigma: float = 2.0
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a heatmap of positions.
        إنشاء خريطة حرارية للمواقع
        
        Args:
            positions: DataFrame with position data
            x_col: X coordinate column name
            y_col: Y coordinate column name
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            sigma: Gaussian smoothing sigma
            
        Returns:
            Figure and axes tuple
        """
        if ax is None:
            fig, ax = self.pitch_viz.draw_pitch()
        else:
            fig = ax.figure
        
        # Create 2D histogram
        x = positions[x_col].dropna()
        y = positions[y_col].dropna()
        
        heatmap, xedges, yedges = np.histogram2d(
            x, y,
            bins=self.bins,
            range=[[0, self.pitch_viz.pitch_length], 
                   [0, self.pitch_viz.pitch_width]]
        )
        
        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Plot heatmap
        extent = [0, self.pitch_viz.pitch_length, 0, self.pitch_viz.pitch_width]
        ax.imshow(
            heatmap.T,
            extent=extent,
            origin='lower',
            cmap=self.cmap,
            alpha=self.alpha,
            zorder=1
        )
        
        self.pitch_viz.add_title(ax, title)
        
        return fig, ax
    
    def create_shot_map(
        self,
        shots: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y',
        goal_col: str = 'goal',
        xg_col: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        title: str = "Shot Map"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a shot location map.
        إنشاء خريطة التسديدات
        
        Args:
            shots: DataFrame with shot data
            x_col: X coordinate column
            y_col: Y coordinate column
            goal_col: Goal indicator column
            xg_col: Expected goals column (for sizing)
            ax: Matplotlib axes
            title: Plot title
            
        Returns:
            Figure and axes tuple
        """
        if ax is None:
            fig, ax = self.pitch_viz.draw_pitch()
        else:
            fig = ax.figure
        
        if xg_col and xg_col in shots.columns:
            sizes = shots[xg_col] * 500 + 50
        else:
            sizes = [100] * len(shots)
        
        # Plot missed shots
        if goal_col in shots.columns:
            missed = shots[shots[goal_col] == 0]
            ax.scatter(
                missed[x_col], missed[y_col],
                s=[sizes[i] for i in missed.index],
                c='red',
                alpha=0.6,
                edgecolors='white',
                zorder=5
            )
            
            # Plot goals
            goals = shots[shots[goal_col] == 1]
            ax.scatter(
                goals[x_col], goals[y_col],
                s=[sizes[i] for i in goals.index],
                c='lime',
                marker='*',
                edgecolors='white',
                zorder=6
            )
        else:
            ax.scatter(
                shots[x_col], shots[y_col],
                s=sizes,
                c='blue',
                alpha=0.6,
                edgecolors='white',
                zorder=5
            )
        
        self.pitch_viz.add_title(ax, title)
        
        return fig, ax
    
    def create_pass_heatmap(
        self,
        passes: pd.DataFrame,
        start_x: str = 'x',
        start_y: str = 'y',
        end_x: str = 'end_x',
        end_y: str = 'end_y',
        heatmap_type: str = 'origin',
        ax: Optional[plt.Axes] = None,
        title: str = "Pass Heatmap"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a pass heatmap.
        
        Args:
            passes: DataFrame with pass data
            start_x, start_y: Starting position columns
            end_x, end_y: Ending position columns
            heatmap_type: 'origin', 'destination', or 'both'
            ax: Matplotlib axes
            title: Plot title
            
        Returns:
            Figure and axes tuple
        """
        if ax is None:
            fig, ax = self.pitch_viz.draw_pitch()
        else:
            fig = ax.figure
        
        if heatmap_type in ['origin', 'both']:
            x = passes[start_x].dropna()
            y = passes[start_y].dropna()
        else:
            x = passes[end_x].dropna()
            y = passes[end_y].dropna()
        
        heatmap, xedges, yedges = np.histogram2d(
            x, y,
            bins=self.bins,
            range=[[0, self.pitch_viz.pitch_length], 
                   [0, self.pitch_viz.pitch_width]]
        )
        
        heatmap = gaussian_filter(heatmap, sigma=2.0)
        
        extent = [0, self.pitch_viz.pitch_length, 0, self.pitch_viz.pitch_width]
        ax.imshow(
            heatmap.T,
            extent=extent,
            origin='lower',
            cmap='Blues',
            alpha=self.alpha,
            zorder=1
        )
        
        self.pitch_viz.add_title(ax, title)
        
        return fig, ax
    
    def create_zone_heatmap(
        self,
        data: pd.DataFrame,
        value_col: str,
        x_col: str = 'zone_x',
        y_col: str = 'zone_y',
        ax: Optional[plt.Axes] = None,
        title: str = "Zone Heatmap",
        n_zones_x: int = 6,
        n_zones_y: int = 4
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a zone-based heatmap.
        
        Args:
            data: DataFrame with zone-aggregated data
            value_col: Column with values to display
            x_col: Zone X index column
            y_col: Zone Y index column
            ax: Matplotlib axes
            title: Plot title
            n_zones_x: Number of zones in X direction
            n_zones_y: Number of zones in Y direction
            
        Returns:
            Figure and axes tuple
        """
        if ax is None:
            fig, ax = self.pitch_viz.draw_pitch()
        else:
            fig = ax.figure
        
        # Create zone grid
        zone_width = self.pitch_viz.pitch_length / n_zones_x
        zone_height = self.pitch_viz.pitch_width / n_zones_y
        
        # Create matrix
        zone_matrix = np.zeros((n_zones_x, n_zones_y))
        
        for _, row in data.iterrows():
            if x_col in row and y_col in row:
                zone_x = int(row[x_col])
                zone_y = int(row[y_col])
                if 0 <= zone_x < n_zones_x and 0 <= zone_y < n_zones_y:
                    zone_matrix[zone_x, zone_y] = row[value_col]
        
        # Plot
        extent = [0, self.pitch_viz.pitch_length, 0, self.pitch_viz.pitch_width]
        ax.imshow(
            zone_matrix.T,
            extent=extent,
            origin='lower',
            cmap='RdYlGn',
            alpha=self.alpha,
            zorder=1
        )
        
        self.pitch_viz.add_title(ax, title)
        
        return fig, ax
