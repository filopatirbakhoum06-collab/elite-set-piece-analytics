"""
Interactive pitch component for Streamlit dashboard.
مكون الملعب التفاعلي
"""

from typing import Optional, List, Dict, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


class PitchComponent:
    """
    Interactive football pitch component for Streamlit.
    مكون ملعب كرة القدم التفاعلي
    
    This component provides an interactive pitch visualization
    using Plotly for use within Streamlit dashboards.
    
    Example:
        >>> pitch = PitchComponent()
        >>> fig = pitch.create_pitch()
        >>> st.plotly_chart(fig)
    """
    
    # Pitch dimensions (FIFA standard)
    PITCH_LENGTH = 105
    PITCH_WIDTH = 68
    
    def __init__(
        self,
        pitch_color: str = "#22312b",
        line_color: str = "white",
        home_color: str = "#3498db",
        away_color: str = "#e74c3c"
    ):
        """
        Initialize the pitch component.
        
        Args:
            pitch_color: Background color
            line_color: Line color
            home_color: Home team color
            away_color: Away team color
        """
        self.pitch_color = pitch_color
        self.line_color = line_color
        self.home_color = home_color
        self.away_color = away_color
    
    def create_pitch(
        self,
        show_zones: bool = False,
        show_thirds: bool = False
    ) -> go.Figure:
        """
        Create an interactive pitch figure.
        
        Args:
            show_zones: Whether to show attacking zones
            show_thirds: Whether to show pitch thirds
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Set background
        fig.update_layout(
            plot_bgcolor=self.pitch_color,
            paper_bgcolor=self.pitch_color,
        )
        
        # Pitch outline
        fig.add_shape(
            type="rect",
            x0=0, y0=0,
            x1=self.PITCH_LENGTH, y1=self.PITCH_WIDTH,
            line=dict(color=self.line_color, width=2),
            fillcolor=self.pitch_color
        )
        
        # Halfway line
        fig.add_shape(
            type="line",
            x0=self.PITCH_LENGTH/2, y0=0,
            x1=self.PITCH_LENGTH/2, y1=self.PITCH_WIDTH,
            line=dict(color=self.line_color, width=2)
        )
        
        # Center circle
        theta = np.linspace(0, 2*np.pi, 100)
        r = 9.15
        x_circle = self.PITCH_LENGTH/2 + r * np.cos(theta)
        y_circle = self.PITCH_WIDTH/2 + r * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=x_circle, y=y_circle,
            mode='lines',
            line=dict(color=self.line_color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Penalty areas
        for x_base in [0, self.PITCH_LENGTH]:
            if x_base == 0:
                x0, x1 = 0, 16.5
            else:
                x0, x1 = self.PITCH_LENGTH - 16.5, self.PITCH_LENGTH
            
            y0 = (self.PITCH_WIDTH - 40.3) / 2
            y1 = y0 + 40.3
            
            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=self.line_color, width=2)
            )
        
        # Goal areas
        for x_base in [0, self.PITCH_LENGTH]:
            if x_base == 0:
                x0, x1 = 0, 5.5
            else:
                x0, x1 = self.PITCH_LENGTH - 5.5, self.PITCH_LENGTH
            
            y0 = (self.PITCH_WIDTH - 18.3) / 2
            y1 = y0 + 18.3
            
            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=self.line_color, width=2)
            )
        
        # Penalty spots
        for x in [11, self.PITCH_LENGTH - 11]:
            fig.add_trace(go.Scatter(
                x=[x], y=[self.PITCH_WIDTH/2],
                mode='markers',
                marker=dict(color=self.line_color, size=5),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Configure layout
        fig.update_layout(
            xaxis=dict(
                range=[-5, self.PITCH_LENGTH + 5],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                range=[-5, self.PITCH_WIDTH + 5],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            height=500
        )
        
        return fig
    
    def add_players(
        self,
        fig: go.Figure,
        home_positions: pd.DataFrame,
        away_positions: Optional[pd.DataFrame] = None,
        show_labels: bool = True
    ) -> go.Figure:
        """
        Add player positions to the pitch.
        
        Args:
            fig: Plotly Figure
            home_positions: Home team positions (x, y columns)
            away_positions: Away team positions
            show_labels: Whether to show player numbers
            
        Returns:
            Updated Figure
        """
        # Home team
        if not home_positions.empty and 'x' in home_positions.columns:
            text = home_positions.get('number', home_positions.index + 1).astype(str)
            
            fig.add_trace(go.Scatter(
                x=home_positions['x'],
                y=home_positions['y'],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    color=self.home_color,
                    size=20,
                    line=dict(color='white', width=2)
                ),
                text=text if show_labels else None,
                textfont=dict(color='white', size=10),
                textposition='middle center',
                name='Home',
                hovertemplate='Player %{text}<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>'
            ))
        
        # Away team
        if away_positions is not None and not away_positions.empty and 'x' in away_positions.columns:
            text = away_positions.get('number', away_positions.index + 1).astype(str)
            
            fig.add_trace(go.Scatter(
                x=away_positions['x'],
                y=away_positions['y'],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    color=self.away_color,
                    size=20,
                    line=dict(color='white', width=2)
                ),
                text=text if show_labels else None,
                textfont=dict(color='white', size=10),
                textposition='middle center',
                name='Away',
                hovertemplate='Player %{text}<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>'
            ))
        
        return fig
    
    def add_ball(
        self,
        fig: go.Figure,
        x: float,
        y: float
    ) -> go.Figure:
        """
        Add ball to the pitch.
        
        Args:
            fig: Plotly Figure
            x: Ball X position
            y: Ball Y position
            
        Returns:
            Updated Figure
        """
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(
                color='yellow',
                size=15,
                line=dict(color='black', width=2)
            ),
            name='Ball',
            hovertemplate='Ball<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>'
        ))
        
        return fig
    
    def add_pass_arrow(
        self,
        fig: go.Figure,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        color: str = "white"
    ) -> go.Figure:
        """
        Add a pass arrow to the pitch.
        
        Args:
            fig: Plotly Figure
            start_x, start_y: Starting position
            end_x, end_y: Ending position
            color: Arrow color
            
        Returns:
            Updated Figure
        """
        fig.add_annotation(
            x=end_x, y=end_y,
            ax=start_x, ay=start_y,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=color
        )
        
        return fig


def render_pitch_component(
    home_positions: Optional[pd.DataFrame] = None,
    away_positions: Optional[pd.DataFrame] = None,
    ball_position: Optional[Tuple[float, float]] = None,
    title: str = "Set-Piece Visualization"
) -> None:
    """
    Render an interactive pitch in Streamlit.
    
    Args:
        home_positions: Home team positions
        away_positions: Away team positions
        ball_position: Ball (x, y) position
        title: Chart title
    """
    pitch = PitchComponent()
    fig = pitch.create_pitch()
    
    if home_positions is not None:
        fig = pitch.add_players(fig, home_positions, away_positions)
    
    if ball_position is not None:
        fig = pitch.add_ball(fig, ball_position[0], ball_position[1])
    
    fig.update_layout(title=title)
    
    st.plotly_chart(fig, use_container_width=True)
