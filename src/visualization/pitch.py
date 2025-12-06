"""
Elite Set-Piece Analytics - Pitch Visualization
رسم الملعب للتحليلات الكروية

This module provides pitch visualization:
- Draw FIFA-standard pitch
- Plot set pieces
- Create heatmaps
- Visualize predictions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Rectangle, Circle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# FIFA standard pitch dimensions (in meters)
PITCH_LENGTH = 105
PITCH_WIDTH = 68
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.3
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.3
PENALTY_SPOT_DISTANCE = 11
CENTER_CIRCLE_RADIUS = 9.15
CORNER_ARC_RADIUS = 1
GOAL_WIDTH = 7.32


def draw_pitch(
    ax: Optional[plt.Axes] = None,
    color: str = '#228B22',
    linecolor: str = 'white',
    orientation: str = 'horizontal',
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a FIFA-standard football pitch
    رسم ملعب كرة قدم بالمواصفات الفيفا القياسية
    
    Args:
        ax: Matplotlib axes (creates new if None)
        color: Pitch background color
        linecolor: Line color
        orientation: 'horizontal' or 'vertical'
        figsize: Figure size
    
    Returns:
        Tuple of (figure, axes)
    """
    if ax is None:
        if orientation == 'horizontal':
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=(figsize[1], figsize[0]))
    else:
        fig = ax.figure
    
    if orientation == 'vertical':
        # Swap dimensions for vertical
        length, width = PITCH_WIDTH, PITCH_LENGTH
    else:
        length, width = PITCH_LENGTH, PITCH_WIDTH
    
    # Set background
    ax.set_facecolor(color)
    
    # Pitch outline
    ax.plot([0, length], [0, 0], color=linecolor, linewidth=2)
    ax.plot([0, length], [width, width], color=linecolor, linewidth=2)
    ax.plot([0, 0], [0, width], color=linecolor, linewidth=2)
    ax.plot([length, length], [0, width], color=linecolor, linewidth=2)
    
    # Halfway line
    ax.plot([length/2, length/2], [0, width], color=linecolor, linewidth=2)
    
    # Center circle
    center_circle = plt.Circle(
        (length/2, width/2), 
        CENTER_CIRCLE_RADIUS, 
        color=linecolor, 
        fill=False, 
        linewidth=2
    )
    ax.add_patch(center_circle)
    
    # Center spot
    ax.plot(length/2, width/2, 'o', color=linecolor, markersize=5)
    
    # Left penalty area
    left_penalty = Rectangle(
        (0, (width - PENALTY_AREA_WIDTH)/2),
        PENALTY_AREA_LENGTH,
        PENALTY_AREA_WIDTH,
        linewidth=2,
        edgecolor=linecolor,
        facecolor='none'
    )
    ax.add_patch(left_penalty)
    
    # Right penalty area
    right_penalty = Rectangle(
        (length - PENALTY_AREA_LENGTH, (width - PENALTY_AREA_WIDTH)/2),
        PENALTY_AREA_LENGTH,
        PENALTY_AREA_WIDTH,
        linewidth=2,
        edgecolor=linecolor,
        facecolor='none'
    )
    ax.add_patch(right_penalty)
    
    # Left goal area
    left_goal_area = Rectangle(
        (0, (width - GOAL_AREA_WIDTH)/2),
        GOAL_AREA_LENGTH,
        GOAL_AREA_WIDTH,
        linewidth=2,
        edgecolor=linecolor,
        facecolor='none'
    )
    ax.add_patch(left_goal_area)
    
    # Right goal area
    right_goal_area = Rectangle(
        (length - GOAL_AREA_LENGTH, (width - GOAL_AREA_WIDTH)/2),
        GOAL_AREA_LENGTH,
        GOAL_AREA_WIDTH,
        linewidth=2,
        edgecolor=linecolor,
        facecolor='none'
    )
    ax.add_patch(right_goal_area)
    
    # Penalty spots
    ax.plot(PENALTY_SPOT_DISTANCE, width/2, 'o', color=linecolor, markersize=4)
    ax.plot(length - PENALTY_SPOT_DISTANCE, width/2, 'o', color=linecolor, markersize=4)
    
    # Penalty arcs
    left_arc = Arc(
        (PENALTY_SPOT_DISTANCE, width/2),
        2 * CENTER_CIRCLE_RADIUS,
        2 * CENTER_CIRCLE_RADIUS,
        angle=0,
        theta1=-53,
        theta2=53,
        color=linecolor,
        linewidth=2
    )
    ax.add_patch(left_arc)
    
    right_arc = Arc(
        (length - PENALTY_SPOT_DISTANCE, width/2),
        2 * CENTER_CIRCLE_RADIUS,
        2 * CENTER_CIRCLE_RADIUS,
        angle=0,
        theta1=127,
        theta2=233,
        color=linecolor,
        linewidth=2
    )
    ax.add_patch(right_arc)
    
    # Corner arcs
    for x, y in [(0, 0), (0, width), (length, 0), (length, width)]:
        corner = Arc(
            (x, y),
            2 * CORNER_ARC_RADIUS,
            2 * CORNER_ARC_RADIUS,
            angle=0,
            theta1=0 if x == 0 and y == 0 else (90 if x == 0 and y == width else (270 if x == length and y == 0 else 180)),
            theta2=90 if x == 0 and y == 0 else (180 if x == 0 and y == width else (360 if x == length and y == 0 else 270)),
            color=linecolor,
            linewidth=2
        )
        ax.add_patch(corner)
    
    # Goals
    goal_y_start = (width - GOAL_WIDTH) / 2
    # Left goal
    ax.plot([-2, -2], [goal_y_start, goal_y_start + GOAL_WIDTH], color='gray', linewidth=4)
    ax.plot([-2, 0], [goal_y_start, goal_y_start], color='gray', linewidth=4)
    ax.plot([-2, 0], [goal_y_start + GOAL_WIDTH, goal_y_start + GOAL_WIDTH], color='gray', linewidth=4)
    # Right goal
    ax.plot([length + 2, length + 2], [goal_y_start, goal_y_start + GOAL_WIDTH], color='gray', linewidth=4)
    ax.plot([length, length + 2], [goal_y_start, goal_y_start], color='gray', linewidth=4)
    ax.plot([length, length + 2], [goal_y_start + GOAL_WIDTH, goal_y_start + GOAL_WIDTH], color='gray', linewidth=4)
    
    # Set limits and aspect
    ax.set_xlim(-5, length + 5)
    ax.set_ylim(-3, width + 3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig, ax


def plot_set_piece(
    set_piece_data: Dict,
    ax: Optional[plt.Axes] = None,
    predictions: Optional[Dict] = None,
    show_positions: bool = True,
    show_arrows: bool = True,
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize a single set piece
    رسم كرة ثابتة واحدة
    
    Args:
        set_piece_data: Dictionary with set piece information
        ax: Matplotlib axes
        predictions: Optional prediction data
        show_positions: Whether to show player positions
        show_arrows: Whether to show movement arrows
        title: Plot title
    
    Returns:
        Tuple of (figure, axes)
    """
    # Draw pitch
    fig, ax = draw_pitch(ax)
    
    # Scale coordinates from 0-100 to pitch dimensions
    def scale_x(x):
        return x / 100 * PITCH_LENGTH
    
    def scale_y(y):
        return y / 100 * PITCH_WIDTH
    
    # Ball position
    ball_x = scale_x(set_piece_data.get('x', 50))
    ball_y = scale_y(set_piece_data.get('y', 50))
    
    # Draw ball
    ax.plot(ball_x, ball_y, 'o', color='white', markersize=12, 
            markeredgecolor='black', markeredgewidth=2, zorder=10)
    
    if show_positions:
        # Draw attackers
        attackers = set_piece_data.get('attacker_positions', [])
        for i, att in enumerate(attackers):
            att_x = scale_x(att['x'])
            att_y = scale_y(att['y'])
            
            # Highlight predicted receiver
            if predictions and predictions.get('predicted_receiver') == i:
                ax.plot(att_x, att_y, 'o', color='gold', markersize=18, 
                        markeredgecolor='black', markeredgewidth=2, zorder=8)
                ax.annotate('★', (att_x, att_y), fontsize=12, ha='center', va='center', 
                           fontweight='bold', color='black', zorder=9)
            else:
                ax.plot(att_x, att_y, 'o', color='red', markersize=12, 
                        markeredgecolor='white', markeredgewidth=1, zorder=6)
            
            # Player number
            ax.annotate(str(att.get('player_id', i+1)), (att_x + 1, att_y + 1), 
                       fontsize=8, color='red', fontweight='bold')
        
        # Draw defenders
        defenders = set_piece_data.get('defender_positions', [])
        for i, defn in enumerate(defenders):
            def_x = scale_x(defn['x'])
            def_y = scale_y(defn['y'])
            ax.plot(def_x, def_y, 's', color='blue', markersize=10, 
                    markeredgecolor='white', markeredgewidth=1, zorder=5)
    
    if show_arrows and predictions:
        # Draw predicted trajectory
        if 'target_position' in predictions:
            target_x = scale_x(predictions['target_position']['x'])
            target_y = scale_y(predictions['target_position']['y'])
            
            ax.annotate('', xy=(target_x, target_y), xytext=(ball_x, ball_y),
                       arrowprops=dict(arrowstyle='->', color='yellow', lw=3),
                       zorder=7)
    
    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    else:
        sp_type = set_piece_data.get('type', 'Set Piece').title()
        outcome = set_piece_data.get('outcome', '')
        minute = set_piece_data.get('minute', 0)
        ax.set_title(f"{sp_type} (min {minute}) - {outcome.title()}", 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_heatmap(
    df: pd.DataFrame,
    zone_type: str = 'receiver',
    ax: Optional[plt.Axes] = None,
    cmap: str = 'hot',
    alpha: float = 0.6,
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create heatmap of set piece data
    إنشاء خريطة حرارية لبيانات الكرات الثابتة
    
    Args:
        df: DataFrame with set piece data
        zone_type: 'receiver', 'success', or 'danger'
        ax: Matplotlib axes
        cmap: Colormap name
        alpha: Transparency
        title: Plot title
    
    Returns:
        Tuple of (figure, axes)
    """
    # Draw pitch
    fig, ax = draw_pitch(ax)
    
    # Create grid
    x_bins = np.linspace(0, PITCH_LENGTH, 21)
    y_bins = np.linspace(0, PITCH_WIDTH, 15)
    
    # Scale coordinates
    x_scaled = df['x'] / 100 * PITCH_LENGTH
    y_scaled = df['y'] / 100 * PITCH_WIDTH
    
    if zone_type == 'success':
        # Weight by success
        weights = df['outcome'].isin(['goal', 'shot']).astype(float)
        heatmap, xedges, yedges = np.histogram2d(
            x_scaled, y_scaled, bins=[x_bins, y_bins], weights=weights
        )
        # Normalize by count
        counts, _, _ = np.histogram2d(x_scaled, y_scaled, bins=[x_bins, y_bins])
        heatmap = np.divide(heatmap, counts, where=counts != 0)
    elif zone_type == 'danger':
        # Weight by danger index if available
        if 'danger_index' in df.columns:
            weights = df['danger_index']
        else:
            weights = None
        heatmap, xedges, yedges = np.histogram2d(
            x_scaled, y_scaled, bins=[x_bins, y_bins], weights=weights
        )
    else:  # receiver
        heatmap, xedges, yedges = np.histogram2d(
            x_scaled, y_scaled, bins=[x_bins, y_bins]
        )
    
    # Transpose for correct orientation
    heatmap = heatmap.T
    
    # Plot heatmap
    im = ax.imshow(
        heatmap,
        extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
        origin='lower',
        cmap=cmap,
        alpha=alpha,
        aspect='auto',
        zorder=1
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    if zone_type == 'success':
        cbar.set_label('Success Rate', fontsize=10)
    elif zone_type == 'danger':
        cbar.set_label('Danger Level', fontsize=10)
    else:
        cbar.set_label('Frequency', fontsize=10)
    
    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"Set Piece Heatmap ({zone_type.title()})", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_team_comparison(
    df: pd.DataFrame,
    teams: List[str],
    metric: str = 'success_rate',
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare teams on set piece metrics
    مقارنة الفرق في مقاييس الكرات الثابتة
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    team_data = []
    for team in teams:
        team_df = df[df['team'] == team]
        if len(team_df) > 0:
            if metric == 'success_rate':
                value = team_df['outcome'].isin(['goal', 'shot']).mean() * 100
            elif metric == 'goal_rate':
                value = (team_df['outcome'] == 'goal').mean() * 100
            elif metric == 'total':
                value = len(team_df)
            else:
                value = 0
            team_data.append({'team': team, 'value': value})
    
    team_data = sorted(team_data, key=lambda x: x['value'], reverse=True)
    
    # Plot bars
    teams_sorted = [d['team'] for d in team_data]
    values = [d['value'] for d in team_data]
    
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(teams_sorted)))
    bars = ax.barh(teams_sorted, values, color=colors)
    
    # Labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%' if metric != 'total' else str(int(val)),
               va='center', fontsize=10)
    
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(f"Team Comparison: {metric.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_set_piece_distribution(
    df: pd.DataFrame,
    by: str = 'type',
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot distribution of set pieces
    رسم توزيع الكرات الثابتة
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    counts = df[by].value_counts()
    
    # Create pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.05] * len(counts),
        shadow=True
    )
    
    ax.set_title(f"Set Piece Distribution by {by.title()}", fontsize=14, fontweight='bold')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    plt.tight_layout()
    return fig, ax


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot feature importance
    رسم أهمية الميزات
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Plot horizontal bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(
        top_features['feature'],
        top_features['importance'],
        color=colors
    )
    
    ax.set_xlabel('Importance Score')
    ax.set_title('Top Feature Importances', fontsize=14, fontweight='bold')
    
    # Invert y axis to have most important at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig, ax


def plot_minute_distribution(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot distribution of set pieces by game minute
    رسم توزيع الكرات الثابتة حسب دقيقة المباراة
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    bins = list(range(0, 100, 5))
    ax.hist(df['minute'], bins=bins, color='steelblue', edgecolor='white', alpha=0.7)
    
    # Add vertical lines for half time and full time
    ax.axvline(x=45, color='red', linestyle='--', linewidth=2, label='Half Time')
    ax.axvline(x=90, color='green', linestyle='--', linewidth=2, label='Full Time')
    
    ax.set_xlabel('Game Minute')
    ax.set_ylabel('Number of Set Pieces')
    ax.set_title('Set Pieces by Game Minute', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return fig, ax


def create_summary_dashboard(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a summary dashboard with multiple plots
    إنشاء لوحة معلومات ملخصة مع عدة رسوم
    """
    fig = plt.figure(figsize=figsize)
    
    # 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Plot 1: Heatmap on pitch
    _, ax1 = plot_heatmap(df, zone_type='receiver', ax=ax1)
    
    # Plot 2: Distribution by type
    counts = df['type'].value_counts()
    ax2.bar(counts.index, counts.values, color=plt.cm.Set2(np.linspace(0, 1, len(counts))))
    ax2.set_title('Set Pieces by Type', fontweight='bold')
    ax2.set_xticklabels(counts.index, rotation=45, ha='right')
    
    # Plot 3: Minute distribution
    ax3.hist(df['minute'], bins=18, color='steelblue', edgecolor='white', alpha=0.7)
    ax3.axvline(x=45, color='red', linestyle='--', label='Half Time')
    ax3.set_title('Set Pieces by Minute', fontweight='bold')
    ax3.set_xlabel('Game Minute')
    ax3.legend()
    
    # Plot 4: Outcome distribution
    outcomes = df['outcome'].value_counts()
    colors = {'goal': 'gold', 'shot': 'orange', 'possession_retained': 'green', 
              'possession_lost': 'red', 'clearance': 'gray'}
    bar_colors = [colors.get(o, 'blue') for o in outcomes.index]
    ax4.bar(outcomes.index, outcomes.values, color=bar_colors)
    ax4.set_title('Outcomes', fontweight='bold')
    ax4.set_xticklabels(outcomes.index, rotation=45, ha='right')
    
    fig.suptitle('⚽ Set Piece Analytics Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Test visualizations
    import sys
    sys.path.append('..')
    from data.loaders import load_wyscout_data
    from data.extractors import extract_set_pieces
    
    print("=" * 50)
    print("Testing Pitch Visualizations")
    print("=" * 50)
    
    # Load data
    df = load_wyscout_data()
    set_pieces = extract_set_pieces(df)
    
    # Test pitch drawing
    print("\n🏟️ Drawing pitch...")
    fig, ax = draw_pitch()
    plt.savefig('/tmp/test_pitch.png', dpi=100, bbox_inches='tight')
    print("   Saved to /tmp/test_pitch.png")
    
    # Test heatmap
    print("\n🔥 Creating heatmap...")
    fig, ax = plot_heatmap(set_pieces, zone_type='receiver')
    plt.savefig('/tmp/test_heatmap.png', dpi=100, bbox_inches='tight')
    print("   Saved to /tmp/test_heatmap.png")
    
    # Test set piece plot
    if len(set_pieces) > 0:
        print("\n⚽ Plotting single set piece...")
        sample = set_pieces.iloc[0].to_dict()
        fig, ax = plot_set_piece(sample)
        plt.savefig('/tmp/test_setpiece.png', dpi=100, bbox_inches='tight')
        print("   Saved to /tmp/test_setpiece.png")
    
    print("\n✅ Visualizations complete!")
