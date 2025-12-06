"""
Elite Set-Piece Analytics - Animations
الرسوم المتحركة لتحليلات الكرات الثابتة

This module provides animation capabilities:
- Animate set piece execution
- Show player movements
- Ball trajectory animation
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
from typing import Dict, List, Optional, Tuple

from .pitch import draw_pitch, PITCH_LENGTH, PITCH_WIDTH


def animate_set_piece(
    set_piece_data: Dict,
    duration: float = 3.0,
    fps: int = 30,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """
    Create animated visualization of set piece
    إنشاء تصور متحرك للكرة الثابتة
    
    Args:
        set_piece_data: Dictionary with set piece information
        duration: Animation duration in seconds
        fps: Frames per second
        save_path: Optional path to save animation
    
    Returns:
        Matplotlib animation object
    """
    # Create figure and draw pitch
    fig, ax = draw_pitch()
    
    # Scale coordinates
    def scale_x(x):
        return x / 100 * PITCH_LENGTH
    
    def scale_y(y):
        return y / 100 * PITCH_WIDTH
    
    # Ball starting position
    ball_start_x = scale_x(set_piece_data.get('x', 0))
    ball_start_y = scale_y(set_piece_data.get('y', 50))
    
    # Ball end position (target area near goal)
    ball_end_x = scale_x(90)
    ball_end_y = scale_y(50)
    
    # Create ball marker
    ball = Circle((ball_start_x, ball_start_y), 1.5, color='white', 
                  ec='black', linewidth=2, zorder=10)
    ax.add_patch(ball)
    
    # Create attacker markers
    attackers = set_piece_data.get('attacker_positions', [])
    attacker_markers = []
    for att in attackers:
        marker = Circle(
            (scale_x(att['x']), scale_y(att['y'])), 
            1.0, color='red', ec='white', linewidth=1, zorder=6
        )
        ax.add_patch(marker)
        attacker_markers.append({
            'marker': marker,
            'start_x': scale_x(att['x']),
            'start_y': scale_y(att['y']),
            'end_x': scale_x(att['x'] + np.random.uniform(5, 15)),  # Move toward goal
            'end_y': scale_y(att['y'] + np.random.uniform(-5, 5))
        })
    
    # Create defender markers
    defenders = set_piece_data.get('defender_positions', [])
    defender_markers = []
    for defn in defenders:
        marker = Circle(
            (scale_x(defn['x']), scale_y(defn['y'])), 
            1.0, color='blue', ec='white', linewidth=1, zorder=5
        )
        ax.add_patch(marker)
        defender_markers.append({
            'marker': marker,
            'start_x': scale_x(defn['x']),
            'start_y': scale_y(defn['y']),
            'end_x': scale_x(defn['x'] + np.random.uniform(-2, 5)),
            'end_y': scale_y(defn['y'] + np.random.uniform(-3, 3))
        })
    
    # Number of frames
    n_frames = int(duration * fps)
    
    def init():
        """Initialize animation"""
        ball.center = (ball_start_x, ball_start_y)
        return [ball] + [m['marker'] for m in attacker_markers + defender_markers]
    
    def animate(frame):
        """Animate each frame"""
        progress = frame / n_frames
        
        # Ease function for smooth movement
        ease = np.sin(progress * np.pi / 2)  # Ease out
        
        # Ball trajectory (curved path)
        t = progress
        # Bezier curve for ball
        control_x = (ball_start_x + ball_end_x) / 2
        control_y = ball_start_y + 20  # Arc up
        
        ball_x = (1-t)**2 * ball_start_x + 2*(1-t)*t * control_x + t**2 * ball_end_x
        ball_y = (1-t)**2 * ball_start_y + 2*(1-t)*t * control_y + t**2 * ball_end_y
        
        ball.center = (ball_x, ball_y)
        
        # Move attackers
        for att in attacker_markers:
            # Delay attacker movement slightly
            delayed_progress = max(0, (progress - 0.2) / 0.8)
            new_x = att['start_x'] + delayed_progress * (att['end_x'] - att['start_x'])
            new_y = att['start_y'] + delayed_progress * (att['end_y'] - att['start_y'])
            att['marker'].center = (new_x, new_y)
        
        # Move defenders
        for defn in defender_markers:
            # Defenders react slower
            delayed_progress = max(0, (progress - 0.3) / 0.7)
            new_x = defn['start_x'] + delayed_progress * (defn['end_x'] - defn['start_x'])
            new_y = defn['start_y'] + delayed_progress * (defn['end_y'] - defn['start_y'])
            defn['marker'].center = (new_x, new_y)
        
        return [ball] + [m['marker'] for m in attacker_markers + defender_markers]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=1000/fps, blit=True
    )
    
    # Save if path provided
    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"💾 Animation saved to {save_path}")
        except Exception as e:
            print(f"⚠️ Could not save animation: {e}")
    
    return anim


def create_trajectory_animation(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    control_points: Optional[List[Tuple[float, float]]] = None,
    duration: float = 2.0,
    fps: int = 30
) -> animation.FuncAnimation:
    """
    Create ball trajectory animation
    إنشاء رسم متحرك لمسار الكرة
    
    Args:
        start_pos: Starting (x, y) position
        end_pos: Ending (x, y) position
        control_points: Optional bezier control points
        duration: Animation duration
        fps: Frames per second
    
    Returns:
        Animation object
    """
    fig, ax = draw_pitch()
    
    # Scale positions
    start_x, start_y = start_pos[0] / 100 * PITCH_LENGTH, start_pos[1] / 100 * PITCH_WIDTH
    end_x, end_y = end_pos[0] / 100 * PITCH_LENGTH, end_pos[1] / 100 * PITCH_WIDTH
    
    # Ball
    ball = Circle((start_x, start_y), 1.5, color='white', ec='black', linewidth=2, zorder=10)
    ax.add_patch(ball)
    
    # Trail
    trail_x, trail_y = [], []
    trail_line, = ax.plot([], [], 'y--', linewidth=2, alpha=0.5, zorder=9)
    
    n_frames = int(duration * fps)
    
    # Control point for bezier curve
    if control_points:
        ctrl_x, ctrl_y = control_points[0][0] / 100 * PITCH_LENGTH, control_points[0][1] / 100 * PITCH_WIDTH
    else:
        ctrl_x = (start_x + end_x) / 2
        ctrl_y = max(start_y, end_y) + 15  # Arc upward
    
    def init():
        ball.center = (start_x, start_y)
        trail_x.clear()
        trail_y.clear()
        trail_line.set_data([], [])
        return [ball, trail_line]
    
    def animate(frame):
        t = frame / n_frames
        
        # Quadratic bezier curve
        x = (1-t)**2 * start_x + 2*(1-t)*t * ctrl_x + t**2 * end_x
        y = (1-t)**2 * start_y + 2*(1-t)*t * ctrl_y + t**2 * end_y
        
        ball.center = (x, y)
        
        trail_x.append(x)
        trail_y.append(y)
        trail_line.set_data(trail_x, trail_y)
        
        return [ball, trail_line]
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=1000/fps, blit=True
    )
    
    return anim


def animate_prediction(
    set_piece_data: Dict,
    predictions: Dict,
    duration: float = 4.0,
    fps: int = 30,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """
    Animate set piece with prediction visualization
    تحريك الكرة الثابتة مع عرض التنبؤات
    
    Shows:
    - Ball trajectory to predicted receiver
    - Receiver highlighted
    - Probability display
    """
    fig, ax = draw_pitch()
    
    def scale_x(x):
        return x / 100 * PITCH_LENGTH
    
    def scale_y(y):
        return y / 100 * PITCH_WIDTH
    
    # Ball
    ball_x = scale_x(set_piece_data.get('x', 0))
    ball_y = scale_y(set_piece_data.get('y', 50))
    ball = Circle((ball_x, ball_y), 1.5, color='white', ec='black', linewidth=2, zorder=10)
    ax.add_patch(ball)
    
    # Attackers
    attackers = set_piece_data.get('attacker_positions', [])
    predicted_idx = predictions.get('predicted_receiver', 0)
    
    attacker_circles = []
    for i, att in enumerate(attackers):
        color = 'gold' if i == predicted_idx else 'red'
        size = 2.0 if i == predicted_idx else 1.0
        circle = Circle(
            (scale_x(att['x']), scale_y(att['y'])),
            size, color=color, ec='white' if i != predicted_idx else 'black',
            linewidth=2 if i == predicted_idx else 1, zorder=8 if i == predicted_idx else 6
        )
        ax.add_patch(circle)
        attacker_circles.append(circle)
    
    # Target position (predicted receiver)
    if predicted_idx < len(attackers):
        target_x = scale_x(attackers[predicted_idx]['x'])
        target_y = scale_y(attackers[predicted_idx]['y'])
    else:
        target_x, target_y = scale_x(90), scale_y(50)
    
    # Trajectory line
    trajectory, = ax.plot([], [], 'y-', linewidth=3, alpha=0.7, zorder=9)
    
    # Probability text
    prob = predictions.get('probability', 0.75)
    prob_text = ax.text(
        PITCH_LENGTH / 2, PITCH_WIDTH + 5,
        f'Predicted Receiver: Player {predicted_idx + 1} ({prob:.1%} confidence)',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
    )
    
    n_frames = int(duration * fps)
    
    def init():
        ball.center = (ball_x, ball_y)
        trajectory.set_data([], [])
        return [ball, trajectory]
    
    def animate(frame):
        progress = frame / n_frames
        
        # Ball movement (wait, then move)
        if progress < 0.3:
            # Wait phase
            pass
        else:
            # Movement phase
            move_progress = (progress - 0.3) / 0.7
            
            # Bezier curve
            t = move_progress
            ctrl_x = (ball_x + target_x) / 2
            ctrl_y = ball_y + 15
            
            curr_x = (1-t)**2 * ball_x + 2*(1-t)*t * ctrl_x + t**2 * target_x
            curr_y = (1-t)**2 * ball_y + 2*(1-t)*t * ctrl_y + t**2 * target_y
            
            ball.center = (curr_x, curr_y)
            
            # Draw trajectory
            ts = np.linspace(0, t, 50)
            xs = (1-ts)**2 * ball_x + 2*(1-ts)*ts * ctrl_x + ts**2 * target_x
            ys = (1-ts)**2 * ball_y + 2*(1-ts)*ts * ctrl_y + ts**2 * target_y
            trajectory.set_data(xs, ys)
            
            # Highlight receiver when ball arrives
            if move_progress > 0.9:
                attacker_circles[predicted_idx].set_radius(2.5)
        
        return [ball, trajectory] + attacker_circles
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=1000/fps, blit=True
    )
    
    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"💾 Animation saved to {save_path}")
        except Exception as e:
            print(f"⚠️ Could not save animation: {e}")
    
    return anim


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Animations")
    print("=" * 50)
    
    # Create sample data
    sample_data = {
        'x': 0,
        'y': 50,
        'type': 'corner',
        'attacker_positions': [
            {'x': 85, 'y': 40, 'player_id': 1},
            {'x': 88, 'y': 50, 'player_id': 2},
            {'x': 82, 'y': 60, 'player_id': 3},
        ],
        'defender_positions': [
            {'x': 90, 'y': 45, 'player_id': 1},
            {'x': 92, 'y': 55, 'player_id': 2},
        ]
    }
    
    print("\n🎬 Creating animation...")
    anim = animate_set_piece(sample_data, duration=2.0, fps=15)
    print("   Animation created (display in interactive mode)")
    
    print("\n✅ Animation module loaded successfully!")
