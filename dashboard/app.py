"""
Elite Set-Piece Analytics - Interactive Dashboard
لوحة التحكم التفاعلية لتحليلات الكرات الثابتة

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_wyscout_data
from src.data.extractors import extract_set_pieces, calculate_success_metrics
from src.features.spatial import calculate_spatial_features
from src.features.temporal import calculate_temporal_features
from src.features.physical import calculate_physical_features
from src.visualization.pitch import (
    draw_pitch, plot_heatmap, plot_set_piece,
    plot_minute_distribution, plot_set_piece_distribution, PITCH_LENGTH, PITCH_WIDTH
)

# Page configuration
st.set_page_config(
    page_title="⚽ Elite Set-Piece Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and process data with caching"""
    df = load_wyscout_data()
    set_pieces = extract_set_pieces(df)
    set_pieces = calculate_spatial_features(set_pieces)
    set_pieces = calculate_temporal_features(set_pieces)
    set_pieces = calculate_physical_features(set_pieces)
    return set_pieces


def main():
    # Sidebar navigation
    st.sidebar.title("⚽ Navigation")
    
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Analysis", "🎯 Predictions", "🗺️ Heatmaps", "📈 Statistics", "🎮 Designer"]
    )
    
    # Load data
    with st.spinner("📥 Loading World Cup 2022 data..."):
        df = load_data()
    
    # Route to different pages
    if page == "🏠 Home":
        show_home(df)
    elif page == "📊 Analysis":
        show_analysis(df)
    elif page == "🎯 Predictions":
        show_predictions(df)
    elif page == "🗺️ Heatmaps":
        show_heatmaps(df)
    elif page == "📈 Statistics":
        show_statistics(df)
    elif page == "🎮 Designer":
        show_designer(df)


def show_home(df):
    """Home page with overview"""
    st.markdown('<h1 class="main-header">⚽ Elite Set-Piece Analytics Platform</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🏆 World Cup 2022 Insights")
    st.markdown("""
    Welcome to the Elite Set-Piece Analytics Platform! This tool provides comprehensive 
    analysis of set pieces from the FIFA World Cup 2022, featuring:
    
    - 📊 **Advanced Statistical Analysis** - Deep dive into set piece patterns
    - 🎯 **AI-Powered Predictions** - Predict first receiver and outcomes  
    - 🗺️ **Interactive Heatmaps** - Visualize tactical patterns
    - 📈 **Team Comparisons** - Compare set piece effectiveness
    """)
    
    # Key metrics
    st.markdown("### 📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Set Pieces", f"{len(df):,}")
    
    with col2:
        st.metric("Unique Matches", f"{df['match_id'].nunique()}")
    
    with col3:
        avg_per_match = len(df) / df['match_id'].nunique()
        st.metric("Avg per Match", f"{avg_per_match:.1f}")
    
    with col4:
        goal_rate = (df['outcome'] == 'goal').mean() * 100
        st.metric("Goal Rate", f"{goal_rate:.1f}%")
    
    # Quick visualizations
    st.markdown("### 🎯 Set Piece Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        type_counts = df['type'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
        ax.bar(type_counts.index, type_counts.values, color=colors)
        ax.set_title('Set Pieces by Type', fontweight='bold')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        outcome_counts = df['outcome'].value_counts()
        colors_outcome = {'goal': 'gold', 'shot': 'orange', 'possession_retained': 'green',
                         'possession_lost': 'red', 'clearance': 'gray'}
        bar_colors = [colors_outcome.get(o, 'blue') for o in outcome_counts.index]
        ax.bar(outcome_counts.index, outcome_counts.values, color=bar_colors)
        ax.set_title('Outcomes', fontweight='bold')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close()
    
    # Sample heatmap
    st.markdown("### 🗺️ Set Piece Location Heatmap")
    fig, ax = plot_heatmap(df, zone_type='receiver')
    st.pyplot(fig)
    plt.close()


def show_analysis(df):
    """Detailed analysis page"""
    st.title("📊 Detailed Analysis")
    
    # Filters
    st.sidebar.markdown("### 🔍 Filters")
    
    teams = ['All Teams'] + sorted(df['team'].unique().tolist())
    selected_team = st.sidebar.selectbox("Select Team", teams)
    
    types = ['All Types'] + sorted(df['type'].unique().tolist())
    selected_type = st.sidebar.selectbox("Set Piece Type", types)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_team != 'All Teams':
        filtered_df = filtered_df[filtered_df['team'] == selected_team]
    if selected_type != 'All Types':
        filtered_df = filtered_df[filtered_df['type'] == selected_type]
    
    st.markdown(f"### Analyzing {len(filtered_df)} set pieces")
    
    # Metrics for filtered data
    metrics = calculate_success_metrics(filtered_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
    with col2:
        st.metric("Goal Rate", f"{metrics['goal_rate']:.1f}%")
    with col3:
        st.metric("Shot Rate", f"{metrics['shot_rate']:.1f}%")
    with col4:
        st.metric("Possession Retained", f"{metrics['possession_retained_rate']:.1f}%")
    
    # Analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⏱️ Set Pieces by Game Minute")
        fig, ax = plot_minute_distribution(filtered_df)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### 🎯 Delivery Type Analysis")
        if 'delivery_type' in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            delivery_counts = filtered_df['delivery_type'].value_counts()
            ax.barh(delivery_counts.index, delivery_counts.values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(delivery_counts))))
            ax.set_xlabel('Count')
            ax.set_title('Delivery Types', fontweight='bold')
            st.pyplot(fig)
            plt.close()
    
    # Danger index distribution
    if 'danger_index' in filtered_df.columns:
        st.markdown("#### 🔥 Danger Index Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(filtered_df['danger_index'], bins=30, color='red', alpha=0.7, edgecolor='white')
        ax.axvline(filtered_df['danger_index'].mean(), color='black', linestyle='--', label=f'Mean: {filtered_df["danger_index"].mean():.1f}')
        ax.set_xlabel('Danger Index')
        ax.set_ylabel('Frequency')
        ax.set_title('Set Piece Danger Index Distribution', fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close()


def show_predictions(df):
    """Predictions page"""
    st.title("🎯 First Receiver Prediction")
    
    st.markdown("""
    Use this tool to predict which player will be the first receiver of a set piece.
    Configure the scenario below and get AI-powered predictions!
    """)
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Set Piece Configuration")
        
        set_piece_type = st.selectbox("Type", ["corner", "free_kick", "throw_in"])
        side = st.selectbox("Side", ["left", "right"])
        delivery_type = st.selectbox("Delivery", ["inswinger", "outswinger", "short", "driven", "lofted"])
        minute = st.slider("Game Minute", 1, 90, 45)
    
    with col2:
        st.markdown("### Players in Box")
        
        n_attackers = st.slider("Attackers in Box", 2, 8, 5)
        n_defenders = st.slider("Defenders in Box", 3, 10, 6)
        score_diff = st.slider("Score Difference", -3, 3, 0)
    
    # Position on pitch
    if set_piece_type == "corner":
        x = 0 if side == "left" else 100
        y = 0 if side == "left" else 100
    else:
        x = st.slider("X Position (0-100)", 0, 100, 75)
        y = st.slider("Y Position (0-100)", 0, 100, 50)
    
    if st.button("🎯 Predict First Receiver", type="primary"):
        with st.spinner("Running prediction..."):
            # Simulate prediction (in real implementation, use trained model)
            np.random.seed(int(minute + n_attackers * 10))
            
            # Calculate simple probabilities based on position
            base_probs = np.random.dirichlet(np.ones(n_attackers) * 2)
            
            # Favor near-post for inswingers
            if delivery_type == "inswinger":
                base_probs[0] += 0.1
            elif delivery_type == "outswinger":
                base_probs[-1] += 0.1
            
            # Normalize
            probs = base_probs / base_probs.sum()
            predicted_receiver = np.argmax(probs)
            confidence = probs[predicted_receiver]
            
            # Display result
            st.success(f"### Predicted First Receiver: **Player #{predicted_receiver + 1}**")
            st.markdown(f"**Confidence:** {confidence:.1%}")
            
            # Show probability distribution
            st.markdown("#### Receiver Probabilities")
            
            prob_df = pd.DataFrame({
                'Player': [f'Player {i+1}' for i in range(n_attackers)],
                'Probability': probs
            })
            
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['gold' if i == predicted_receiver else 'steelblue' for i in range(n_attackers)]
            ax.bar(prob_df['Player'], prob_df['Probability'], color=colors)
            ax.set_ylabel('Probability')
            ax.set_title('First Receiver Probability Distribution', fontweight='bold')
            
            for i, (p, prob) in enumerate(zip(prob_df['Player'], prob_df['Probability'])):
                ax.text(i, prob + 0.02, f'{prob:.1%}', ha='center', fontsize=10)
            
            st.pyplot(fig)
            plt.close()
            
            # Visualize on pitch
            st.markdown("#### Set Piece Visualization")
            
            # Generate positions
            np.random.seed(42)
            attackers = [
                {'x': np.random.uniform(80, 95), 'y': np.random.uniform(30, 70), 'player_id': i+1}
                for i in range(n_attackers)
            ]
            defenders = [
                {'x': np.random.uniform(82, 98), 'y': np.random.uniform(25, 75), 'player_id': i+1}
                for i in range(n_defenders)
            ]
            
            set_piece_data = {
                'x': x,
                'y': y,
                'type': set_piece_type,
                'minute': minute,
                'attacker_positions': attackers,
                'defender_positions': defenders,
                'outcome': 'pending'
            }
            
            predictions = {
                'predicted_receiver': predicted_receiver,
                'probability': confidence
            }
            
            fig, ax = plot_set_piece(set_piece_data, predictions=predictions)
            st.pyplot(fig)
            plt.close()


def show_heatmaps(df):
    """Heatmaps page"""
    st.title("🗺️ Interactive Heatmaps")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        teams = ['All Teams'] + sorted(df['team'].unique().tolist())
        selected_team = st.selectbox("Team", teams)
    
    with col2:
        heatmap_type = st.selectbox("Heatmap Type", ["receiver", "success", "danger"])
    
    # Apply team filter
    if selected_team != 'All Teams':
        filtered_df = df[df['team'] == selected_team]
        title = f"{selected_team} Set Pieces"
    else:
        filtered_df = df
        title = "All Teams Set Pieces"
    
    # Display heatmap
    fig, ax = plot_heatmap(filtered_df, zone_type=heatmap_type, 
                          title=f"{title} - {heatmap_type.title()} Heatmap")
    st.pyplot(fig)
    plt.close()
    
    # Additional heatmaps by type
    st.markdown("### By Set Piece Type")
    
    types = filtered_df['type'].unique()
    cols = st.columns(min(3, len(types)))
    
    for i, sp_type in enumerate(types[:3]):
        type_df = filtered_df[filtered_df['type'] == sp_type]
        with cols[i]:
            st.markdown(f"**{sp_type.title()}**")
            fig, ax = plot_heatmap(type_df, zone_type='receiver')
            st.pyplot(fig)
            plt.close()


def show_statistics(df):
    """Statistics page"""
    st.title("📈 Detailed Statistics")
    
    # Team comparison
    st.markdown("### 🏆 Team Rankings")
    
    team_stats = df.groupby('team').agg({
        'event_id': 'count',
        'outcome': lambda x: (x == 'goal').sum(),
    }).rename(columns={'event_id': 'total', 'outcome': 'goals'})
    
    team_stats['goal_rate'] = team_stats['goals'] / team_stats['total'] * 100
    team_stats['success_rate'] = df.groupby('team')['outcome'].apply(
        lambda x: x.isin(['goal', 'shot']).mean() * 100
    )
    
    team_stats = team_stats.sort_values('goal_rate', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(team_stats.index[:10], team_stats['goal_rate'].head(10), 
                   color=plt.cm.RdYlGn(np.linspace(0.8, 0.2, 10)))
    ax.set_xlabel('Goal Rate (%)')
    ax.set_title('Top 10 Teams by Set Piece Goal Rate', fontweight='bold')
    
    for bar, rate in zip(bars, team_stats['goal_rate'].head(10)):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
               f'{rate:.1f}%', va='center')
    
    st.pyplot(fig)
    plt.close()
    
    # Detailed stats table
    st.markdown("### 📊 Team Statistics Table")
    
    display_df = team_stats.reset_index()
    display_df.columns = ['Team', 'Total Set Pieces', 'Goals', 'Goal Rate (%)', 'Success Rate (%)']
    display_df = display_df.round(2)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Stage analysis
    st.markdown("### 🏟️ By Tournament Stage")
    
    if 'stage' in df.columns:
        stage_stats = df.groupby('stage').agg({
            'event_id': 'count',
            'outcome': lambda x: (x.isin(['goal', 'shot'])).mean() * 100
        }).rename(columns={'event_id': 'total', 'outcome': 'success_rate'})
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(stage_stats.index, stage_stats['success_rate'], color='steelblue')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Set Piece Success Rate by Tournament Stage', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close()


def show_designer(df):
    """Set piece designer page"""
    st.title("🎮 Set-Piece Designer Tool")
    
    st.markdown("""
    ### Design Your Own Set Piece Scenario
    
    Use this interactive tool to design set piece formations and analyze their effectiveness.
    """)
    
    # Interactive pitch display
    st.markdown("### ⚽ Position Your Players")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Draw base pitch
        fig, ax = draw_pitch()
        
        # Add sample positions
        np.random.seed(123)
        
        # Ball position
        ball_x = st.slider("Ball X Position", 0, 100, 0) / 100 * PITCH_LENGTH
        ball_y = st.slider("Ball Y Position", 0, 100, 50) / 100 * PITCH_WIDTH
        
        ax.plot(ball_x, ball_y, 'o', color='white', markersize=15, 
                markeredgecolor='black', markeredgewidth=2, zorder=10)
        
        # Sample attacker positions
        n_att = st.slider("Number of Attackers", 3, 8, 5)
        for i in range(n_att):
            att_x = 70 + np.random.uniform(10, 30)
            att_y = 20 + np.random.uniform(0, 28) + i * 5
            ax.plot(att_x / 100 * PITCH_LENGTH, att_y / 100 * PITCH_WIDTH, 'o', color='red', 
                   markersize=12, markeredgecolor='white', markeredgewidth=1)
        
        # Sample defender positions
        n_def = st.slider("Number of Defenders", 4, 10, 6)
        for i in range(n_def):
            def_x = 75 + np.random.uniform(10, 22)
            def_y = 18 + np.random.uniform(0, 32) + i * 4
            ax.plot(def_x / 100 * PITCH_LENGTH, def_y / 100 * PITCH_WIDTH, 's', color='blue', 
                   markersize=10, markeredgecolor='white', markeredgewidth=1)
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### ⚙️ Settings")
        
        delivery = st.selectbox("Delivery Type", ["inswinger", "outswinger", "short", "driven", "lofted"])
        target_zone = st.selectbox("Target Zone", ["near_post", "far_post", "penalty_spot", "edge_of_box"])
        
        st.markdown("---")
        st.markdown("### 📊 Quick Analysis")
        
        # Simple analysis based on configuration
        success_prob = 0.15 + (n_att - n_def) * 0.05
        success_prob = max(0.05, min(0.40, success_prob))
        
        st.metric("Estimated Success Rate", f"{success_prob:.1%}")
        
        if delivery == "inswinger":
            rec = "Near post target recommended"
        elif delivery == "short":
            rec = "Consider switch play"
        else:
            rec = "Far post offers space"
        
        st.info(f"💡 **Tip:** {rec}")
    
    st.markdown("---")
    st.markdown("""
    ### 🚧 Coming Soon
    
    - **Drag and drop** player positioning
    - **Historical comparison** with similar scenarios
    - **Animated preview** of set piece execution
    - **Export** your designs
    """)


if __name__ == "__main__":
    main()
