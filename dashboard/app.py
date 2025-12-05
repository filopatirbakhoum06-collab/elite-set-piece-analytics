"""
Elite Set-Piece Analytics Dashboard
====================================

Main Streamlit application for the Elite Set-Piece Analytics platform.
This dashboard provides interactive visualizations and predictions
for football set-piece analysis.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Main dashboard application."""
    # Page configuration
    st.set_page_config(
        page_title="Elite Set-Piece Analytics",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80?text=Elite+Analytics", width=200)
        st.markdown("---")
        
        st.markdown("## 📊 Navigation")
        page = st.radio(
            "Select a page:",
            ["🏠 Overview", "🎯 Predictions", "🎨 Designer", "📈 Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📁 Data Source")
        data_source = st.selectbox(
            "Select data source:",
            ["World Cup 2022", "StatsBomb Open Data", "Sample Data"]
        )
        
        st.markdown("---")
        st.markdown("""
        ### 👥 Team
        Elite Set-Piece Analytics Team
        
        [📚 Documentation](https://github.com/filopatirbakhoum06-collab/elite-set-piece-analytics)
        
        [🐛 Report Bug](https://github.com/filopatirbakhoum06-collab/elite-set-piece-analytics/issues)
        """)
    
    # Main content based on selected page
    if page == "🏠 Overview":
        render_overview_page()
    elif page == "🎯 Predictions":
        render_predictions_page()
    elif page == "🎨 Designer":
        render_designer_page()
    elif page == "📈 Analysis":
        render_analysis_page()


def render_overview_page():
    """Render the overview page."""
    st.markdown('<p class="main-header">⚽ Elite Set-Piece Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Football Set-Piece Analysis & Prediction Platform</p>', unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Set-Pieces", value="1,247", delta="↑ 12%")
    with col2:
        st.metric(label="Prediction Accuracy", value="73.5%", delta="↑ 2.1%")
    with col3:
        st.metric(label="Matches Analyzed", value="64", delta=None)
    with col4:
        st.metric(label="Teams Covered", value="32", delta=None)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Set-Piece Distribution")
        # Placeholder chart data
        chart_data = pd.DataFrame({
            'Type': ['Corners', 'Free Kicks', 'Throw-ins', 'Penalties'],
            'Count': [450, 380, 350, 67]
        })
        st.bar_chart(chart_data.set_index('Type'))
    
    with col2:
        st.markdown("### 🎯 Prediction Performance")
        # Placeholder chart data
        perf_data = pd.DataFrame({
            'Model': ['XGBoost', 'Neural Net', 'Ensemble'],
            'Accuracy': [0.71, 0.68, 0.74]
        })
        st.bar_chart(perf_data.set_index('Model'))
    
    st.markdown("---")
    
    # Recent Activity
    st.markdown("### 📋 Recent Set-Pieces Analyzed")
    recent_data = pd.DataFrame({
        'Match': ['Argentina vs France', 'Argentina vs Croatia', 'France vs Morocco'],
        'Type': ['Corner', 'Free Kick', 'Corner'],
        'Predicted Receiver': ['Messi', 'Álvarez', 'Giroud'],
        'Actual Receiver': ['Messi', 'Álvarez', 'Kolo Muani'],
        'Correct': ['✅', '✅', '❌']
    })
    st.dataframe(recent_data, use_container_width=True)


def render_predictions_page():
    """Render the predictions page."""
    st.markdown("## 🎯 First Receiver Predictions")
    st.markdown("Predict the most likely first receiver for a set-piece delivery.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        set_piece_type = st.selectbox(
            "Set-Piece Type",
            ["Corner Kick", "Free Kick", "Throw-in"]
        )
        
        delivery_type = st.selectbox(
            "Delivery Type",
            ["Inswinging", "Outswinging", "Driven", "Floated"]
        )
        
        team = st.selectbox(
            "Attacking Team",
            ["Argentina", "France", "Morocco", "Croatia"]
        )
        
        match = st.selectbox(
            "Match",
            ["Final", "Semi-final", "Quarter-final", "Round of 16"]
        )
        
        if st.button("🔮 Predict Receiver", use_container_width=True):
            st.success("Prediction complete! See results below.")
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        # Placeholder predictions
        predictions = pd.DataFrame({
            'Player': ['Messi', 'Álvarez', 'Fernández', 'Romero', 'Di María'],
            'Position': ['RW', 'ST', 'MF', 'CB', 'LW'],
            'Probability': [0.32, 0.25, 0.18, 0.15, 0.10]
        })
        
        # Create bar chart
        st.bar_chart(predictions.set_index('Player')['Probability'])
        
        st.markdown("#### Top Predictions")
        for idx, row in predictions.head(3).iterrows():
            st.write(f"**{idx + 1}. {row['Player']}** ({row['Position']}) - {row['Probability']:.1%}")


def render_designer_page():
    """Render the set-piece designer page."""
    st.markdown("## 🎨 Set-Piece Designer")
    st.markdown("Design and analyze custom set-piece routines.")
    
    st.info("🚧 This feature is under development. Check back soon!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎮 Controls")
        
        st.selectbox("Template", ["Blank", "Near Post", "Far Post", "Short Corner"])
        st.slider("Number of Attackers", 1, 11, 6)
        st.slider("Number of Defenders", 1, 11, 5)
        st.checkbox("Show Zones")
        st.checkbox("Show Heat Map")
    
    with col2:
        st.markdown("### ⚽ Pitch View")
        st.image("https://via.placeholder.com/600x400?text=Interactive+Pitch+Designer", 
                use_container_width=True)


def render_analysis_page():
    """Render the analysis page."""
    st.markdown("## 📈 Deep Analysis")
    st.markdown("Explore detailed analytics and patterns in set-piece data.")
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🔍 Patterns", "📉 Trends"])
    
    with tab1:
        st.markdown("### Set-Piece Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### By Type")
            type_stats = pd.DataFrame({
                'Type': ['Corner', 'Free Kick', 'Throw-in', 'Penalty'],
                'Total': [450, 380, 350, 67],
                'Goals': [23, 18, 5, 58],
                'Conversion %': ['5.1%', '4.7%', '1.4%', '86.6%']
            })
            st.dataframe(type_stats, use_container_width=True)
        
        with col2:
            st.markdown("#### By Team")
            team_stats = pd.DataFrame({
                'Team': ['Argentina', 'France', 'Morocco', 'Croatia'],
                'Set-Pieces': [89, 95, 72, 78],
                'Goals': [8, 7, 3, 4],
                'xG': [6.2, 5.8, 4.1, 4.5]
            })
            st.dataframe(team_stats, use_container_width=True)
    
    with tab2:
        st.markdown("### Pattern Analysis")
        st.info("Pattern analysis coming soon!")
    
    with tab3:
        st.markdown("### Trend Analysis")
        st.info("Trend analysis coming soon!")


if __name__ == "__main__":
    main()
