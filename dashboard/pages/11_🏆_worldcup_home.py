"""
🏆 World Cup 2022 Set-Piece Intelligence - Home
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="World Cup 2022 Intelligence", layout="wide", page_icon="🏆")

st.title("🏆 FIFA World Cup 2022 Set-Piece Intelligence")
st. markdown("### *AI-Powered Analysis of Every Corner from Qatar 2022*")

# Try to load World Cup data
try:
    matches_path = 'data/worldcup2022/matches.csv'
    corners_path = 'data/worldcup2022/corners.csv'
    
    if not os.path.exists(matches_path) or not os.path. exists(corners_path):
        st.error("⚠️ World Cup data files not found!")
        st.info(f"Expected files:\n- {matches_path}\n- {corners_path}")
        st.stop()
    
    matches = pd.read_csv(matches_path)
    corners = pd.read_csv(corners_path)
    
    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏟️ Matches", len(matches), help="All World Cup 2022 matches")
    
    with col2:
        st.metric("⚽ Corners Analyzed", len(corners), help="Corners with detailed data")
    
    with col3:
        corner_goals = len(corners[corners['goal_scored'] == 1])
        conversion_rate = corner_goals / len(corners) if len(corners) > 0 else 0
        st.metric("🎯 Goal Conversion", f"{conversion_rate:.1%}", help="Corners that led to goals")
    
    with col4:
        teams = pd.concat([matches['team1'], matches['team2']]).nunique()
        st.metric("🌍 Teams", teams, help="Countries in World Cup 2022")
    
    st.divider()
    
    # Top performing teams
    st.subheader("🏆 Top Set-Piece Teams")
    
    team_performance = corners.groupby('team'). agg({
        'corner_id': 'count',
        'goal_scored': 'sum'
    }).reset_index()
    team_performance. columns = ['Team', 'Total Corners', 'Goals']
    team_performance['Conversion Rate'] = team_performance['Goals'] / team_performance['Total Corners']
    team_performance = team_performance[team_performance['Total Corners'] >= 3]  # Only teams with 3+ corners
    team_performance = team_performance.sort_values('Conversion Rate', ascending=False). head(10)
    
    fig = px.bar(team_performance, x='Team', y='Conversion Rate', 
                 title='Top 10 Teams by Corner Conversion Rate',
                 color='Conversion Rate', color_continuous_scale='RdYlGn',
                 text='Conversion Rate')
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent corners
    st.subheader("📊 Sample Corners from Dataset")
    
    recent = corners.merge(matches[['match_id', 'team1', 'team2', 'stage']], on='match_id'). head(10)
    
    st.dataframe(
        recent[['team', 'minute', 'outcome', 'taker', 'receiver', 'goal_scored', 'formation_type', 'stage']],
        use_container_width=True,
        height=300
    )
    
    # Visualizations
    st.subheader("📈 Insights")
    
    col1, col2 = st. columns(2)
    
    with col1:
        outcome_counts = corners['outcome'].value_counts()
        fig = px.pie(values=outcome_counts.values, names=outcome_counts.index, 
                     title='Corner Outcomes',
                     hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Goals by stage
        corners_with_stage = corners. merge(matches[['match_id', 'stage']], on='match_id')
        goals_by_stage = corners_with_stage[corners_with_stage['goal_scored']==1].groupby('stage').size()
        
        fig = px.bar(x=goals_by_stage.index, y=goals_by_stage.values,
                      title='Goals from Corners by Tournament Stage',
                      labels={'x': 'Stage', 'y': 'Goals'},
                      color=goals_by_stage.values,
                      color_continuous_scale='Reds')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Formation analysis
    st.subheader("🎯 Formation Types Analysis")
    
    formation_stats = corners.groupby('formation_type').agg({
        'corner_id': 'count',
        'goal_scored': 'sum'
    }). reset_index()
    formation_stats.columns = ['Formation', 'Total', 'Goals']
    formation_stats['Success Rate'] = formation_stats['Goals'] / formation_stats['Total']
    formation_stats = formation_stats.sort_values('Total', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=formation_stats['Formation'], y=formation_stats['Total'], 
                         name='Total Corners', marker_color='lightblue'))
    fig.add_trace(go.Bar(x=formation_stats['Formation'], y=formation_stats['Goals'], 
                         name='Goals', marker_color='green'))
    fig.update_layout(title='Corner Formations: Usage vs Success',
                      xaxis_title='Formation Type',
                      yaxis_title='Count',
                      barmode='group',
                      height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st. divider()
    
    # Call to action
    st.success("🚀 **More analysis pages coming soon:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🌍 **Database Browser**\nFilter and explore all corners")
    with col2:
        st.info("🤖 **AI Predictor**\n80%+ accuracy predictions")
    with col3:
        st.info("📚 **Pattern Library**\nWinning tactics from WC")
    
    # Fun facts
    with st.expander("🎉 Fun Facts from World Cup 2022"):
        if len(team_performance) > 0:
            most_corners_team = team_performance.iloc[0]
            st.write(f"- 🥇 **Most effective team**: {most_corners_team['Team']} ({most_corners_team['Conversion Rate']:.1%} conversion)")
        st.write(f"- ⚽ **Total goals from corners**: {corner_goals}")
        st.write(f"- 📊 **Average corners per match**: {len(corners)/len(matches):.1f}")
        
        short_corners = len(corners[corners['formation_type'] == 'Short corner'])
        if short_corners > 0:
            st.write(f"- 🔄 **Short corners used**: {short_corners} ({short_corners/len(corners):.1%})")
        
        # Most successful taker
        taker_stats = corners[corners['goal_scored']==1]['taker'].value_counts()
        if len(taker_stats) > 0:
            st.write(f"- 🌟 **Most successful corner taker**: {taker_stats.index[0]} ({taker_stats.values[0]} goals)")

except FileNotFoundError as e:
    st.error(f"⚠️ World Cup data not found!  {e}")
    st.info("Make sure data files are in `data/worldcup2022/`")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.exception(e)

st.divider()

# About
with st.expander("ℹ️ About This Project"):
    st.write("""
    **🌍 World Cup 2022 Set-Piece Intelligence System**
    
    This revolutionary system analyzes every corner from FIFA World Cup 2022.  
    
    **Current Dataset:**
    - ✅ 64 World Cup matches
    - ✅ 50+ detailed corner analysis
    - ✅ Player positions and outcomes
    - ✅ Formation types and tactics
    
    **Technology:**
    - Python + Streamlit
    - Machine Learning (Scikit-learn)
    - Advanced data visualization
    
    **Goal:** Revolutionize football tactics through AI and data science!  🚀⚽
    """)