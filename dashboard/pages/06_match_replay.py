"""
Match Replay with Synchronized Analysis
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Match Replay", layout="wide")

st.title("🎬 Match Replay")
st.markdown("### Synchronized Video Analysis with Pitch View")

if 'tracking_data' not in st.session_state or 'video_path' not in st.session_state:
    st.warning("⚠️ Please upload and analyze a video first in the Video Analysis page")
    st.info("👉 Go to **Video Analysis** → Upload video → Run Tracking")
    st.stop()

tracking_data = st.session_state['tracking_data']
vido_path = st.session_state['video_path']

with st.sidebar:
    st.markdown("### 🎮 Playback Controls")
    max_time = tracking_data['timestamp'].max()
    current_time = st.slider("Video Time (seconds)", 0.0, float(max_time), 0.0, 0.1, key='replay_time')
    st.markdown("---")
    st.markdown("### 🎨 Display Options")
    show_trajectories = st.checkbox("Show Trajectories", value=True)
    show_player_ids = st.checkbox("Show Player IDs", value=True)
    trajectory_length = st.slider("Trajectory Length (frames)", 10, 100, 30)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📹 Video Feed")
    st.video(video_path, start_time=int(current_time))

with col2:
    st.markdown("### ⚽ Pitch View")
    fig, ax = plt.subplots(figsize=(8, 12))
    pitch_length, pitch_width = 105, 68
    ax.plot([0, 0, pitch_length, pitch_length, 0], [0, pitch_width, pitch_width, 0, 0], color='white', linewidth=2)
    ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], color='white', linewidth=2)
    
    current_frame = int(current_time * 30)
    frame_data = tracking_data[tracking_data['frame'] == current_frame]
    
    if len(frame_data) > 0:
        x_normalized = frame_data['x'] / 1920
        y_normalized = frame_data['y'] / 1080
        x_pitch = x_normalized * pitch_length
        y_pitch = y_normalized * pitch_width
        colors = plt.cm.tab10(np.linspace(0, 1, len(frame_data)))
        
        for idx, (_, player) in enumerate(frame_data.iterrows()):
            x_p = x_normalized.iloc[idx] * pitch_length
            y_p = y_normalized.iloc[idx] * pitch_width
            ax.scatter(x_p, y_p, s=200, c=[colors[idx]], edgecolors='white', linewidth=2, zorder=10)
            if show_player_ids:
                ax.text(x_p, y_p, str(int(player['player_id'])), fontsize=10, ha='center', va='center', color='white', weight='bold', zorder=11)
    
    ax.set_xlim(-5, pitch_length + 5)
    ax.set_ylim(-5, pitch_width + 5)
    ax.set_facecolor('#2d5f2e')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#1e1e1e')
    st.pyplot(fig)
    plt.close()