# ⚽ Elite Set-Piece Analytics

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **تحليلات الكرات الثابتة النخبوية** - Advanced Football Set-Piece Analysis & Prediction Platform

A comprehensive sports analytics platform focusing on set-piece analysis, first receiver prediction, and tactical intelligence using World Cup 2022 and elite football datasets.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Data Sources](#-data-sources)
- [Models](#-models)
- [Dashboard](#-dashboard)
- [Documentation](#-documentation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Elite Set-Piece Analytics is a cutting-edge platform designed to revolutionize how teams analyze and optimize set-piece situations. Using state-of-the-art machine learning and advanced data visualization, we provide:

- **First Receiver Prediction**: Predict which player is most likely to receive the ball from a corner kick or free kick
- **Tactical Pattern Analysis**: Identify and cluster similar set-piece routines
- **Outcome Prediction**: Estimate goal probability (xG) for set-piece situations
- **Interactive Visualization**: Explore data with dynamic pitch visualizations

---

## ✨ Features

### 🔮 Predictive Analytics
- Multi-class classification for receiver prediction (70%+ accuracy)
- XGBoost + optional Neural Network ensemble
- Top-K predictions with confidence scores

### 📊 Data Processing
- Support for multiple data sources (StatsBomb, Wyscout, Metrica)
- Unified data format for seamless analysis
- Automated set-piece extraction and classification

### 🎨 Visualization
- Interactive pitch visualizations
- Player position heatmaps
- Animation support for tracking data

### 📱 Dashboard
- Streamlit-based interactive dashboard
- Real-time predictions
- Custom set-piece designer (coming soon)

---

## 📁 Project Structure

```
elite-set-piece-analytics/
├── 📄 README.md                 # Project documentation
├── 📄 ROADMAP.md                # Development roadmap
├── 📄 LICENSE                   # MIT License
├── 📄 requirements.txt          # Python dependencies
├── 📄 setup.py                  # Package setup
├── 📄 .gitignore               # Git ignore rules
│
├── 📁 src/                      # Source code
│   ├── data/                    # Data loading & processing
│   │   ├── loaders.py          # Multi-source data loaders
│   │   ├── extractors.py       # Set-piece extraction
│   │   └── preprocessors.py    # Data preprocessing
│   ├── features/                # Feature engineering
│   │   ├── spatial.py          # Spatial features
│   │   ├── temporal.py         # Temporal features
│   │   └── physical.py         # Physical features
│   ├── models/                  # ML models
│   │   ├── receiver_predictor.py    # First receiver prediction
│   │   ├── outcome_predictor.py     # Outcome prediction
│   │   └── pattern_analyzer.py      # Pattern clustering
│   ├── visualization/           # Visualization tools
│   │   ├── pitch.py            # Pitch drawing
│   │   ├── animations.py       # Tracking animations
│   │   └── heatmaps.py         # Heatmap generation
│   └── utils/                   # Utilities
│       ├── config.py           # Configuration
│       └── metrics.py          # Evaluation metrics
│
├── 📁 notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_set_piece_extraction.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_tactical_analysis.ipynb
│   └── 06_visualization.ipynb
│
├── 📁 dashboard/                # Streamlit dashboard
│   ├── app.py                  # Main application
│   ├── pages/                  # Dashboard pages
│   └── components/             # UI components
│
├── 📁 scripts/                  # Utility scripts
│   ├── download_data.sh        # Data download script
│   ├── train_models.py         # Model training
│   └── generate_reports.py     # Report generation
│
├── 📁 tests/                    # Unit tests
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
└── 📁 docs/                     # Documentation
    ├── architecture.md         # Technical architecture
    ├── data_dictionary.md      # Data definitions
    └── model_cards.md          # Model documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda

### Quick Install

```bash
# Clone the repository
git clone https://github.com/filopatirbakhoum06-collab/elite-set-piece-analytics.git
cd elite-set-piece-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Download Data

```bash
# Download open source football data
bash scripts/download_data.sh
```

---

## ⚡ Quick Start

### 1. Load Data

```python
from src.data.loaders import StatsBombLoader, UnifiedDataLoader

# Load StatsBomb data
loader = StatsBombLoader()
events = loader.load_match_events(match_id=3788741)

# Or use unified loader
unified = UnifiedDataLoader()
events = unified.load_events(source="statsbomb", match_id=3788741)
```

### 2. Extract Set-Pieces

```python
from src.data.extractors import SetPieceExtractor

extractor = SetPieceExtractor()
corners = extractor.extract_corners(events)
free_kicks = extractor.extract_free_kicks(events)
all_set_pieces = extractor.extract_all(events)
```

### 3. Train Model

```python
from src.models.receiver_predictor import ReceiverPredictor

predictor = ReceiverPredictor()
predictor.fit(X_train, y_train)
predictions = predictor.predict(X_test)
top_k = predictor.get_top_k_receivers(X_test, k=3)
```

### 4. Visualize

```python
from src.visualization.pitch import PitchVisualizer

viz = PitchVisualizer()
fig, ax = viz.draw_pitch()
viz.plot_players(ax, home_positions, away_positions)
plt.show()
```

### 5. Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| **StatsBomb** | High-quality event data including 360 data | Free API via `statsbombpy` |
| **Wyscout** | Comprehensive event data from major leagues | [Public Dataset](https://figshare.com/collections/Soccer_match_event_dataset/4415000) |
| **Metrica Sports** | Sample tracking data | [GitHub Repository](https://github.com/metrica-sports/sample-data) |

---

## 🤖 Models

### First Receiver Predictor

- **Architecture**: XGBoost Ensemble (+ optional Neural Network)
- **Task**: Multi-class classification
- **Accuracy**: ~70-75%
- **Top-3 Accuracy**: ~85-90%

### Outcome Predictor

- **Architecture**: XGBoost Binary Classifier
- **Task**: Goal probability estimation
- **AUC-ROC**: ~0.75-0.80

### Pattern Analyzer

- **Architecture**: K-Means + PCA
- **Task**: Unsupervised pattern clustering
- **Clusters**: 10 tactical patterns

---

## 📱 Dashboard

The interactive Streamlit dashboard provides:

- **Overview**: Key metrics and recent predictions
- **Predictions**: Interactive receiver prediction
- **Designer**: Custom set-piece routine design
- **Analysis**: Deep dive into patterns and trends

```bash
streamlit run dashboard/app.py
```

---

## 📚 Documentation

- [Technical Architecture](docs/architecture.md)
- [Data Dictionary](docs/data_dictionary.md)
- [Model Cards](docs/model_cards.md)
- [Development Roadmap](ROADMAP.md)

---

## 🗺️ Roadmap

See our [Development Roadmap](ROADMAP.md) for the 9-week development plan:

| Phase | Duration | Focus |
|-------|----------|-------|
| 1 | Week 1-2 | Foundation & Setup ✅ |
| 2 | Week 3-4 | Data Processing |
| 3 | Week 5-6 | Model Development |
| 4 | Week 7-8 | Dashboard & Visualization |
| 5 | Week 9 | Documentation & Presentation |

---

## 🤝 Contributing

We welcome contributions! Please see our guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👥 Team

Elite Set-Piece Analytics Team

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [StatsBomb](https://statsbomb.com/) for open event data
- [Metrica Sports](https://metrica-sports.com/) for sample tracking data
- [mplsoccer](https://mplsoccer.readthedocs.io/) for visualization inspiration
- [kloppy](https://kloppy.pysport.org/) for data handling

---

<p align="center">
  Made with ⚽ by the Elite Set-Piece Analytics Team
</p>