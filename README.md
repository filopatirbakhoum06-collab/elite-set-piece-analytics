# ⚽ Elite Set-Piece Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Advanced analytics and AI-powered predictions for football set pieces, featuring FIFA World Cup 2022 data.

![Dashboard Preview](docs/dashboard_preview.png)

---

## 🎯 Overview

The **Elite Set-Piece Analytics Platform** provides comprehensive analysis of football set pieces using machine learning and advanced statistics. Built with data from the FIFA World Cup 2022, this platform enables:

- 📊 **Statistical Analysis** - Deep dive into set piece patterns and trends
- 🎯 **AI Predictions** - Predict first receiver and outcomes using XGBoost
- 🗺️ **Interactive Heatmaps** - Visualize tactical patterns on pitch
- 📈 **Team Comparisons** - Compare set piece effectiveness across teams
- 🎮 **Set Piece Designer** - Design and analyze custom scenarios

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/filopatirbakhoum06-collab/elite-set-piece-analytics.git
cd elite-set-piece-analytics

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.data.loaders import load_wyscout_data; print('✅ Ready!')"
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501` 🎉

### Quick Analysis

```python
from src.data.loaders import load_wyscout_data
from src.data.extractors import extract_set_pieces, calculate_success_metrics

# Load World Cup 2022 data
data = load_wyscout_data()
set_pieces = extract_set_pieces(data)

# Get insights
metrics = calculate_success_metrics(set_pieces)
print(f"Goal Rate: {metrics['goal_rate']:.2f}%")
print(f"Success Rate: {metrics['success_rate']:.2f}%")
```

---

## 📁 Project Structure

```
elite-set-piece-analytics/
├── src/
│   ├── data/                # Data loading and extraction
│   │   ├── loaders.py       # Load from Wyscout/StatsBomb
│   │   ├── extractors.py    # Extract set pieces
│   │   └── preprocessors.py # Clean and prepare data
│   ├── features/            # Feature engineering
│   │   ├── spatial.py       # Spatial features (distance, angle)
│   │   ├── temporal.py      # Time-based features
│   │   └── physical.py      # Player physical attributes
│   ├── models/              # Machine learning models
│   │   ├── receiver_predictor.py  # First receiver prediction
│   │   └── outcome_predictor.py   # Outcome prediction
│   └── visualization/       # Plotting and visualization
│       ├── pitch.py         # Pitch drawing and heatmaps
│       └── animations.py    # Animated visualizations
├── dashboard/               # Streamlit dashboard
│   └── app.py               # Main dashboard application
├── notebooks/               # Jupyter notebooks
│   ├── 01_quick_start.ipynb
│   ├── 02_full_analysis.ipynb
│   └── 03_tactical_insights.ipynb
├── scripts/                 # Utility scripts
│   ├── train_all_models.py
│   └── generate_report.py
├── tests/                   # Unit tests
├── docs/                    # Documentation
│   ├── QUICKSTART.md
│   └── RESULTS.md
├── data/                    # Data files
├── models/                  # Saved models
└── requirements.txt         # Dependencies
```

---

## 📊 Features

### Data Pipeline
- Load Wyscout World Cup 2022 data
- StatsBomb integration support
- Automatic data generation for demos
- Robust error handling

### Feature Engineering
- **Spatial Features**: Distance to goal, angle to goal, pitch zones, danger index
- **Temporal Features**: Game phase, fatigue factor, momentum, score pressure
- **Physical Features**: Height advantage, header probability, delivery speed

### Machine Learning Models
- **First Receiver Predictor**: XGBoost classifier (71% accuracy)
- **Outcome Predictor**: Binary and multi-class prediction
- Model persistence and loading
- Feature importance analysis

### Visualization
- FIFA-standard pitch drawing
- Interactive heatmaps (receiver, success, danger zones)
- Team comparison charts
- Animated set piece visualization

---

## 🎮 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview with key metrics and visualizations |
| 📊 Analysis | Filtered analysis by team and set piece type |
| 🎯 Predictions | AI-powered first receiver prediction |
| 🗺️ Heatmaps | Interactive zone analysis |
| 📈 Statistics | Team rankings and comparisons |
| 🎮 Designer | Design custom set piece scenarios |

---

## 📈 Results

### Model Performance

| Model | Accuracy | Top-3 Accuracy | F1-Score |
|-------|----------|----------------|----------|
| First Receiver | 71% | 89% | 0.68 |
| Outcome (Binary) | 72% | - | 0.70 |
| Outcome (Multi) | 65% | - | 0.63 |

### Key Insights

1. **Goal Rate**: ~3% of set pieces result in goals
2. **Success Rate**: ~18% result in goals or shots
3. **Optimal Delivery**: Inswingers 2.3x more effective
4. **Late Game**: 75+ minute set pieces more dangerous

See [docs/RESULTS.md](docs/RESULTS.md) for full analysis.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src
```

---

## 📚 Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [Results & Insights](docs/RESULTS.md)
- [API Documentation](src/README.md)

---

## 🛠️ Technologies

- **Python 3.8+**
- **Data**: pandas, numpy
- **ML**: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn, mplsoccer
- **Dashboard**: Streamlit, Plotly
- **Football Data**: statsbombpy

---

## 👥 Team

Created with ❤️ for football analytics.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Wyscout](https://wyscout.com/) for World Cup 2022 data
- [StatsBomb](https://statsbomb.com/) for open data
- [mplsoccer](https://mplsoccer.readthedocs.io/) for visualization inspiration

---

**⚽ Ready to analyze set pieces like never before!**